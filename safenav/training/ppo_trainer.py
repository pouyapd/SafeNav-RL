"""
ppo_trainer.py

Proximal Policy Optimization (PPO) training loop.

Key ideas:
  1. Collect rollouts with the current policy
  2. Compute advantages (GAE)
  3. Update the policy with the clipped surrogate objective
  4. Update the value function by regression on the returns

When the safety filter is enabled it is applied inside the environment
(SafetyFilterWrapper). The buffer stores the policy's own sampled action and
its log-probability, so the PPO probability ratio is computed for the action
the policy actually chose; the filter is part of the environment dynamics from
the agent's point of view.

Reference:
  Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347
"""

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from safenav.agents.actor_critic import ActorCritic
from safenav.agents.policy import PPOPolicy
from safenav.config import validate_config
from safenav.env.curriculum import CurriculumScheduler
from safenav.env.obstacle_map import Obstacle
from safenav.evaluation.metrics import summarize
from safenav.evaluation.runner import episode_seeds, evaluate_policy
from safenav.factory import make_env
from safenav.training.callbacks import CheckpointCallback, TrainingLogger, environment_info, write_json
from safenav.training.rollout_buffer import RolloutBuffer


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PPOTrainer:
    """
    Manages the full PPO training loop.

    Training flow per update cycle:
      ┌─ Collect rollout_steps transitions using the current policy
      ├─ Compute GAE advantages
      └─ For n_epochs:
           └─ For each mini-batch:
                ├─ Policy ratio r = π_new(a|s) / π_old(a|s)
                ├─ Clipped surrogate loss L_CLIP
                ├─ Value loss L_VF
                ├─ Entropy bonus L_ENT
                └─ θ ← θ - α·∇(L_CLIP + c1·L_VF - c2·L_ENT)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Optional[str] = None,
        safety_filter: Optional[bool] = None,
        verbose: bool = True,
        resume: bool = False,
    ):
        """
        Args:
            config: Validated configuration.
            run_dir: Output directory (default: training.log_dir).
            safety_filter: Train with the CBF filter (default: safety.enabled).
            verbose: Print progress.
            resume: Continue from run_dir/checkpoints/resume.pt if it exists
                (exact: model, optimizer, all random streams, environment,
                curriculum and logs are restored).
        """
        validate_config(config)
        self.config = config
        ppo_cfg = config.get("ppo", {})
        train_cfg = config.get("training", {})
        self.verbose = verbose

        # ── Device ────────────────────────────────────────────────
        device_str = train_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # ── Seed ─────────────────────────────────────────────────
        self.seed = int(train_cfg.get("seed", 42))
        set_global_seeds(self.seed)

        # ── PPO hyperparameters ───────────────────────────────────
        self.total_timesteps = int(ppo_cfg.get("total_timesteps", 1_000_000))
        self.rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        self.n_epochs = int(ppo_cfg.get("n_epochs", 10))
        self.batch_size = int(ppo_cfg.get("batch_size", 64))
        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_epsilon = float(ppo_cfg.get("clip_epsilon", 0.2))
        self.value_loss_coef = float(ppo_cfg.get("value_loss_coef", 0.5))
        self.entropy_coef = float(ppo_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        self.learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))
        self.lr_schedule = ppo_cfg.get("lr_schedule", "linear")
        self.target_kl = ppo_cfg.get("target_kl")

        # ── Curriculum ───────────────────────────────────────────
        self.curriculum = None
        cur_cfg = config.get("curriculum", {}) or {}
        if cur_cfg.get("enabled", True) and cur_cfg.get("stages"):
            self.curriculum = CurriculumScheduler(cur_cfg)

        # ── Environment (optionally wrapped with the safety filter) ─
        if safety_filter is None:
            safety_filter = bool(config.get("safety", {}).get("enabled", True))
        self.safety_filter = safety_filter
        task = None
        if self.curriculum is None and cur_cfg.get("stages"):
            task = cur_cfg["stages"][-1]  # no curriculum: train on the hardest stage
        self.env = make_env(config, task=task, curriculum=self.curriculum, safety_filter=safety_filter,
                            training=True)

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # ── Model and optimizer ───────────────────────────────────
        self.model = ActorCritic(config, obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        # ── Rollout buffer ────────────────────────────────────────
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
            seed=self.seed,
        )

        # ── Logging, validation and checkpoints ───────────────────
        self.run_dir = Path(run_dir or train_cfg.get("log_dir", "logs"))
        self.resume_path = self.run_dir / "checkpoints" / "resume.pt"
        self._resume_state = None
        if resume and self.resume_path.is_file():
            # Our own file (written by _save_resume); it holds RNG states, hence weights_only=False.
            self._resume_state = torch.load(self.resume_path, map_location="cpu", weights_only=False)
        self.logger = TrainingLogger(str(self.run_dir), verbose=verbose,
                                     resume_state=self._resume_state["logger"] if self._resume_state else None)
        self.checkpoints = CheckpointCallback(str(self.run_dir / "checkpoints"))
        self.eval_every = int(train_cfg.get("eval_every", 50_000))
        self.eval_episodes = int(train_cfg.get("eval_episodes", 20))
        self.validation_seed_base = int(train_cfg.get("validation_seed", 1_000_000))
        self.resume_every = int(train_cfg.get("resume_every", 50_000))
        self._last_validation_step: Optional[int] = None
        if self._resume_state is not None:
            self._restore(self._resume_state)

        if verbose:
            print(f"[Trainer] device={self.device} seed={self.seed} safety_filter={self.safety_filter} "
                  f"obs_dim={obs_dim} params={self.model.num_parameters:,}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def train(self, stop_after_updates: Optional[int] = None) -> Dict[str, Any]:
        """
        Run training; returns a short summary dictionary.

        Args:
            stop_after_updates: Save a resume checkpoint and return after this
                many updates (used to test exact resumption).
        """
        n_updates = math.ceil(self.total_timesteps / self.rollout_steps)
        if self._resume_state is not None:
            loop = self._resume_state["loop"]
            obs = loop["obs"]
            first_update = int(loop["update"]) + 1
            next_eval, next_resume = int(loop["next_eval"]), int(loop["next_resume"])
            self._ep_return, self._ep_length = float(loop["ep_return"]), int(loop["ep_length"])
            self._ep_interventions = int(loop["ep_interventions"])
            self._last_validation_step = loop["last_validation_step"]
            start = time.time() - float(loop["elapsed"])
            if self.verbose:
                print(f"[Trainer] resumed at step {self.logger.total_timesteps:,} (update {first_update - 1})")
        else:
            write_json(self.run_dir / "run_info.json", {
                "seed": self.seed,
                "safety_filter": self.safety_filter,
                "config": self.config,
                "environment": environment_info(),
            })
            obs, _ = self.env.reset(seed=self.seed)
            self._ep_return, self._ep_length, self._ep_interventions = 0.0, 0, 0
            first_update, next_eval, next_resume = 1, self.eval_every, self.resume_every
            start = time.time()

        for update in range(first_update, n_updates + 1):
            progress = (update - 1) / n_updates
            if self.lr_schedule == "linear":
                lr = max(self.learning_rate * (1.0 - progress), 1e-7)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
            lr = self.optimizer.param_groups[0]["lr"]

            t0 = time.time()
            obs, rollout_info = self._collect_rollout(obs)
            update_info = self._ppo_update()
            fps = self.rollout_steps / max(time.time() - t0, 1e-9)

            self.logger.log_update(dict(
                update=update, lr=lr, fps=fps,
                stage=self.curriculum.stage_name if self.curriculum else "fixed",
                **rollout_info, **update_info,
            ))

            if self.eval_episodes > 0 and self.logger.total_timesteps >= next_eval:
                self._validate()
                next_eval += self.eval_every

            interrupted = stop_after_updates is not None and update >= stop_after_updates
            if self.resume_every > 0 and (self.logger.total_timesteps >= next_resume or interrupted):
                while next_resume <= self.logger.total_timesteps:
                    next_resume += self.resume_every
                self._save_resume(update, obs, next_eval, next_resume, time.time() - start)
            if interrupted and update < n_updates:
                self.logger.close()
                return {"interrupted_after_update": update, "timesteps": self.logger.total_timesteps}

        if self.eval_episodes > 0 and self._last_validation_step != self.logger.total_timesteps:
            self._validate()
        final = self.checkpoints.save(
            self.checkpoints.checkpoint_dir / "final.pt", model=self.model, config=self.config,
            timestep=self.logger.total_timesteps,
            extra={"seed": self.seed, "safety_filter": self.safety_filter},
        )
        self.logger.close()
        summary = {
            "timesteps": self.logger.total_timesteps,
            "episodes": self.logger.total_episodes,
            "wall_time_s": time.time() - start,
            "best_validation": self.checkpoints.best_score,
            "final_checkpoint": str(final),
        }
        write_json(self.run_dir / "train_summary.json", summary)
        if self.resume_path.is_file():
            self.resume_path.unlink()   # finished: the resume state is no longer needed
        if self.verbose:
            print(f"[Trainer] done in {summary['wall_time_s'] / 60:.1f} min; checkpoints in {self.checkpoints.checkpoint_dir}")
        return summary

    def _collect_rollout(self, obs: np.ndarray):
        """Collect rollout_steps transitions from the environment."""
        self.buffer.reset()
        self.model.eval()
        interventions = 0
        penalty_sum, filter_gap_sum = 0.0, 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob, value = self.model(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.logger.total_timesteps += 1
            if info.get("safety_intervened", False):
                interventions += 1
                self._ep_interventions += 1
            # Logged only (E1): the correction penalty applied to this reward, and how far the
            # filter moved the in-bounds action, ||clip(u) - u_exec||^2, with or without a penalty.
            penalty_sum += float(info.get("correction_penalty", 0.0))
            if "nominal_action" in info:
                filter_gap_sum += float(np.sum((info["nominal_action"] - info["executed_action"]) ** 2))

            # Time-limit truncation is not a real terminal state: bootstrap from
            # the value of the final observation (Pardo et al., 2018).
            stored_reward = reward
            if truncated and not terminated:
                with torch.no_grad():
                    next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    stored_reward = reward + self.gamma * self.model.get_value(next_t).item()

            done = terminated or truncated
            self.buffer.add(
                obs=obs,
                action=action,          # the policy's own (unfiltered) sample
                reward=stored_reward,
                done=done,
                value=value.item(),
                log_prob=log_prob.item(),
            )

            self._ep_return += reward
            self._ep_length += 1
            if done:
                stage = self.curriculum.stage_name if self.curriculum else "fixed"
                self.logger.log_episode(self._ep_return, self._ep_length, info, stage=stage,
                                        interventions=self._ep_interventions)
                if self.curriculum is not None:
                    if self.curriculum.record_outcome(info.get("goal_reached", False)) and self.verbose:
                        print(f"[Curriculum] advanced to '{self.curriculum.stage_name}' "
                              f"at step {self.logger.total_timesteps:,}")
                obs, _ = self.env.reset()
                self._ep_return, self._ep_length, self._ep_interventions = 0.0, 0, 0
            else:
                obs = next_obs

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.model.get_value(obs_tensor).item()
        self.buffer.compute_advantages(last_value=last_value)
        return obs, {"intervention_rate": interventions / self.rollout_steps,
                     "correction_penalty_mean": penalty_sum / self.rollout_steps,
                     "filter_gap_sq_mean": filter_gap_sum / self.rollout_steps}

    def _ppo_update(self) -> Dict[str, float]:
        """n_epochs passes over the rollout buffer, updating policy and value function."""
        self.model.train()
        logs = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}
        stop = False

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_log_prob_b, adv_b, ret_b in self.buffer.get_batches(self.batch_size):
                log_probs, entropy, values = self.model.evaluate_actions(obs_b, act_b)

                log_ratio = log_probs - old_log_prob_b
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * nn.functional.mse_loss(values, ret_b)
                entropy_mean = entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
                logs["policy_loss"].append(policy_loss.item())
                logs["value_loss"].append(value_loss.item())
                logs["entropy"].append(entropy_mean.item())
                logs["approx_kl"].append(approx_kl)
                logs["clip_fraction"].append(clip_frac)

                if self.target_kl is not None and approx_kl > 1.5 * float(self.target_kl):
                    stop = True
                    break
            if stop:
                break

        out = {k: float(np.mean(v)) for k, v in logs.items()}
        returns, values = self.buffer.returns, self.buffer.values
        var_ret = float(np.var(returns))
        out["explained_variance"] = float(1.0 - np.var(returns - values) / var_ret) if var_ret > 1e-12 else 0.0
        return out

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Evaluate the deterministic policy on fixed validation seeds (all stages)."""
        policy = PPOPolicy(self.model, deterministic=True)
        stages = (self.config.get("curriculum", {}) or {}).get("stages") or []
        n_per_stage = max(1, self.eval_episodes // max(1, len(stages)))
        results = []
        for idx in range(len(stages)):
            seeds = episode_seeds(self.validation_seed_base, idx, n_per_stage)
            results += evaluate_policy(self.config, policy, idx, seeds, safety_filter=self.safety_filter)
        self.model.train()
        s = summarize(results)
        row = {
            "episodes": s["episodes"],
            "success_rate": s["success"]["rate"],
            "collision_rate": s["collision"]["rate"],
            "timeout_rate": s["timeout"]["rate"],
            "mean_return": s["mean_return"],
        }
        self.logger.log_validation(row)
        self._last_validation_step = self.logger.total_timesteps
        score = (row["success_rate"], -row["collision_rate"], row["mean_return"])
        self.checkpoints.maybe_save_best(
            score, model=self.model, config=self.config, timestep=self.logger.total_timesteps,
            extra={"seed": self.seed, "safety_filter": self.safety_filter, "validation": row},
        )

    # ── Exact resumption ──────────────────────────────────────────────────────

    def _save_resume(self, update: int, obs: np.ndarray, next_eval: int, next_resume: int, elapsed: float) -> None:
        """Everything needed to continue bit-for-bit after an interruption (written atomically)."""
        base = self.env.unwrapped
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "buffer_rng": self.buffer._rng.bit_generator.state,
            "env": {
                "rng": base._rng.bit_generator.state,
                "robot_pos": base.robot_pos.copy(), "robot_theta": base.robot_theta,
                "start_pos": base.start_pos.copy(), "goal_pos": base.goal_pos.copy(),
                "obstacles": [(o.x, o.y, o.radius) for o in base.obstacle_map.obstacles],
                "min_obstacles": base.obstacle_map.min_obstacles, "max_obstacles": base.obstacle_map.max_obstacles,
                "step_count": base._step_count, "prev_dist": base._prev_dist_to_goal,
                "trajectory": [t.copy() for t in base.trajectory], "headings": list(base.headings),
                "max_steps": base.max_steps,
            },
            "curriculum": None if self.curriculum is None else {
                "stage": self.curriculum.current_stage_idx,
                "window": list(self.curriculum._outcome_window),
            },
            "best_score": self.checkpoints.best_score,
            "logger": self.logger.state_dict(),
            "loop": {
                "update": update, "obs": np.asarray(obs).copy(), "next_eval": next_eval,
                "next_resume": next_resume, "ep_return": self._ep_return, "ep_length": self._ep_length,
                "ep_interventions": self._ep_interventions, "elapsed": elapsed,
                "last_validation_step": self._last_validation_step,
            },
        }
        tmp = self.resume_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self.resume_path)

    def _restore(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.buffer._rng.bit_generator.state = state["buffer_rng"]
        base, env_state = self.env.unwrapped, state["env"]
        base._rng.bit_generator.state = env_state["rng"]
        base.robot_pos = np.asarray(env_state["robot_pos"], dtype=float).copy()
        base.robot_theta = float(env_state["robot_theta"])
        base.start_pos = np.asarray(env_state["start_pos"], dtype=float).copy()
        base.goal_pos = np.asarray(env_state["goal_pos"], dtype=float).copy()
        base.obstacle_map.min_obstacles = env_state["min_obstacles"]
        base.obstacle_map.max_obstacles = env_state["max_obstacles"]
        base.obstacle_map.set_obstacles([Obstacle(*o) for o in env_state["obstacles"]])
        base._step_count = int(env_state["step_count"])
        base._prev_dist_to_goal = float(env_state["prev_dist"])
        base.trajectory = [np.asarray(t, dtype=float).copy() for t in env_state["trajectory"]]
        base.headings = list(env_state["headings"])
        base.max_steps = int(env_state["max_steps"])
        if self.curriculum is not None and state["curriculum"] is not None:
            self.curriculum.current_stage_idx = int(state["curriculum"]["stage"])
            self.curriculum._outcome_window.clear()
            self.curriculum._outcome_window.extend(state["curriculum"]["window"])
        best = state["best_score"]
        self.checkpoints.best_score = tuple(best) if best is not None else None

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs) -> "PPOTrainer":
        from safenav.config import load_config
        return cls(load_config(config_path), **kwargs)
