"""
ppo_trainer.py

Full Proximal Policy Optimization (PPO) training pipeline.

PPO is the standard algorithm for continuous-control RL tasks.
Key ideas:
  1. Collect rollouts with current policy
  2. Compute advantages (GAE)
  3. Update policy using clipped surrogate objective (prevents too-large updates)
  4. Update value function to minimize MSE with returns

Reference:
  Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from typing import Dict, Any, Optional
from collections import deque

from env.navigation_env import NavigationEnv
from env.curriculum import CurriculumScheduler
from models.actor_critic import ActorCritic
from models.safety_layer import SafetyLayer
from training.rollout_buffer import RolloutBuffer
from training.callbacks import TrainingLogger, CheckpointCallback


class PPOTrainer:
    """
    Manages the full PPO training loop.

    Training flow per update cycle:
      ┌─ Collect rollout_steps transitions using current policy
      ├─ Compute GAE advantages
      └─ For n_epochs:
           └─ For each mini-batch:
                ├─ Compute policy ratio r = π_new(a|s) / π_old(a|s)
                ├─ Clipped surrogate loss: L_CLIP
                ├─ Value loss: L_VF
                ├─ Entropy bonus: L_ENT
                └─ Update θ ← θ - α·∇(L_CLIP + c1·L_VF - c2·L_ENT)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ppo_cfg = config.get("ppo", {})
        train_cfg = config.get("training", {})

        # ── Device ────────────────────────────────────────────────
        device_str = train_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        print(f"[Trainer] Using device: {self.device}")

        # ── Seed ─────────────────────────────────────────────────
        seed = train_cfg.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── PPO hyperparameters ───────────────────────────────────
        self.total_timesteps = ppo_cfg.get("total_timesteps", 1_000_000)
        self.rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        self.n_epochs = ppo_cfg.get("n_epochs", 10)
        self.batch_size = ppo_cfg.get("batch_size", 64)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.value_loss_coef = ppo_cfg.get("value_loss_coef", 0.5)
        self.entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.learning_rate = ppo_cfg.get("learning_rate", 3e-4)
        self.lr_schedule = ppo_cfg.get("lr_schedule", "linear")

        # ── Curriculum ───────────────────────────────────────────
        curriculum = None
        if config.get("curriculum", {}).get("enabled", True):
            curriculum = CurriculumScheduler(config["curriculum"])
            print(f"[Trainer] Curriculum learning enabled: {len(curriculum.stages)} stages")

        # ── Environment ───────────────────────────────────────────
        self.env = NavigationEnv(config, curriculum=curriculum)
        self.curriculum = curriculum

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # ── Models ────────────────────────────────────────────────
        self.model = ActorCritic(config).to(self.device)
        self.safety_layer = SafetyLayer(config)
        print(f"[Trainer] Model parameters: {self.model.num_parameters:,}")

        # ── Optimizer ─────────────────────────────────────────────
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        # ── Rollout Buffer ────────────────────────────────────────
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

        # ── Callbacks ─────────────────────────────────────────────
        self.logger = TrainingLogger(train_cfg.get("log_dir", "logs"))
        self.checkpoint_cb = CheckpointCallback(
            checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
            save_every=train_cfg.get("save_every", 50_000),
        )

        self.eval_every = train_cfg.get("eval_every", 10_000)
        self.eval_episodes = train_cfg.get("eval_episodes", 20)

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  SafeNav-RL — PPO Training")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"{'='*60}\n")

        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        n_updates = 0
        last_eval_step = 0

        for timestep in range(0, self.total_timesteps, self.rollout_steps):
            # ── Collect Rollout ────────────────────────────────────
            obs, episode_reward, episode_length = self._collect_rollout(
                obs, episode_reward, episode_length, timestep
            )

            # ── Update LR ─────────────────────────────────────────
            if self.lr_schedule == "linear":
                progress = timestep / self.total_timesteps
                lr = self.learning_rate * (1.0 - progress)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = max(lr, 1e-7)

            # ── PPO Update ─────────────────────────────────────────
            update_info = self._ppo_update()
            self.logger.log_update(update_info)
            n_updates += 1

            # ── Logging ───────────────────────────────────────────
            self.logger.total_timesteps = timestep
            if n_updates % 10 == 0:
                curriculum_info = str(self.curriculum) if self.curriculum else None
                self.logger.print_summary(timestep, curriculum_info)

            # ── Checkpointing ─────────────────────────────────────
            if self.checkpoint_cb.should_save(timestep):
                mean_reward = float(np.mean(self.logger.episode_rewards)) if self.logger.episode_rewards else -999.0
                self.checkpoint_cb.save(
                    self.model, self.optimizer, timestep, mean_reward, self.config
                )

        print("\n[Trainer] Training complete.")
        self.logger.close()
        # Final checkpoint
        mean_reward = float(np.mean(self.logger.episode_rewards)) if self.logger.episode_rewards else -999.0
        self.checkpoint_cb.save(
            self.model, self.optimizer, self.total_timesteps, mean_reward, self.config, tag="final"
        )

    def _collect_rollout(
        self, obs: np.ndarray, ep_reward: float, ep_length: int, timestep: int
    ):
        """Collect rollout_steps transitions from the environment."""
        self.buffer.reset()
        self.model.eval()

        last_info = {}
        for _ in range(self.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob, value = self.model(obs_tensor)

            action_np = action_tensor.squeeze(0).cpu().numpy()

            # Apply safety layer
            info = self.env._get_info()
            safe_action = self.safety_layer.project_action(
                action_np,
                self.env.robot_pos,
                self.env.robot_theta,
                self.env.obstacle_map.obstacles,
            )

            next_obs, reward, terminated, truncated, info = self.env.step(safe_action)
            done = terminated or truncated

            self.buffer.add(
                obs=obs,
                action=safe_action,
                reward=reward,
                done=done,
                value=value.item(),
                log_prob=log_prob.item(),
            )

            obs = next_obs
            ep_reward += reward
            ep_length += 1
            last_info = info

            if done:
                # Log episode
                self.logger.log_episode(ep_reward, ep_length, last_info)

                # Update curriculum
                if self.curriculum:
                    advanced = self.curriculum.record_outcome(last_info.get("goal_reached", False))
                    if advanced:
                        print(f"\n[Curriculum] Advanced to: {self.curriculum.stage_name}\n")

                obs, _ = self.env.reset()
                ep_reward = 0.0
                ep_length = 0

        # Bootstrap value for last state
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.model.get_value(obs_tensor).item()

        self.buffer.compute_advantages(last_value=last_value, last_done=done)
        return obs, ep_reward, ep_length

    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform n_epochs passes over the rollout buffer,
        updating the policy and value function.
        """
        self.model.train()
        all_losses = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_log_prob_b, adv_b, ret_b in self.buffer.get_batches(self.batch_size):

                # Evaluate current policy on batch
                log_probs, entropy, values = self.model.evaluate_actions(obs_b, act_b)

                # ── Policy (Actor) Loss ────────────────────────────
                # Ratio r = π_new / π_old
                ratio = torch.exp(log_probs - old_log_prob_b)

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value (Critic) Loss ────────────────────────────
                value_loss = 0.5 * nn.functional.mse_loss(values, ret_b)

                # ── Entropy Bonus ──────────────────────────────────
                # Encourages exploration by penalizing low-entropy policies
                entropy_loss = -entropy.mean()

                # ── Combined Loss ──────────────────────────────────
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["value_loss"].append(value_loss.item())
                all_losses["entropy"].append(-entropy_loss.item())
                all_losses["total_loss"].append(total_loss.item())

        return {k: float(np.mean(v)) for k, v in all_losses.items()}

    @classmethod
    def from_config_file(cls, config_path: str) -> "PPOTrainer":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
