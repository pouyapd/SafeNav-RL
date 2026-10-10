"""
runner.py

Seeded evaluation of any Policy (trained PPO agent or baseline controller).

Every episode is generated from its own seed (`episode_seeds`), so different
methods are evaluated on exactly the same start, goal and obstacle layouts.
Evaluation seeds are kept apart from the training stream and from the
validation seeds used for checkpoint selection (see default_config.yaml).
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from safenav.agents.policy import Policy
from safenav.config import get_stage
from safenav.evaluation.metrics import EpisodeResult, mean_abs_change, path_length
from safenav.factory import make_env


def episode_seeds(seed_base: int, stage_index: int, n_episodes: int) -> List[int]:
    """Deterministic, non-overlapping episode seeds for one difficulty stage."""
    return [int(seed_base + 100_000 * stage_index + i) for i in range(n_episodes)]


def run_episode(
    env,
    policy: Policy,
    seed: int,
    method: Optional[str] = None,
    stage: str = "",
    keep_trajectory: bool = False,
    train_seed: Optional[int] = None,
) -> EpisodeResult:
    """Run one episode from `seed` and return its metrics."""
    base = env.unwrapped
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)

    risks = [info["risk_score"]]
    clearances = [info["obstacle_clearance"]]
    wall_clearances = [info["wall_clearance"]]
    v_cmd: List[float] = []
    w_cmd: List[float] = []
    intervened: List[bool] = []
    total_reward = 0.0
    done = False

    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        executed = info["executed_action"]
        v_cmd.append(float(executed[0]))
        w_cmd.append(float(executed[1]))
        intervened.append(bool(info.get("safety_intervened", False)))
        risks.append(info["risk_score"])
        clearances.append(info["obstacle_clearance"])
        wall_clearances.append(info["wall_clearance"])

    traj = np.asarray(base.trajectory)
    goal = base.goal_pos.copy()
    straight = max(0.0, float(np.linalg.norm(goal - traj[0])) - base.goal_radius)
    finite_clear = [c for c in clearances if np.isfinite(c)]

    result = EpisodeResult(
        method=method or getattr(policy, "name", "policy"),
        stage=stage,
        episode_seed=int(seed),
        safety_filter=hasattr(env, "safety_filter"),
        success=bool(info["goal_reached"]),
        obstacle_collision=bool(info["obstacle_collision"]),
        out_of_bounds=bool(info["out_of_bounds"]),
        timeout=not info["goal_reached"] and not info["collision"],
        steps=int(info["step"]),
        episode_return=float(total_reward),
        straight_line=straight,
        path_length=path_length(traj),
        min_clearance=float(min(finite_clear)) if finite_clear else float("inf"),
        min_wall_clearance=float(min(wall_clearances)),
        mean_risk=float(np.mean(risks)),
        max_risk=float(np.max(risks)),
        interventions=int(np.sum(intervened)),
        mean_abs_dv=mean_abs_change(v_cmd),
        mean_abs_domega=mean_abs_change(w_cmd),
        mean_speed=float(np.mean(v_cmd)) if v_cmd else 0.0,
        dt=float(base.dt),
        train_seed=train_seed,
    )
    if keep_trajectory:
        result.trajectory = traj
        result.headings = np.asarray(base.headings)
        result.goal = goal
        result.obstacles = [(o.x, o.y, o.radius) for o in base.obstacle_map.obstacles]
        result.interventions_mask = np.asarray(intervened, dtype=bool)
    return result


def evaluate_policy(
    config: Dict[str, Any],
    policy: Policy,
    stage: str | int,
    seeds: Iterable[int],
    safety_filter: bool,
    method: Optional[str] = None,
    keep_trajectories: int = 0,
    train_seed: Optional[int] = None,
) -> List[EpisodeResult]:
    """
    Evaluate a policy on one difficulty stage.

    Args:
        config: Validated configuration.
        policy: Policy to evaluate.
        stage: Curriculum stage name or index that fixes the difficulty.
        seeds: Episode seeds (see episode_seeds()).
        safety_filter: Wrap the environment with the CBF safety filter.
        method: Label stored with each result (default: policy.name).
        keep_trajectories: Keep full trajectories for the first k episodes.
        train_seed: Training seed of the evaluated checkpoint (for PPO).
    """
    task = get_stage(config, stage)
    env = make_env(config, task=task, safety_filter=safety_filter)
    stage_name = task.get("name", str(stage))
    results = []
    for i, seed in enumerate(seeds):
        results.append(run_episode(
            env, policy, seed, method=method, stage=stage_name,
            keep_trajectory=i < keep_trajectories, train_seed=train_seed,
        ))
    env.close()
    return results
