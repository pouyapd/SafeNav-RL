"""
sweep.py

Safety-margin sweep: for each margin d_safe of the CBF filter, measure
  * the share of test scenarios in which the goal is reachable at all while
    keeping the margin (geometric oracle, method-independent), and
  * success / collision / timeout of the two baseline controllers with the
    filter at that margin.

This shows the price of a larger margin (fewer reachable goals, more
stopping) without retraining any agent.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy
from safenav.config import apply_overrides, get_stage
from safenav.env.navigation_env import NavigationEnv
from safenav.evaluation.benchmark import STAGES
from safenav.evaluation.feasibility import check_feasibility
from safenav.evaluation.metrics import summarize
from safenav.evaluation.runner import episode_seeds, evaluate_policy
from safenav.training.callbacks import environment_info, write_json


def _sweep_one(args) -> List[Dict[str, Any]]:
    config, margin, n_episodes, seed_base = args
    cfg = apply_overrides(config, {"safety.safe_distance": float(margin)})
    rows = []
    for stage_idx, stage in enumerate(STAGES):
        seeds = episode_seeds(seed_base, stage_idx, n_episodes)
        env = NavigationEnv(cfg, task=get_stage(cfg, stage))
        reachable = []
        for seed in seeds:
            env.reset(seed=seed)
            obstacles = [(o.x, o.y, o.radius) for o in env.obstacle_map.obstacles]
            res = check_feasibility(env.start_pos, env.goal_pos, obstacles, env.workspace_size,
                                    env.robot_radius, env.goal_radius, margin=float(margin),
                                    wall_margin=float(cfg["safety"].get("wall_margin", 0.0)))
            reachable.append(res.reachable)
        feasible_rate = sum(reachable) / len(reachable)
        for policy in (GoToGoalPolicy(), PotentialFieldPolicy.from_config(cfg)):
            s = summarize(evaluate_policy(cfg, policy, stage, seeds, safety_filter=True))
            rows.append({
                "margin": float(margin), "stage": stage, "method": policy.name,
                "feasible_rate": feasible_rate,
                "success_rate": s["success"]["rate"], "collision_rate": s["collision"]["rate"],
                "timeout_rate": s["timeout"]["rate"], "spl": s["spl"],
                "min_clearance_p05": s["min_clearance_p05"], "intervention_rate": s["intervention_rate"],
            })
    return rows


def run_safety_sweep(
    config: Dict[str, Any],
    out_dir: Path,
    margins: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8),
    n_episodes: int = 100,
    seed_base: int | None = None,
    jobs: int = 4,
) -> pd.DataFrame:
    seed_base = int(seed_base if seed_base is not None else config.get("evaluation", {}).get("test_seed", 2_000_000))
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(config, m, n_episodes, seed_base) for m in margins]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(_sweep_one, tasks))
    else:
        parts = [_sweep_one(t) for t in tasks]
    df = pd.DataFrame([row for part in parts for row in part])
    df.to_csv(out_dir / "safety_sweep.csv", index=False)
    write_json(out_dir / "metadata.json", {
        "margins": list(margins), "episodes_per_stage": n_episodes, "test_seed_base": seed_base,
        "config": config, "environment": environment_info(),
    })
    return df
