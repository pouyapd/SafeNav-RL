"""
benchmark.py

Multi-method, multi-seed, multi-difficulty evaluation with machine-readable
output.

Methods (each evaluated on the same seeded test scenarios):
  PPO agents, one per training seed and training variant
      (trained with or without the safety filter), each evaluated with and
      without the filter at test time (a 2 x 2 ablation);
  GoToGoal and PotentialField baselines, with and without the filter.

Output directory:
  episodes.csv      one row per (method, filter, training seed, stage, episode)
  scenarios.csv     per test scenario: obstacle count, straight-line distance,
                    oracle reachability and shortest path at margin 0 and at the
                    safety margin
  summary.csv       per (method, filter, stage): means, 95% intervals
  metadata.json     command, config, seeds, software versions, git commit
  trajectories.npz  full trajectories of the first few episodes per condition
                    (used for the trajectory figures)
"""

import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy
from safenav.agents.policy import PPOPolicy
from safenav.config import get_stage
from safenav.env.navigation_env import NavigationEnv
from safenav.evaluation.feasibility import check_feasibility
from safenav.evaluation.metrics import EpisodeResult, t_interval, wilson_interval
from safenav.evaluation.runner import episode_seeds, evaluate_policy
from safenav.training.callbacks import environment_info, write_json

STAGES = ("beginner", "intermediate", "expert")


@dataclass
class Condition:
    """One evaluated condition: a policy, a filter setting and (for PPO) a training seed."""

    method: str                 # label used in tables and figures
    kind: str                   # "ppo" | "gotogoal" | "potential_field"
    safety_filter: bool         # filter applied at test time
    checkpoint: Optional[str] = None
    train_seed: Optional[int] = None
    trained_with_filter: Optional[bool] = None


def discover_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """
    Find trained runs laid out as <runs_dir>/<variant>/seed_<k>/checkpoints/best.pt,
    where <variant> is "filtered" or "unfiltered".
    """
    found = []
    for variant in ("filtered", "unfiltered"):
        for seed_dir in sorted((runs_dir / variant).glob("seed_*")):
            ckpt = seed_dir / "checkpoints" / "best.pt"
            m = re.fullmatch(r"seed_(\d+)", seed_dir.name)
            if ckpt.is_file() and m:
                found.append({"variant": variant, "seed": int(m.group(1)), "checkpoint": str(ckpt)})
    return found


def build_conditions(runs: Sequence[Dict[str, Any]], include_baselines: bool = True) -> List[Condition]:
    conds: List[Condition] = []
    for run in runs:
        trained_f = run["variant"] == "filtered"
        for test_f in (True, False):
            label = f"PPO ({'filter' if trained_f else 'no filter'} in training)"
            conds.append(Condition(label, "ppo", test_f, run["checkpoint"], run["seed"], trained_f))
    if include_baselines:
        for test_f in (True, False):
            conds.append(Condition("GoToGoal", "gotogoal", test_f))
            conds.append(Condition("PotentialField", "potential_field", test_f))
    return conds


def _make_policy(cond: Condition, config: Dict[str, Any]):
    if cond.kind == "ppo":
        return PPOPolicy.from_checkpoint(cond.checkpoint, deterministic=True, name=cond.method)
    if cond.kind == "gotogoal":
        return GoToGoalPolicy()
    if cond.kind == "potential_field":
        return PotentialFieldPolicy.from_config(config)
    raise ValueError(f"Unknown policy kind: {cond.kind}")


def _run_condition(args) -> Dict[str, Any]:
    """Worker: evaluate one condition on all stages. Returns rows + kept trajectories."""
    cond, config, n_episodes, seed_base, keep = args
    import torch
    torch.set_num_threads(1)
    policy = _make_policy(cond, config)
    rows, trajs = [], []
    for stage_idx, stage in enumerate(STAGES):
        seeds = episode_seeds(seed_base, stage_idx, n_episodes)
        results = evaluate_policy(config, policy, stage, seeds, safety_filter=cond.safety_filter,
                                  method=cond.method, keep_trajectories=keep, train_seed=cond.train_seed)
        for r in results:
            row = r.to_row()
            row["trained_with_filter"] = cond.trained_with_filter
            row["policy_kind"] = cond.kind
            rows.append(row)
            if r.trajectory is not None:
                trajs.append(_pack_trajectory(r, cond))
    return {"rows": rows, "trajectories": trajs}


def _pack_trajectory(r: EpisodeResult, cond: Condition) -> Dict[str, Any]:
    return {
        "method": cond.method, "safety_filter": cond.safety_filter, "train_seed": cond.train_seed,
        "trained_with_filter": cond.trained_with_filter, "stage": r.stage, "episode_seed": r.episode_seed,
        "outcome": r.outcome, "trajectory": r.trajectory, "goal": r.goal,
        "obstacles": np.asarray(r.obstacles, dtype=float).reshape(-1, 3),
        "interventions": r.interventions_mask,
    }


def scenario_table(config: Dict[str, Any], n_episodes: int, seed_base: int) -> pd.DataFrame:
    """Method-independent properties of every test scenario (incl. oracle feasibility)."""
    env_cfg, safety = config["env"], config["safety"]
    rows = []
    for stage_idx, stage in enumerate(STAGES):
        env = NavigationEnv(config, task=get_stage(config, stage))
        for seed in episode_seeds(seed_base, stage_idx, n_episodes):
            env.reset(seed=seed)
            obstacles = [(o.x, o.y, o.radius) for o in env.obstacle_map.obstacles]
            common = dict(start=env.start_pos, goal=env.goal_pos, obstacles=obstacles,
                          workspace_size=env.workspace_size, robot_radius=env.robot_radius,
                          goal_radius=env.goal_radius)
            free = check_feasibility(**common, margin=0.0, wall_margin=0.0)
            safe = check_feasibility(**common, margin=float(safety["safe_distance"]),
                                     wall_margin=float(safety.get("wall_margin", 0.0)))
            rows.append({
                "stage": stage, "episode_seed": seed, "n_obstacles": len(obstacles),
                "straight_line": max(0.0, float(np.linalg.norm(env.goal_pos - env.start_pos)) - env.goal_radius),
                "reachable_free": free.reachable, "shortest_path_free": free.shortest_path,
                "reachable_with_margin": safe.reachable, "shortest_path_with_margin": safe.shortest_path,
            })
    return pd.DataFrame(rows)


def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (method, safety_filter, stage).

    PPO conditions: per-seed rates, then mean, std and 95% t-interval across
    training seeds. Deterministic baselines: pooled rate with 95% Wilson interval.
    """
    out = []
    keys = ["method", "safety_filter", "stage"]
    for (method, filt, stage), g in df.groupby(keys, sort=False):
        row: Dict[str, Any] = {"method": method, "safety_filter": filt, "stage": stage,
                               "episodes": len(g), "train_seeds": g["train_seed"].nunique(dropna=True)}
        is_ppo = g["train_seed"].notna().any()
        for metric in ("success", "collision", "obstacle_collision", "out_of_bounds", "timeout"):
            if is_ppo:
                per_seed = g.groupby("train_seed")[metric].mean().to_numpy(dtype=float)
                mean, low, high = t_interval(per_seed, bounds=(0.0, 1.0))   # a rate cannot leave [0, 1]
                row[f"{metric}_rate"] = mean
                row[f"{metric}_std"] = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")
            else:
                k = int(g[metric].sum())
                low, high = wilson_interval(k, len(g))
                row[f"{metric}_rate"] = k / len(g)
                row[f"{metric}_std"] = float("nan")
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = low, high
        succ = g[g["success"]]
        row["spl"] = float(g["spl"].mean())
        row["spl_oracle"] = float(g["spl_oracle"].mean()) if "spl_oracle" in g else float("nan")
        row["path_efficiency_success"] = float(succ["path_efficiency"].mean()) if len(succ) else float("nan")
        row["time_to_goal_success_s"] = float(succ["time_to_goal"].mean()) if len(succ) else float("nan")
        row["min_clearance_p05"] = float(np.percentile(g["min_clearance"], 5))
        row["min_clearance_min"] = float(g["min_clearance"].min())
        row["intervention_rate"] = float(g["intervention_rate"].mean())
        row["mean_abs_domega"] = float(g["mean_abs_domega"].mean())
        row["mean_speed"] = float(g["mean_speed"].mean())
        out.append(row)
    return pd.DataFrame(out)


def run_benchmark(
    config: Dict[str, Any],
    runs_dir: Optional[Path],
    out_dir: Path,
    n_episodes: Optional[int] = None,
    seed_base: Optional[int] = None,
    jobs: int = 1,
    keep_trajectories: int = 12,
    include_baselines: bool = True,
    command: str = "",
) -> pd.DataFrame:
    """Evaluate all conditions and write the benchmark directory. Returns the summary table."""
    eval_cfg = config.get("evaluation", {}) or {}
    n_episodes = int(n_episodes or eval_cfg.get("episodes_per_stage", 100))
    seed_base = int(seed_base if seed_base is not None else eval_cfg.get("test_seed", 2_000_000))
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_dir) if runs_dir is not None else []
    conds = build_conditions(runs, include_baselines=include_baselines)
    if not conds:
        raise ValueError("Nothing to evaluate: no runs found and baselines disabled")

    t0 = time.time()
    scenarios = scenario_table(config, n_episodes, seed_base)
    scenarios.to_csv(out_dir / "scenarios.csv", index=False)

    # Keep trajectories for the baselines and for the first training seed of each PPO variant.
    first_seed = min((r["seed"] for r in runs), default=None)
    tasks = []
    for cond in conds:
        keep = keep_trajectories if cond.train_seed in (None, first_seed) else 0
        tasks.append((cond, config, n_episodes, seed_base, keep))
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outputs = list(pool.map(_run_condition, tasks))
    else:
        outputs = [_run_condition(t) for t in tasks]

    rows, trajs = [], []
    for out in outputs:
        rows += out["rows"]
        trajs += out["trajectories"]
    df = pd.DataFrame(rows)

    # SPL with the oracle shortest path (margin 0) as reference length.
    ref = scenarios.set_index(["stage", "episode_seed"])["shortest_path_free"]
    lengths = df.set_index(["stage", "episode_seed"]).index.map(ref.to_dict())
    lengths = np.asarray(lengths, dtype=float)
    df["shortest_path_free"] = lengths
    df["spl_oracle"] = np.where(df["success"], lengths / np.maximum(df["path_length"].to_numpy(), lengths), 0.0)

    df.to_csv(out_dir / "episodes.csv", index=False)
    summary = summarize_frame(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    _save_trajectories(out_dir / "trajectories.npz", trajs)

    write_json(out_dir / "metadata.json", {
        "command": command,
        "episodes_per_stage": n_episodes,
        "test_seed_base": seed_base,
        "stages": list(STAGES),
        "runs": runs,
        "conditions": [c.__dict__ for c in conds],
        "config": config,
        "environment": environment_info(),
        "wall_time_s": time.time() - t0,
    })
    return summary


def _save_trajectories(path: Path, trajs: List[Dict[str, Any]]) -> None:
    """Store variable-length trajectories in one compressed npz (object arrays avoided)."""
    arrays: Dict[str, np.ndarray] = {}
    index = []
    for i, t in enumerate(trajs):
        key = f"t{i}"
        arrays[f"{key}_xy"] = np.asarray(t["trajectory"], dtype=np.float32)
        arrays[f"{key}_obstacles"] = np.asarray(t["obstacles"], dtype=np.float32)
        arrays[f"{key}_goal"] = np.asarray(t["goal"], dtype=np.float32)
        arrays[f"{key}_interventions"] = np.asarray(t["interventions"], dtype=bool)
        index.append({k: t[k] for k in ("method", "safety_filter", "train_seed", "trained_with_filter",
                                        "stage", "episode_seed", "outcome")} | {"key": key})
    arrays["index_json"] = np.frombuffer(json.dumps(index, default=str).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_trajectories(path: Path) -> List[Dict[str, Any]]:
    """Inverse of _save_trajectories."""
    data = np.load(path)
    index = json.loads(bytes(data["index_json"]).decode("utf-8"))
    out = []
    for item in index:
        k = item["key"]
        out.append(dict(item, trajectory=data[f"{k}_xy"], obstacles=data[f"{k}_obstacles"],
                        goal=data[f"{k}_goal"], interventions=data[f"{k}_interventions"]))
    return out
