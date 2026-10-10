"""
Deterministic vs. sampled actions for the trained PPO agents (filter on, test scenarios).

The benchmark evaluates the policy mean (deterministic). Failure replays showed
agents that stop at a keep-out boundary and never turn away; this script checks
how often sampling actions from the policy (seeded per episode) avoids that.

    python scripts/stochastic_eval.py --runs runs/benchmark --out results/stochastic_eval

Falls back to the checkpoints in results/checkpoints/ when --runs has none.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from safenav.config import load_config  # noqa: E402
from safenav.evaluation.metrics import t_interval  # noqa: E402

STAGES = ("beginner", "intermediate", "expert")


def _checkpoint(runs: Path, variant: str, seed: int) -> Path:
    p = runs / variant / f"seed_{seed}" / "checkpoints" / "best.pt"
    return p if p.is_file() else Path("results/checkpoints") / variant / f"seed_{seed}.pt"


def _job(args):
    variant, seed, ckpt, deterministic, episodes, seed_base = args
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.evaluation.runner import episode_seeds, evaluate_policy
    config = load_config()
    policy = PPOPolicy.from_checkpoint(ckpt, deterministic=deterministic)
    rows = []
    for idx, stage in enumerate(STAGES):
        for r in evaluate_policy(config, policy, stage, episode_seeds(seed_base, idx, episodes), safety_filter=True,
                                 method=variant, train_seed=seed):
            row = r.to_row()
            row.update(variant=variant, actions="mean (deterministic)" if deterministic else "sampled")
            rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs/benchmark")
    p.add_argument("--out", default="results/stochastic_eval")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    config = load_config()
    seed_base = int(config["evaluation"]["test_seed"])
    jobs = [(v, s, str(_checkpoint(Path(args.runs), v, s)), det, args.episodes, seed_base)
            for v in ("filtered", "unfiltered") for s in range(5) for det in (True, False)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for part in pool.map(_job, jobs) for r in part]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "episodes.csv", index=False)
    summary = []
    for (variant, actions, stage), g in df.groupby(["variant", "actions", "stage"], sort=False):
        per_seed = g.groupby("train_seed")
        stats = {}
        for metric in ("success", "collision", "timeout"):
            mean, low, high = t_interval(per_seed[metric].mean().to_numpy(dtype=float))
            stats[f"{metric}_rate"], stats[f"{metric}_ci"] = mean, [low, high]
        summary.append({"trained": "with filter" if variant == "filtered" else "without filter", "actions": actions,
                        "stage": stage, "episodes": len(g), **stats})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = pd.DataFrame(summary)[["trained", "actions", "stage", "success_rate", "collision_rate", "timeout_rate"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
