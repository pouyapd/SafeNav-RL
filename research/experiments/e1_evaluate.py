"""
e1_evaluate.py — evaluation for experiment E1 (does the correction penalty alone remove the deadlocks?).
See research/experiments/E1_correction_penalty.md for the full protocol. Evaluation only; no training.

    python research/experiments/e1_evaluate.py \
        --arm penalty=runs/e1_penalty/filtered --arm control=runs/e1_control/filtered \
        --arm committed=results/checkpoints/filtered --out runs/e1_eval --jobs 4

Each arm is a directory with seed_<k>/checkpoints/best.pt (train-many layout) or seed_<k>.pt
(committed layout). Every agent is evaluated with the filter on, with mean and with sampled
actions, on two scenario sets:
  committed  the 300 benchmark test scenarios (seed base 2,000,000; already inspected, secondary)
  fresh      500 expert scenarios from seed base 5,000,000 (never used before; primary)
Each timeout is classified as filter-blocked (Type I), self-stop (Type II), moving or other
(same rules as research/scripts/audit_diagnostics.py).

Outputs: <out>/episodes.csv and <out>/summary.json (rates per arm, paired hierarchical
bootstrap differences penalty - control and penalty - committed, Holm-adjusted over the
two primary tests).
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAIL, STALL_DISP = 100, 0.10
STAGES = ("beginner", "intermediate", "expert")
FRESH_SEED_BASE, FRESH_EPISODES = 5_000_000, 500


def find_checkpoints(arm_dir: Path):
    out = {}
    for p in sorted(arm_dir.glob("seed_*")):
        m = re.fullmatch(r"seed_(\d+)(\.pt)?", p.name)
        if not m:
            continue
        ckpt = p if m.group(2) else p / "checkpoints" / "best.pt"
        if ckpt.is_file():
            out[int(m.group(1))] = ckpt
    return out


def _job(args):
    arm, seed, ckpt, deterministic, limit = args
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config
    from safenav.evaluation.runner import episode_seeds
    from safenav.factory import make_env

    config = load_config(ROOT / "configs" / "default_config.yaml")
    policy = PPOPolicy.from_checkpoint(ckpt, deterministic=deterministic)
    sets = [("committed", s, episode_seeds(config["evaluation"]["test_seed"], i, 100)) for i, s in enumerate(STAGES)]
    sets.append(("fresh", "expert", [FRESH_SEED_BASE + i for i in range(FRESH_EPISODES)]))
    rows = []
    for set_name, stage, seeds in sets:
        seeds = seeds[:limit] if limit else seeds
        env = make_env(config, task=get_stage(config, stage), safety_filter=True)   # evaluation: no penalty
        for ep_seed in seeds:
            obs, _ = env.reset(seed=ep_seed)
            policy.reset(seed=ep_seed)
            v_nom, interv, done = [], [], False
            while not done:
                obs, _, term, trunc, info = env.step(policy.act(obs))
                done = term or trunc
                v_nom.append(float(info["nominal_action"][0]))
                interv.append(bool(info["safety_intervened"]))
            base = env.unwrapped
            outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
            cls = ""
            if outcome == "timeout":
                traj = np.asarray(base.trajectory)
                disp = float(np.linalg.norm(traj[-1] - traj[-TAIL - 1]))
                if disp >= STALL_DISP:
                    cls = "moving"
                elif np.mean(interv[-TAIL:]) >= 0.5:
                    cls = "filter-blocked"
                elif np.mean(np.asarray(v_nom[-TAIL:]) <= 1e-3) >= 0.5:
                    cls = "self-stop"
                else:
                    cls = "other stall"
            rows.append({"arm": arm, "train_seed": seed, "actions": "mean" if deterministic else "sampled",
                         "set": set_name, "stage": stage, "episode_seed": ep_seed, "outcome": outcome,
                         "class": cls, "steps": int(info["step"])})
        env.close()
    return rows


def paired_hier_bootstrap(a, b, n_boot=10_000, seed=0):
    """Same estimator as research/scripts/audit_diagnostics.py (scenarios paired, seeds resampled per arm)."""
    rng = np.random.default_rng(seed)
    n_sc = a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sc = rng.integers(0, n_sc, n_sc)
        diffs[i] = (a[np.ix_(rng.integers(0, a.shape[0], a.shape[0]), sc)].mean()
                    - b[np.ix_(rng.integers(0, b.shape[0], b.shape[0]), sc)].mean())
    p_two_sided = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(p_two_sided)


def summarize(df):
    out = {"rates": [], "contrasts": []}
    df = df.assign(success=(df.outcome == "success").astype(float),
                   collision=(df.outcome == "collision").astype(float),
                   deadlock=df["class"].isin(["filter-blocked", "self-stop", "other stall"]).astype(float),
                   type1=(df["class"] == "filter-blocked").astype(float),
                   type2=(df["class"] == "self-stop").astype(float))
    for (arm, actions, set_name, stage), g in df.groupby(["arm", "actions", "set", "stage"]):
        out["rates"].append({"arm": arm, "actions": actions, "set": set_name, "stage": stage, "episodes": len(g),
                             **{m: float(g[m].mean()) for m in ("success", "collision", "deadlock", "type1", "type2")}})

    def mat(arm, actions, metric):
        g = df[(df.arm == arm) & (df.actions == actions) & (df.set == "fresh")]
        return g.pivot_table(index="train_seed", columns="episode_seed", values=metric, aggfunc="mean").to_numpy(float)

    arms = set(df.arm)
    tests = []
    for other in ("control", "committed"):
        if {"penalty", other} <= arms:
            for metric in ("deadlock", "success", "type1", "type2", "collision"):
                for actions in ("mean", "sampled"):
                    d, lo, hi, p = paired_hier_bootstrap(mat("penalty", actions, metric), mat(other, actions, metric))
                    row = {"contrast": f"penalty - {other}", "metric": metric, "actions": actions,
                           "difference": d, "ci95_low": lo, "ci95_high": hi, "p_boot": p,
                           "primary": other == "control" and metric in ("deadlock", "success") and actions == "mean"}
                    out["contrasts"].append(row)
                    if row["primary"]:
                        tests.append(row)
    # Holm adjustment over the primary tests (deadlock and success, mean actions, penalty vs control)
    for rank, row in enumerate(sorted(tests, key=lambda r: r["p_boot"])):
        row["p_holm"] = min(1.0, row["p_boot"] * (len(tests) - rank))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=directory (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="episodes per scenario set (smoke test only; 0 = all)")
    args = ap.parse_args()
    tasks = []
    for spec in args.arm:
        name, d = spec.split("=", 1)
        ckpts = find_checkpoints(Path(d))
        if not ckpts:
            raise SystemExit(f"no checkpoints found for arm {name} in {d}")
        tasks += [(name, s, str(c), det, args.limit) for s, c in ckpts.items() for det in (True, False)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for part in pool.map(_job, tasks) for r in part]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    summary = summarize(df)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary["rates"]).round(3).to_string(index=False))
    if summary["contrasts"]:
        print(pd.DataFrame(summary["contrasts"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
