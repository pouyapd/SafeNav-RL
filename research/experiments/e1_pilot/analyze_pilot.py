"""
analyze_pilot.py — pre-registered analysis of the E1 pilot (README.md in this folder).
Reads results/<tag>/episodes.csv and the training logs in runs/e1_pilot; writes results/summary.json.

    python research/experiments/e1_pilot/analyze_pilot.py      # seconds

One training seed per arm: the bootstrap resamples the 500 evaluation scenarios only, so its
intervals describe scenario variability for these two trained agents, not training-seed
variability. They are not a test of the penalty's effect.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
RUNS = ROOT / "runs" / "e1_pilot"
N_BOOT = 10_000
STALLS = ("filter-blocked", "self-stop", "other stall")


def load(tag, root=RES):
    p = root / tag / "episodes.csv"
    return pd.read_csv(p).sort_values("episode_seed").reset_index(drop=True) if p.is_file() else None


def indicators(df):
    return pd.DataFrame({
        "success": (df.outcome == "success").astype(float),
        "deadlock": df["class"].isin(STALLS).astype(float),
        "type1": (df["class"] == "filter-blocked").astype(float),
        "type2": (df["class"] == "self-stop").astype(float),
    }).set_index(df.episode_seed)


def boot(a, b, seed=0):
    """Paired difference of means a - b over the same scenarios (scenario resampling only)."""
    assert list(a.index) == list(b.index), "scenario sets differ"
    d = a.to_numpy() - b.to_numpy()
    rng = np.random.default_rng(seed)
    reps = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return {"difference": float(d.mean()), "ci95_scenarios_only": [float(np.percentile(reps, 2.5)),
                                                                    float(np.percentile(reps, 97.5))]}


def rates(df):
    t = df[df.outcome == "timeout"]
    return {"episodes": int(len(df)), "success": float((df.outcome == "success").mean()),
            "collisions": int((df.outcome == "collision").sum()),
            "min_clearance_m": float(df.min_obstacle_clearance.min()),
            "deadlock_rate": float(df["class"].isin(STALLS).mean()),
            "timeouts": int(len(t)), "classes": t["class"].value_counts().to_dict()}


def _relative_paths(summary):
    # The trainer stores absolute checkpoint paths; keep machine-specific paths out of the summary.
    p = summary.get("final_checkpoint")
    if p and Path(p).is_absolute():
        summary["final_checkpoint"] = Path(p).resolve().relative_to(ROOT).as_posix()
    return summary


def training(arm):
    run = RUNS / arm / "filtered" / "seed_0"
    if not (run / "train_summary.json").is_file():
        return None
    import torch
    prog = pd.read_csv(run / "progress.csv")
    val = pd.read_csv(run / "validation.csv")
    best = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)["timestep"]
    first = {s: int(prog[prog.stage == s].timestep.min()) for s in ("intermediate", "expert") if (prog.stage == s).any()}
    expert = prog[prog.stage == "expert"]
    return {
        "train_summary": _relative_paths(json.loads((run / "train_summary.json").read_text(encoding="utf-8"))),
        "first_timestep_at_stage": first,
        "best_checkpoint_timestep": int(best),
        "validation": val.to_dict(orient="records"),
        "correction_penalty_mean": {"all": float(prog.correction_penalty_mean.mean()),
                                    "first_10_updates": float(prog.correction_penalty_mean.head(10).mean()),
                                    "last_10_updates": float(prog.correction_penalty_mean.tail(10).mean())},
        "filter_gap_sq_mean": {"all": float(prog.filter_gap_sq_mean.mean()),
                               "last_10_updates": float(prog.filter_gap_sq_mean.tail(10).mean())},
        "expert_stage_last_10_updates": {
            "success_rate_100": float(expert.success_rate_100.tail(10).mean()) if len(expert) else None,
            "intervention_rate": float(expert.intervention_rate.tail(10).mean()) if len(expert) else None,
            "entropy": float(expert.entropy.tail(10).mean()) if len(expert) else None},
    }


def decide(ctrl, pen, diff):
    """Decision rules fixed in README.md before training."""
    out = {}
    out["G0_collisions"] = "pass" if ctrl["collisions"] == 0 and pen["collisions"] == 0 else "FAIL: debug"
    out["G2_control_has_stalls"] = ("pass" if ctrl["deadlock_rate"] >= 0.10
                                    else "fail: the reduced budget may not reproduce the stalls")
    d = diff["deadlock"]["difference"]
    out["G3_effect_screen"] = ("large reduction in this seed (<= -10 points)" if d <= -0.10 else
                               "large increase in this seed (>= +10 points)" if d >= 0.10 else
                               "no large effect in this seed (|difference| < 10 points): inconclusive")
    return out


def main() -> int:
    summary = {"note": "E1 pilot, one training seed per arm; intervals resample scenarios only (README.md)."}
    summary["training"] = {arm: training(arm) for arm in ("control", "penalty")}
    reach = summary["training"]
    if all(reach.values()):
        summary["G1_both_reach_expert"] = all("expert" in r["first_timestep_at_stage"] for r in reach.values())

    for ck in ("best", "final"):
        c, p = load(f"eval_control_{ck}"), load(f"eval_penalty_{ck}")
        if c is None or p is None or len(c) != 500 or len(p) != 500:
            continue
        ic, ip = indicators(c), indicators(p)
        diff = {m: boot(ip[m], ic[m]) for m in ("deadlock", "type1", "type2", "success")}
        block = {"control": rates(c), "penalty": rates(p), "penalty_minus_control": diff}
        if ck == "best":
            block["decisions"] = decide(block["control"], block["penalty"], diff)
        summary[ck] = block

    # Reference only: committed filtered seed-0 agent (1M-step run, best checkpoint at 100k) on the same scenarios.
    ref = load("L3_fresh_mean_k0", HERE.parent / "lightweight" / "results")
    if ref is not None:
        ref = ref[(ref.variant == "filtered") & (ref.train_seed == 0)].sort_values("episode_seed")
        summary["reference_committed_filtered_seed0"] = rates(ref)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "training"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
