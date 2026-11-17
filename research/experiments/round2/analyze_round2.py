"""
analyze_round2.py — pre-registered analysis of round 2 (README.md here). Reads results/<tag>/episodes.csv
only and writes results/summary.json. Sections that lack their inputs are skipped and listed.

    python research/experiments/round2/analyze_round2.py

Statistics: paired hierarchical bootstrap (10,000 replicates). Scenarios are resampled jointly for
both sides of a contrast; seeds are resampled independently for different agent groups and jointly
(paired) for two conditions of the same agents. Seed-level summaries (per-seed rates, mean ± SD,
t-interval over seeds) are reported next to every bootstrap interval. Baselines are deterministic:
Wilson intervals over episodes.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
N_BOOT = 10_000
STALL_CLASSES = ("filter-blocked", "self-stop", "other stall")


def load(tag):
    p = RES / tag / "episodes.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["success"] = (df.outcome == "success").astype(float)
    df["collision"] = (df.outcome == "collision").astype(float)
    df["timeout"] = (df.outcome == "timeout").astype(float)
    df["type1"] = (df["class"] == "filter-blocked").astype(float)
    df["type2"] = (df["class"] == "self-stop").astype(float)
    df["policy_stall"] = df["class"].isin(["self-stop", "other stall"]).astype(float)
    df["deadlock"] = df["class"].isin(STALL_CLASSES).astype(float)
    return df


def matrix(df, agent, metric):
    g = df[df.agent == agent]
    m = g.pivot_table(index="train_seed", columns=["stage", "episode_seed"], values=metric, aggfunc="mean")
    return m.sort_index(axis=1)


def boot(a, b, paired_seeds, seed=0):
    """a - b for seeds x scenarios matrices with the same scenario columns."""
    assert list(a.columns) == list(b.columns), "scenario sets differ"
    A, B = a.to_numpy(float), b.to_numpy(float)
    if paired_seeds:
        assert list(a.index) == list(b.index), "seed sets differ"
    rng = np.random.default_rng(seed)
    n_sc = A.shape[1]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sc = rng.integers(0, n_sc, n_sc)
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if paired_seeds else rng.integers(0, B.shape[0], B.shape[0])
        d[i] = A[np.ix_(sa, sc)].mean() - B[np.ix_(sb, sc)].mean()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"difference": float(A.mean() - B.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_boot": float(p)}


def seed_summary(m):
    per_seed = m.mean(axis=1)
    n = len(per_seed)
    mean, sd = float(per_seed.mean()), float(per_seed.std(ddof=1)) if n > 1 else 0.0
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = t * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"per_seed": {int(k): float(v) for k, v in per_seed.items()}, "mean": mean, "sd": sd,
            "t_ci95": [mean - half, mean + half]}


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def rates(df):
    out = {}
    for (agent, stage), g in df.groupby(["agent", "stage"]):
        t = g[g.outcome == "timeout"]
        r = {"episodes": int(len(g)), "success": float(g.success.mean()), "collision": float(g.collision.mean()),
             "timeout": float(g.timeout.mean()), "min_clearance_m": float(g.min_obstacle_clearance.min()),
             "classes": t["class"].value_counts().to_dict()}
        if (g.kind != "ppo").all():
            r["success_wilson95"] = wilson(int(g.success.sum()), len(g))
        else:
            r["success_seed"] = seed_summary(matrix(g, agent, "success"))
        out[f"{agent}|{stage}"] = r
    return out


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k]["p_boot"])
    for rank, k in enumerate(order):
        tests[k]["p_holm"] = min(1.0, tests[k]["p_boot"] * (len(order) - rank))
    return tests


def e3(summary, missing):
    stages = {}
    for stage in ("expert", "intermediate"):
        mean, smp = load(f"e3_{stage}_mean"), load(f"e3_{stage}_sampled")
        if mean is None or smp is None:
            missing.append(f"E3 {stage}")
            continue
        c = {
            "1_type1_with_minus_without_mean": boot(matrix(mean, "filtered", "type1"), matrix(mean, "unfiltered", "type1"), False),
            "2_type2_with_minus_without_mean": boot(matrix(mean, "filtered", "type2"), matrix(mean, "unfiltered", "type2"), False),
            "3_type2_filtered_sampled_minus_mean": boot(matrix(smp, "filtered", "type2"), matrix(mean, "filtered", "type2"), True),
            "4_type1_filtered_sampled_minus_mean": boot(matrix(smp, "filtered", "type1"), matrix(mean, "filtered", "type1"), True),
        }
        crit = {
            "1": c["1_type1_with_minus_without_mean"]["ci95"][1] < 0,
            "2": c["2_type2_with_minus_without_mean"]["ci95"][0] > 0,
            "3": c["3_type2_filtered_sampled_minus_mean"]["ci95"][1] < 0,
            "4": c["4_type1_filtered_sampled_minus_mean"]["ci95"][0] <= 0 <= c["4_type1_filtered_sampled_minus_mean"]["ci95"][1],
        }
        held = sum(crit.values())
        stages[stage] = {"contrasts": c, "criteria_met": crit,
                         "H3": "supported" if held == 4 else ("partly supported" if held >= 2 else "not supported"),
                         "rates_mean": rates(mean), "rates_sampled": rates(smp),
                         "success_with_minus_without_mean": boot(matrix(mean, "filtered", "success"),
                                                                 matrix(mean, "unfiltered", "success"), False)}
        bins = pd.cut(mean.start_goal_dist_m, [0, 3, 5, 7, 9, 13])
        stages[stage]["success_by_distance_bin_mean"] = (
            mean.groupby([mean.agent, bins], observed=True).success.agg(["mean", "count"]).reset_index()
            .astype({"start_goal_dist_m": str}).to_dict(orient="records"))
        stages[stage]["goal_distance_m"] = {"min": float(mean.start_goal_dist_m.min()),
                                            "median": float(mean.start_goal_dist_m.median()),
                                            "max": float(mean.start_goal_dist_m.max()),
                                            "fallbacks": int(mean.goal_sampling_fallback.sum())}
    if stages:
        summary["E3"] = stages
    dense = load("e3_dense_mean")
    if dense is not None:
        summary["E3_dense_exploratory"] = {"rates": rates(dense)}
    else:
        missing.append("E3-dense")


def e4(summary, missing):
    mean = load("e4_mean")
    if mean is None or not {"control", "riskfree", "beta_policy"} <= set(mean.agent):
        missing.append("E4 (training arms or evaluation missing)")
    else:
        tests = {
            "H4a_beta_minus_control_policy_stall": boot(matrix(mean, "beta_policy", "policy_stall"),
                                                        matrix(mean, "control", "policy_stall"), False),
            "H4b_riskfree_minus_control_policy_stall": boot(matrix(mean, "riskfree", "policy_stall"),
                                                            matrix(mean, "control", "policy_stall"), False),
        }
        holm(tests)
        dec = {}
        for k, t in tests.items():
            dec[k] = bool(t["difference"] <= -0.10 and t["ci95"][1] < 0 and t["p_holm"] < 0.05)
        a, b = dec["H4a_beta_minus_control_policy_stall"], dec["H4b_riskfree_minus_control_policy_stall"]
        reading = ("aliasing supported" if a and not b else "reward supported" if b and not a
                   else "both" if a and b else "neither")
        secondary = {}
        for arm in ("beta_policy", "riskfree"):
            for m in ("success", "type1", "type2", "deadlock", "collision"):
                secondary[f"{arm}_minus_control_{m}"] = boot(matrix(mean, arm, m), matrix(mean, "control", m), False)
        summary["E4"] = {"primary": tests, "decisions": dec, "reading": reading, "secondary_mean": secondary,
                         "rates_mean": rates(mean),
                         "policy_stall_seed": {arm: seed_summary(matrix(mean, arm, "policy_stall"))
                                               for arm in ("control", "riskfree", "beta_policy")}}
        smp = load("e4_sampled")
        if smp is not None:
            summary["E4"]["rates_sampled"] = rates(smp)
    diag = load("e4_diag_committed_fresh")
    if diag is not None:
        ss = diag[diag["class"] == "self-stop"]
        summary["E4_diagnostics_descriptive"] = {
            "self_stops": int(len(ss)),
            "tail_mean_raw_v": {"median": float(ss.tail_mean_raw_v.median()),
                                "q25": float(ss.tail_mean_raw_v.quantile(.25)), "q75": float(ss.tail_mean_raw_v.quantile(.75))},
            "share_self_stops_raw_v_mostly_nonpositive": float((ss.tail_share_raw_v_nonpositive >= 0.5).mean()),
            "share_self_stops_raw_v_below_minus_0p5": float((ss.tail_mean_raw_v < -0.5).mean()),
            "end_risk_score": {"median": float(ss.end_risk_score.median()),
                               "share_zero": float((ss.end_risk_score <= 0).mean())},
            "end_barrier_m": {"median": float(ss.end_obstacle_barrier_m.median())},
        }
    else:
        missing.append("E4 diagnostics")


def dwa(summary, missing):
    sel = RES / "dwa_selection.json"
    if not sel.is_file():
        missing.append("DWA tuning")
        return
    out = {"selection": json.loads(sel.read_text(encoding="utf-8"))}
    for tag in ("dwa_test_on", "dwa_test_off", "dwa_fresh_on"):
        df = load(tag)
        if df is None:
            missing.append(tag)
            continue
        out[tag] = rates(df)
    summary["DWA"] = out


def main() -> int:
    summary = {"note": "Pre-registered round-2 analysis (README.md). Rates are shares of episodes."}
    missing = []
    e3(summary, missing)
    e4(summary, missing)
    dwa(summary, missing)
    summary["missing_inputs"] = missing
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v if k == "missing_inputs" else "…") for k, v in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
