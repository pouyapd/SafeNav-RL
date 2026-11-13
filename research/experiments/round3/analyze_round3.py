"""
analyze_round3.py — pre-registered analysis of round 3 (PROTOCOL.md). Reads results/<tag>/episodes.csv only.

    python research/experiments/round3/analyze_round3.py [--tag main]

Writes results/<tag>/summary.json: per method and agent type the rates with seed-level summaries,
the primary family (C vs baseline: deadlock and success per agent type, Holm over 4 tests),
the secondary family (A and B vs baseline, Holm over 8), the method contrasts C-B, C-A, C-C_euclid,
recovery statistics, overhead and a failure-case table.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
STALLS = ("filter-blocked", "self-stop", "other stall")
TYPES = {"with_filter": "filtered", "without_filter": "unfiltered"}
N_BOOT = 10_000


def prep(df):
    c = df["class"].fillna("")
    df = df.assign(success=(df.outcome == "success").astype(float), collision=(df.outcome == "collision").astype(float),
                   deadlock=c.isin(STALLS).astype(float), filter_blocked=(c == "filter-blocked").astype(float),
                   self_stop=(c == "self-stop").astype(float), other_stall=(c == "other stall").astype(float),
                   moving=(c == "moving").astype(float), intervention_rate=df.intervention_steps / df.steps)
    return df


def mat(df, method, agent, metric):
    g = df[(df.method == method) & (df.agent == agent)]
    return g.pivot_table(index="train_seed", columns="episode_seed", values=metric).sort_index(axis=1)


def boot_paired(a, b, seed=0):
    assert list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
    A, B = a.to_numpy(float), b.to_numpy(float)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sc = rng.integers(0, A.shape[1], A.shape[1])
        sd = rng.integers(0, A.shape[0], A.shape[0])
        d[i] = A[np.ix_(sd, sc)].mean() - B[np.ix_(sd, sc)].mean()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"difference": float(A.mean() - B.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_boot": float(p)}


def seed_level(a, b):
    """Per-seed rates, paired t-test over seeds, d_z and Hedges' g_av."""
    ra, rb = a.mean(axis=1), b.mean(axis=1)
    diff = ra - rb
    n = len(diff)
    t = stats.ttest_rel(ra, rb) if n > 1 and diff.std(ddof=1) > 0 else None
    sd_avg = (ra.std(ddof=1) + rb.std(ddof=1)) / 2 if n > 1 else float("nan")
    g_av = diff.mean() / sd_avg * (1 - 3 / (4 * (n - 1) - 1)) if n > 2 and sd_avg > 0 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"per_seed_method": {int(k): float(v) for k, v in ra.items()}, "per_seed_reference": {int(k): float(v) for k, v in rb.items()},
            "mean_difference": float(diff.mean()), "sd_difference": float(diff.std(ddof=1)) if n > 1 else float("nan"),
            "t_ci95_difference": [float(diff.mean() - half), float(diff.mean() + half)],
            "paired_t_p": float(t.pvalue) if t is not None else None,
            "d_z": float(diff.mean() / diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) > 0 else float("nan"),
            "hedges_g_av": float(g_av)}


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k]["p_boot"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, tests[k]["p_boot"] * (len(order) - rank)))
        tests[k]["p_holm"] = running
    return tests


def contrast(df, m1, m2, agent, metric, seed=0):
    a, b = mat(df, m1, agent, metric), mat(df, m2, agent, metric)
    r = boot_paired(a, b, seed)
    r["seed_level"] = seed_level(a, b)
    return r


def rates(df):
    out = {}
    for (method, agent), g in df.groupby(["method", "agent"]):
        per = g.groupby("train_seed")
        r = {"episodes": int(len(g))}
        for m in ("success", "deadlock", "filter_blocked", "self_stop", "other_stall", "moving", "collision", "intervention_rate"):
            s = per[m].mean()
            half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
            r[m] = {"mean": float(g[m].mean()), "seed_mean": float(s.mean()), "seed_sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "t_ci95": [float(s.mean() - half), float(s.mean() + half)], "per_seed": {int(k): float(v) for k, v in s.items()}}
        r["min_clearance_m"] = float(g.min_obstacle_clearance.min())
        acts = g.activations.sum()
        r["recovery"] = {"activations": int(acts), "activations_per_episode": float(g.activations.mean()),
                         "episodes_with_activation": float((g.activations > 0).mean()),
                         "recovery_success_rate": float(g.activations_success.sum() / acts) if acts else None,
                         "repeat_activations": int(g.repeat_activations.sum()),
                         "mean_activation_steps": float(np.average(pd.to_numeric(g.mean_activation_steps, errors="coerce").fillna(0),
                                                                   weights=g.activations)) if acts else None,
                         "max_activations_in_episode": int(g.activations.max()),
                         "deadlocks_with_recovery_in_tail": int(((g.deadlock == 1) & (g.tail_recovery_share > 0.5)).sum())}
        calls = g.planner_calls.sum()
        r["overhead"] = {"planner_calls": int(calls),
                         "planner_ms_per_call": float(1000 * g.planner_time_s.sum() / calls) if calls else None,
                         "controller_ms_per_step": float(1000 * g.controller_time_s.sum() / g.steps.sum()),
                         "field_setup_ms_mean": float(1000 * g.field_setup_s.mean()),
                         "episode_wall_s_mean": float(g.episode_wall_s.mean())}
        out[f"{method}|{agent}"] = r
    return out


def failure_cases(df):
    g = df[(df.method == "C") & (df.deadlock == 1)]
    reasons = {}
    for s in g.exit_reasons.fillna(""):
        for r in [x for x in s.split(";") if x]:
            reasons[r] = reasons.get(r, 0) + 1
    return {"deadlocks": int(len(g)), "by_class": g["class"].value_counts().to_dict(),
            "no_activation": int((g.activations == 0).sum()), "exit_reasons": reasons,
            "geodesic_unreachable_at_start": int((~np.isfinite(g.start_geo_cost)).sum()),
            "median_geodesic_progress_m": float((g.start_geo_cost - g.end_geo_cost).replace([np.inf, -np.inf], np.nan).median()),
            "with_repeat_activations": int((g.repeat_activations > 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    res = HERE / "results" / args.tag
    df = prep(pd.read_csv(res / "episodes.csv"))
    methods = sorted(set(df.method))
    summary = {"tag": args.tag, "episodes": int(len(df)), "methods": methods,
               "collisions": int(df.collision.sum()), "min_clearance_m": float(df.min_obstacle_clearance.min()),
               "rates": rates(df)}
    if "C" in methods:
        prim = {}
        for tname, agent in TYPES.items():
            for metric in ("deadlock", "success"):
                prim[f"C_minus_baseline_{metric}_{tname}"] = contrast(df, "C", "baseline", agent, metric)
        holm(prim)
        dec = {}
        for tname in TYPES:
            d, s = prim[f"C_minus_baseline_deadlock_{tname}"], prim[f"C_minus_baseline_success_{tname}"]
            h1 = d["difference"] <= -0.10 and d["ci95"][1] < 0 and d["p_holm"] < 0.05
            h2 = s["difference"] > 0 and s["ci95"][0] > 0 and s["p_holm"] < 0.05
            safe = summary["collisions"] == 0 and summary["min_clearance_m"] >= 0.30 - 1e-9
            dec[tname] = {"H1": bool(h1), "H2": bool(h2), "safe": bool(safe), "claim": bool(h1 and h2 and safe)}
        summary["primary"], summary["decisions"] = prim, dec
        sec = {}
        for m in ("A", "B"):
            if m in methods:
                for tname, agent in TYPES.items():
                    for metric in ("deadlock", "success"):
                        sec[f"{m}_minus_baseline_{metric}_{tname}"] = contrast(df, m, "baseline", agent, metric)
        summary["secondary_family"] = holm(sec)
        other = {}
        for m2 in ("B", "A", "C_euclid"):
            if m2 in methods:
                for tname, agent in TYPES.items():
                    for metric in ("deadlock", "success", "moving", "self_stop", "filter_blocked"):
                        other[f"C_minus_{m2}_{metric}_{tname}"] = contrast(df, "C", m2, agent, metric)
        for m in [x for x in methods if x != "baseline"]:
            for tname, agent in TYPES.items():
                for metric in ("self_stop", "filter_blocked", "moving", "intervention_rate"):
                    other[f"{m}_minus_baseline_{metric}_{tname}"] = contrast(df, m, "baseline", agent, metric)
        summary["other_contrasts_unadjusted"] = other
        summary["failure_cases_C"] = failure_cases(df)
    calls = df[df.planner_calls > 0]
    if len(calls):
        per_call = (calls.planner_time_s / calls.planner_calls) * 1000
        summary["overhead_overall"] = {"planner_ms_per_call_episode_mean_p99": float(np.percentile(per_call, 99)),
                                       "note": "per-call time is averaged within an episode; p99 is over episodes"}
    (res / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"episodes": summary["episodes"], "collisions": summary["collisions"],
                      "min_clearance_m": summary["min_clearance_m"], "decisions": summary.get("decisions")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
