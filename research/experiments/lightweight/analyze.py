"""
analyze.py — pre-registered analysis of the lightweight experiments (README.md in this folder).
Reads results/<tag>/episodes.csv only; writes results/summary.json and prints the tables.

    python research/experiments/lightweight/analyze.py      # seconds
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 10_000


def load(tag):
    p = RES / tag / "episodes.csv"
    return pd.read_csv(p) if p.is_file() else None


def indicators(df):
    return df.assign(success=(df.outcome == "success").astype(float),
                     collision=(df.outcome == "collision").astype(float),
                     type1=(df["class"] == "filter-blocked").astype(float),
                     type2=(df["class"] == "self-stop").astype(float),
                     timeout=(df.outcome == "timeout").astype(float))


def mat(df, variant, metric):
    g = df[df.variant == variant]
    return g.pivot_table(index="train_seed", columns="episode_seed", values=metric, aggfunc="mean")


def boot(a, b, paired_seeds, seed=0):
    """Difference of means a - b (DataFrames seeds x scenarios, same scenario columns)."""
    a, b = a.sort_index(axis=1), b.sort_index(axis=1)
    assert list(a.columns) == list(b.columns), "scenario sets differ"
    A, B = a.to_numpy(float), b.to_numpy(float)
    if paired_seeds:
        assert list(a.index) == list(b.index)
    rng = np.random.default_rng(seed)
    n_sc = A.shape[1]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sc = rng.integers(0, n_sc, n_sc)
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if paired_seeds else rng.integers(0, B.shape[0], B.shape[0])
        d[i] = A[np.ix_(sa, sc)].mean() - B[np.ix_(sb, sc)].mean()
    return {"difference": float(A.mean() - B.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def rates(df):
    out = {}
    for v, g in df.groupby("variant"):
        t = g[g.outcome == "timeout"]
        out[v] = {"episodes": int(len(g)), "success": float((g.outcome == "success").mean()),
                  "collisions": int((g.outcome == "collision").sum()),
                  "min_clearance_m": float(g.min_obstacle_clearance.min()),
                  "timeouts": int(len(t)), "classes": t["class"].value_counts().to_dict()}
    return out


def main() -> int:
    summary = {"note": "Pre-registered analysis; see README.md. Rates are shares of episodes (x100 = per 100 episodes)."}

    # L2: kappa selection on validation (rule fixed in README.md)
    l2 = {k: load(f"L2_val_k{k}") for k in ("0", "0.5", "1.0")}
    if all(v is not None for v in l2.values()):
        sel = {k: {"success": float((d.outcome == "success").mean()), "collisions": int((d.outcome == "collision").sum())}
               for k, d in l2.items()}
        cand = {k: v for k, v in sel.items() if k != "0" and v["collisions"] == 0}
        best = max(cand, key=lambda k: cand[k]["success"])
        if abs(cand["0.5"]["success"] - cand["1.0"]["success"]) <= 0.005:
            best = "0.5"
        summary["L2_kappa_selection"] = {"validation": sel, "kappa_star": float(best)}
        print("L2 kappa selection:", sel, "-> kappa* =", best)

    # L3: fresh scenarios, policy mean
    k0, ks = load("L3_fresh_mean_k0"), load("L3_fresh_mean_k0.5")
    if k0 is not None and ks is not None and len(k0) == len(ks) == 5000:
        k0, ks = indicators(k0), indicators(ks)
        summary["L3_rates"] = {"kappa0": rates(k0), "kappa0.5": rates(ks)}
        tests = {
            "H1_type1_unfiltered_kstar_minus_k0": boot(mat(ks, "unfiltered", "type1"), mat(k0, "unfiltered", "type1"), True),
            "H2_type2_filtered_kstar_minus_k0": boot(mat(ks, "filtered", "type2"), mat(k0, "filtered", "type2"), True),
            "H4_success_unfiltered_kstar_minus_k0": boot(mat(ks, "unfiltered", "success"), mat(k0, "unfiltered", "success"), True),
            "expl_success_filtered_kstar_minus_k0": boot(mat(ks, "filtered", "success"), mat(k0, "filtered", "success"), True),
            "expl_type1_filtered_kstar_minus_k0": boot(mat(ks, "filtered", "type1"), mat(k0, "filtered", "type1"), True),
            "expl_type2_unfiltered_kstar_minus_k0": boot(mat(ks, "unfiltered", "type2"), mat(k0, "unfiltered", "type2"), True),
            "R1_type1_filtered_minus_unfiltered_k0": boot(mat(k0, "filtered", "type1"), mat(k0, "unfiltered", "type1"), False),
            "R1_type2_filtered_minus_unfiltered_k0": boot(mat(k0, "filtered", "type2"), mat(k0, "unfiltered", "type2"), False),
            "expl_success_filtered_minus_unfiltered_k0": boot(mat(k0, "filtered", "success"), mat(k0, "unfiltered", "success"), False),
        }
        r = summary["L3_rates"]["kappa0"]
        h1, h2, h4 = (tests[k] for k in ("H1_type1_unfiltered_kstar_minus_k0", "H2_type2_filtered_kstar_minus_k0",
                                         "H4_success_unfiltered_kstar_minus_k0"))
        coll = sum(v["collisions"] for v in summary["L3_rates"]["kappa0.5"].values())
        minclear = min(v["min_clearance_m"] for v in summary["L3_rates"]["kappa0.5"].values())
        maj = lambda v, c: r[v]["classes"].get(c, 0) > r[v]["timeouts"] / 2  # noqa: E731
        decisions = {
            "H1": "supported" if h1["ci95"][1] < 0 and h1["difference"] <= -0.10 else
                  ("contradicted" if h1["ci95"][0] <= 0 <= h1["ci95"][1] else "partly (interval < 0, effect < 10 points)"),
            "H2": "contradicted" if h2["difference"] <= -0.10 and h2["ci95"][1] < 0 else
                  ("supported" if abs(h2["difference"]) < 0.05 or h2["ci95"][0] <= 0 <= h2["ci95"][1] else "neither"),
            "H3": "supported" if coll == 0 and minclear >= 0.3 - 1e-9 else "contradicted",
            "H4": "supported" if h4["ci95"][0] > 0 else ("contradicted" if h4["ci95"][1] < 0 else "inconclusive"),
            "R1": "supported" if (maj("unfiltered", "filter-blocked") and maj("filtered", "self-stop")
                                  and tests["R1_type1_filtered_minus_unfiltered_k0"]["ci95"][1] < 0
                                  and tests["R1_type2_filtered_minus_unfiltered_k0"]["ci95"][0] > 0) else "contradicted",
        }
        summary["L3_tests"], summary["L3_decisions"] = tests, decisions
        summary["L3_safety_kappa0.5"] = {"collisions": coll, "min_clearance_m": minclear}
        print(json.dumps(summary["L3_rates"], indent=1))
        for k, v in tests.items():
            print(f"{k:48s} {100 * v['difference']:+6.1f} [{100 * v['ci95'][0]:+6.1f}, {100 * v['ci95'][1]:+6.1f}] points")
        print("decisions:", decisions)

    # L4: sampled actions (secondary, descriptive)
    s0, ss = load("L4_fresh_sampled_k0"), load("L4_fresh_sampled_k0.5")
    if s0 is not None and ss is not None and len(s0) == len(ss) == 5000:
        summary["L4_rates"] = {"sampled_kappa0": rates(indicators(s0)), "sampled_kappa0.5": rates(indicators(ss))}
        print("L4:", json.dumps(summary["L4_rates"], indent=1))

    # L5: final vs best checkpoints (kappa 0, mean)
    f0 = load("L5_fresh_final_k0")
    if f0 is not None and k0 is not None:
        f0 = indicators(f0)
        # Only variants with all 5 seeds x 500 scenarios are analysed (L5 was narrowed to the
        # filter-trained agents during the run; see README.md, "Deviations").
        complete = [v for v in ("filtered", "unfiltered") if (f0.variant == v).sum() == 2500]
        f0 = f0[f0.variant.isin(complete)]
        summary["L5_variants_analysed"] = complete
        summary["L5_rates_final"] = rates(f0)
        c1 = {f"{m}_{v}_final_minus_best": boot(mat(f0, v, m), mat(k0, v, m), True)
              for v in complete for m in ("success", "type1", "type2")}
        rf = summary["L5_rates_final"]["filtered"]
        largest = max(rf["classes"], key=rf["classes"].get) if rf["classes"] else None
        summary["L5_tests"] = c1
        summary["L5_decision_C1"] = "supported" if largest == "self-stop" else f"contradicted (largest class: {largest})"
        print(json.dumps(summary["L5_rates_final"], indent=1))
        for k, v in c1.items():
            print(f"{k:40s} {100 * v['difference']:+6.1f} [{100 * v['ci95'][0]:+6.1f}, {100 * v['ci95'][1]:+6.1f}] points")
        print("C1:", summary["L5_decision_C1"])

    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
