"""
verify_primary.py — independent re-computation of the primary endpoints of E1a, E3 (expert) and E4.

    python research/experiments/tools/verify_primary.py

It reads only the per-episode CSV files. It shares no code with e1_evaluate.py or analyze_round2.py,
and it uses a different bootstrap seed (12345). It checks that:
  (1) the point estimates match the stored summaries exactly;
  (2) the 95 % bootstrap bounds agree within 1.5 points;
  (3) there are no collisions.
It writes research/experiments/tools/verify_primary.json and exits with 1 on any mismatch.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "experiments"


def ind(df, metric):
    c = df["class"].fillna("")
    return {"deadlock": c.isin(["filter-blocked", "self-stop", "other stall"]),
            "policy_stall": c.isin(["self-stop", "other stall"]),
            "type1": c == "filter-blocked", "type2": c == "self-stop",
            "success": df.outcome == "success"}[metric].astype(float)


def mat(df, agent_col, agent, metric):
    g = df[df[agent_col] == agent]
    g = g.assign(x=ind(g, metric).to_numpy())
    return g.pivot_table(index="train_seed", columns="episode_seed", values="x").sort_index(axis=1)


def boot(a, b, paired, n=10_000, seed=12345):
    assert list(a.columns) == list(b.columns)
    A, B = a.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        sc = rng.integers(0, A.shape[1], A.shape[1])
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if paired else rng.integers(0, B.shape[0], B.shape[0])
        out[i] = A[np.ix_(sa, sc)].mean() - B[np.ix_(sb, sc)].mean()
    return float(A.mean() - B.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> int:
    checks, ok = [], True

    def check(name, est, ci, ref_est, ref_ci):
        nonlocal ok
        good = abs(est - ref_est) < 1e-9 and all(abs(x - y) <= 0.015 for x, y in zip(ci, ref_ci))
        ok &= good
        checks.append({"check": name, "estimate": est, "ci95": ci, "stored_estimate": ref_est,
                       "stored_ci95": ref_ci, "agrees": bool(good)})

    # E1a
    e1 = pd.read_csv(EXP / "e1a" / "eval" / "episodes.csv")
    ok &= int((e1.outcome == "collision").sum()) == 0
    f = e1[(e1.set == "fresh") & (e1.actions == "mean")]
    stored = {(r["metric"]): r for r in json.loads((EXP / "e1a" / "eval" / "summary.json").read_text())["contrasts"]
              if r["contrast"] == "penalty - control" and r["actions"] == "mean"}
    for m in ("deadlock", "success"):
        est, ci = boot(mat(f, "arm", "penalty", m), mat(f, "arm", "control", m), paired=False)
        check(f"E1a {m} penalty-control", est, ci, stored[m]["difference"], [stored[m]["ci95_low"], stored[m]["ci95_high"]])

    summ = json.loads((EXP / "round2" / "results" / "summary.json").read_text())
    # E3 expert contrasts 1-4
    mean = pd.read_csv(EXP / "round2" / "results" / "e3_expert_mean" / "episodes.csv")
    smp = pd.read_csv(EXP / "round2" / "results" / "e3_expert_sampled" / "episodes.csv")
    for d in (mean, smp):
        ok &= int((d.outcome == "collision").sum()) == 0
    c = summ["E3"]["expert"]["contrasts"]
    for key, a, b, paired in (("1_type1_with_minus_without_mean", (mean, "filtered", "type1"), (mean, "unfiltered", "type1"), False),
                              ("2_type2_with_minus_without_mean", (mean, "filtered", "type2"), (mean, "unfiltered", "type2"), False),
                              ("3_type2_filtered_sampled_minus_mean", (smp, "filtered", "type2"), (mean, "filtered", "type2"), True),
                              ("4_type1_filtered_sampled_minus_mean", (smp, "filtered", "type1"), (mean, "filtered", "type1"), True)):
        est, ci = boot(mat(a[0], "agent", a[1], a[2]), mat(b[0], "agent", b[1], b[2]), paired)
        check(f"E3 expert {key}", est, ci, c[key]["difference"], c[key]["ci95"])

    # E4
    e4 = pd.read_csv(EXP / "round2" / "results" / "e4_mean" / "episodes.csv")
    ok &= int((e4.outcome == "collision").sum()) == 0
    p = summ["E4"]["primary"]
    for arm, key in (("beta_policy", "H4a_beta_minus_control_policy_stall"),
                     ("riskfree", "H4b_riskfree_minus_control_policy_stall")):
        est, ci = boot(mat(e4, "agent", arm, "policy_stall"), mat(e4, "agent", "control", "policy_stall"), False)
        check(f"E4 {key}", est, ci, p[key]["difference"], p[key]["ci95"])

    out = EXP / "tools" / "verify_primary.json"
    out.write_text(json.dumps({"all_agree": bool(ok), "checks": checks}, indent=1), encoding="utf-8")
    for r in checks:
        print(f"{r['check']:55s} {r['estimate']*100:+7.2f} [{r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}]"
              f"  stored [{r['stored_ci95'][0]*100:+.1f}, {r['stored_ci95'][1]*100:+.1f}]  {'OK' if r['agrees'] else 'MISMATCH'}")
    print("all agree:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
