"""
audit_round3.py — independent audit of the round-3 results (PRs #10-#12) before round 4.

    python research/experiments/round4/audit_round3.py

Shares no code with analyze_round3.py. Checks:
  A. the four primary contrasts and the eight secondary ones (point estimates exactly; 95 % bounds within
     1.5 points with another bootstrap seed), the Holm step and the claim rule;
  B. seed-level results: per-seed rates, paired t-test p, d_z;
  C. the rates quoted in PROTOCOL.md / README.md / manuscript (table values recomputed from raw rows);
  D. structure of the raw data: 25,000 unique rows, 5 methods x 10 agents x 500 scenarios, seeds
     8,500,000-8,500,499 only, 0 collisions, minimum clearance, baseline rows untouched by recovery;
  E. the 776 rows written by the invocation without a run record (the first 776 rows of the file):
     all are replayed with the current code and compared on every non-timing column;
  F. a further replay sample: 20 scenarios x 5 methods for two other agents;
  G. pairing: every method episode is identical to the baseline episode when no activation occurred.
Writes research/experiments/round4/audit/audit_round3.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
R3 = ROOT / "research" / "experiments" / "round3"
OUT = Path(__file__).resolve().parent / "audit"
STALLS = ("filter-blocked", "self-stop", "other stall")
TIMING = ["planner_time_s", "controller_time_s", "field_setup_s", "episode_wall_s", "planner_max_call_ms"]


def ind(df, metric):
    c = df["class"].fillna("")
    return {"deadlock": c.isin(STALLS), "success": df.outcome == "success", "moving": c == "moving"}[metric].astype(float).to_numpy()


def mat(df, method, agent, metric):
    g = df[(df.method == method) & (df.agent == agent)]
    return g.assign(_x=ind(g, metric)).pivot_table(index="train_seed", columns="episode_seed", values="_x").sort_index(axis=1)


def boot(a, b, n=10_000, seed=4242):
    A, B = a.to_numpy(float), b.to_numpy(float)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        sc = rng.integers(0, A.shape[1], A.shape[1])
        sd = rng.integers(0, A.shape[0], A.shape[0])
        d[i] = A[np.ix_(sd, sc)].mean() - B[np.ix_(sd, sc)].mean()
    return float(A.mean() - B.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def replay(tag, agents, methods, seed_base, n, out):
    cmd = [sys.executable, str(R3 / "run_recovery.py"), "--tag", tag, "--out-root", str(out), "--methods", *methods,
           "--stages", "expert", "--seed-base", str(seed_base), "--n", str(n), "--set", "env.goal_sampling=rejection",
           "--deadline-s", "6000"]
    for a in agents:
        cmd += ["--agent", a]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return pd.read_csv(out / tag / "episodes.csv")


def compare(new, old, key):
    cols = [c for c in old.columns if c not in TIMING and c not in key]
    m = new.merge(old, on=key, suffixes=("_n", "_o"))
    bad = 0
    for c in cols:
        a, b = m[c + "_n"], m[c + "_o"]
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
            eq = (np.isclose(x, y, rtol=0, atol=1e-9) | (x.isna() & y.isna()) | ((x == y)))
        else:
            eq = a.fillna("").astype(str) == b.fillna("").astype(str)
        bad += int((~eq).sum())
    return len(m), bad


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks, findings = [], []

    def check(sec, name, ok, detail=None):
        checks.append({"section": sec, "check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            findings.append(f"[{sec}] {name}: {detail}")

    df = pd.read_csv(R3 / "results" / "main" / "episodes.csv")
    s = json.loads((R3 / "results" / "main" / "summary.json").read_text())
    types = {"with_filter": "filtered", "without_filter": "unfiltered"}

    # A, B
    for fam, methods in (("primary", ["C"]), ("secondary_family", ["A", "B"])):
        for m in methods:
            for tname, agent in types.items():
                for metric in ("deadlock", "success"):
                    ref = s[fam][f"{m}_minus_baseline_{metric}_{tname}"]
                    a, b = mat(df, m, agent, metric), mat(df, "baseline", agent, metric)
                    est, ci = boot(a, b)
                    check("A", f"{m}-baseline {metric} {tname}",
                          abs(est - ref["difference"]) < 1e-12 and all(abs(x - y) <= 0.015 for x, y in zip(ci, ref["ci95"])),
                          {"recomputed": [est, ci], "stored": [ref["difference"], ref["ci95"]]})
                    ra, rb = a.mean(axis=1), b.mean(axis=1)
                    t = stats.ttest_rel(ra, rb)
                    dz = (ra - rb).mean() / (ra - rb).std(ddof=1)
                    sl = ref["seed_level"]
                    check("B", f"seed level {m}-baseline {metric} {tname}",
                          abs(t.pvalue - sl["paired_t_p"]) < 1e-12 and abs(dz - sl["d_z"]) < 1e-9
                          and all(abs(ra[k] - sl["per_seed_method"][str(k)]) < 1e-12 for k in ra.index)
                          and bool((np.sign(ra - rb) == np.sign(est)).all()),
                          {"p": [float(t.pvalue), sl["paired_t_p"]], "d_z": [float(dz), sl["d_z"]]})
    ps = sorted(v["p_boot"] for v in s["primary"].values())
    check("A", "primary Holm: all bootstrap p = 0 -> adjusted 0 (< 1e-4)", ps[-1] == 0.0 and all(v["p_holm"] == 0.0 for v in s["primary"].values()))
    for tname in types:
        d, u = s["primary"][f"C_minus_baseline_deadlock_{tname}"], s["primary"][f"C_minus_baseline_success_{tname}"]
        claim = d["difference"] <= -0.10 and d["ci95"][1] < 0 and u["difference"] > 0 and u["ci95"][0] > 0
        check("A", f"claim rule {tname}", claim == s["decisions"][tname]["claim"] and claim)

    # C: quoted table values
    docs = {"PROTOCOL": (R3 / "PROTOCOL.md").read_text(encoding="utf-8"), "README": (ROOT / "README.md").read_text(encoding="utf-8"),
            "manuscript": (ROOT / "paper_draft" / "manuscript.tex").read_text(encoding="utf-8")}
    lab = {"baseline": "baseline", "A": "A", "B": "B", "C": "**C**", "C_euclid": "C_euclid"}
    for agent, alab in (("filtered", "With filter"), ("unfiltered", "Without filter")):
        for m in ("baseline", "A", "B", "C", "C_euclid"):
            g = df[(df.method == m) & (df.agent == agent)]
            per = g.assign(su=ind(g, "success"), de=ind(g, "deadlock")).groupby("train_seed")[["su", "de"]].mean()
            su, de, mv = ind(g, "success").mean() * 100, ind(g, "deadlock").mean() * 100, ind(g, "moving").mean() * 100
            needle_tex = f"${su:.1f}\\pm{per.su.std(ddof=1) * 100:.1f}$ & ${de:.1f}\\pm{per.de.std(ddof=1) * 100:.1f}$ & {mv:.1f}"
            check("C", f"manuscript Table 6 row {agent} {m}", needle_tex in docs["manuscript"], needle_tex)
            needle_readme = f"{su:.1f} % / {de:.1f} % / {mv:.1f} %"
            check("C", f"README round-3 table {agent} {m}", needle_readme in docs["README"], needle_readme)
            row = f"{su:.1f} ± {per.su.std(ddof=1) * 100:.1f}"
            check("C", f"PROTOCOL results table {agent} {m}", row in docs["PROTOCOL"], row)

    # D
    key = ["method", "agent", "train_seed", "stage", "episode_seed"]
    check("D", "25,000 unique rows", len(df) == 25_000 and not df.duplicated(key).any())
    check("D", "balanced design", (df.groupby(["method", "agent", "train_seed"]).size() == 500).all() and df.method.nunique() == 5)
    check("D", "seeds 8,500,000-8,500,499 only", df.episode_seed.min() == 8_500_000 and df.episode_seed.max() == 8_500_499)
    check("D", "0 collisions, clearance >= 0.30 m", (df.outcome != "collision").all() and df.min_obstacle_clearance.min() >= 0.30 - 1e-9,
          float(df.min_obstacle_clearance.min()))
    b = df[df.method == "baseline"]
    check("D", "baseline rows have no recovery activity", (b.activations == 0).all() and (b.planner_calls == 0).all())
    rec = json.loads((R3 / "results" / "main" / "run_record.json").read_text())["invocations"]
    check("D", "run record: recorded invocations cover the end of the run",
          rec[-1]["episodes_completed"] == 25_000 and not any(i["code_modified_since_commit"] for i in rec),
          {"invocations": len(rec), "commits": sorted({i["git_commit"][:7] for i in rec}),
           "first_recorded_completed": rec[0]["episodes_completed"]})

    # G: pairing
    bb = b.set_index(["agent", "train_seed", "episode_seed"])
    for m in ("B", "C", "C_euclid"):
        g = df[(df.method == m) & (df.activations == 0)].set_index(["agent", "train_seed", "episode_seed"])
        j = g.join(bb[["outcome", "steps"]], rsuffix="_b")
        check("G", f"{m}: episodes without activation identical to baseline", ((j.outcome == j.outcome_b) & (j.steps == j.steps_b)).all(), len(j))

    # E: the 776 unrecorded rows
    first = df.iloc[:776]
    tmp = OUT / "replay"
    combos = first.groupby(["agent", "train_seed", "method"]).episode_seed.agg(["min", "max", "size"]).reset_index()
    newrows = []
    for r in combos.itertuples():
        assert r.train_seed == 0 and r.agent == "filtered"
        newrows.append(replay(f"unrecorded_{r.method}", ["filtered=results/checkpoints/filtered"], [r.method], int(r.min), int(r.size), tmp))
    new = pd.concat(newrows)
    new = new[new.train_seed == 0]
    n, bad = compare(new, first, key)
    check("E", "the 776 rows without a run record replay identically (all non-timing columns)", n == 776 and bad == 0,
          {"matched": n, "differing_cells": bad, "combos": combos.to_dict(orient="records")})

    # F: further sample
    new = replay("sample", ["unfiltered=results/checkpoints/unfiltered"], ["baseline", "A", "B", "C", "C_euclid"], 8_500_200, 20, tmp)
    new = new[new.train_seed.isin([1, 3])]
    n, bad = compare(new, df, key)
    check("F", "replay sample (unfiltered seeds 1 and 3, scenarios 200-219, 5 methods)", n == 200 and bad == 0, {"matched": n, "differing_cells": bad})

    rep = {"n_checks": len(checks), "n_failed": len(findings), "findings": findings, "checks": checks}
    (OUT / "audit_round3.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(len(checks), "checks,", len(findings), "failed")
    for f in findings:
        print(" -", f[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
