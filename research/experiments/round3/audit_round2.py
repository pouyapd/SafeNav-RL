"""
audit_round2.py — independent audit of the round-2 results (PRs #7-#9) before round 3.

    python research/experiments/round3/audit_round2.py [--replay N]

It shares no analysis code with e1_evaluate.py, analyze_round2.py or verify_primary.py. It checks:
  A. every bootstrap contrast stored in e1a/eval/summary.json and round2/results/summary.json:
     point estimate recomputed exactly; 95 % bounds within 1.5 points (different bootstrap seed);
  B. seed-level statistics (per-seed rates, mean, SD, t-interval, Welch p, Hedges' g);
  C. Wilson intervals and rates of the deterministic baselines;
  D. separation of scenario sets (seed ranges per purpose; overlaps listed);
  E. run records: steps, overrides per arm, commits, modified-code flag, power state;
  F. checkpoints: SHA-256 in the evaluation records == committed copies;
  G. determinism: N episodes per checked file are replayed from the committed checkpoint copies
     and compared with the stored rows (outcome, steps, stall class);
  H. numbers quoted in README.md, the manuscript and the experiment READMEs (curated list).
Writes research/experiments/round3/audit/audit_round2.json and prints a findings list.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "experiments"
R2 = EXP / "round2" / "results"
OUT = EXP / "round3" / "audit"
STALLS = ("filter-blocked", "self-stop", "other stall")
TOL = 0.015


def indicator(df, metric):
    c = df["class"].fillna("")
    m = {"success": df.outcome == "success", "collision": df.outcome == "collision",
         "timeout": df.outcome == "timeout", "deadlock": c.isin(STALLS),
         "policy_stall": c.isin(["self-stop", "other stall"]), "type1": c == "filter-blocked",
         "type2": c == "self-stop"}[metric]
    return m.astype(float).to_numpy()


def mat(df, col, name, metric):
    g = df[df[col] == name]
    g = g.assign(_x=indicator(g, metric))
    return g.pivot_table(index="train_seed", columns=[c for c in ("stage", "episode_seed") if c in g], values="_x").sort_index(axis=1)


def boot(a, b, paired, n=10_000, seed=777):
    assert list(a.columns) == list(b.columns)
    A, B = a.to_numpy(float), b.to_numpy(float)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        sc = rng.integers(0, A.shape[1], A.shape[1])
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if paired else rng.integers(0, B.shape[0], B.shape[0])
        d[i] = A[np.ix_(sa, sc)].mean() - B[np.ix_(sb, sc)].mean()
    return float(A.mean() - B.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


class Audit:
    def __init__(self):
        self.checks, self.findings = [], []

    def check(self, section, name, ok, detail=None):
        self.checks.append({"section": section, "check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.findings.append(f"[{section}] {name}: {detail}")

    def contrast(self, section, name, est, ci, ref_est, ref_ci):
        ok = abs(est - ref_est) < 1e-9 and all(abs(x - y) <= TOL for x, y in zip(ci, ref_ci))
        self.check(section, name, ok, {"recomputed": [est, ci], "stored": [ref_est, ref_ci]})


def section_a(au):
    # E1a: 20 stored contrasts
    e1 = pd.read_csv(EXP / "e1a" / "eval" / "episodes.csv")
    for row in json.loads((EXP / "e1a" / "eval" / "summary.json").read_text())["contrasts"]:
        a, b = row["contrast"].split(" - ")
        f = e1[(e1.set == "fresh") & (e1.actions == row["actions"])]
        est, ci = boot(mat(f, "arm", a, row["metric"]), mat(f, "arm", b, row["metric"]), paired=False)
        au.contrast("A", f"E1a {row['contrast']} {row['metric']} {row['actions']}", est, ci,
                    row["difference"], [row["ci95_low"], row["ci95_high"]])
    # E1a Holm over the two primary tests
    prim = [r for r in json.loads((EXP / "e1a" / "eval" / "summary.json").read_text())["contrasts"] if r["primary"]]
    ps = sorted(r["p_boot"] for r in prim)
    holm = [min(1, ps[0] * 2), min(1, max(ps[0] * 2, ps[1]))]
    stored = sorted(r["p_holm"] for r in prim)
    au.check("A", "E1a Holm adjustment (monotone step-down)", all(abs(x - y) < 1e-9 for x, y in zip(holm, stored)),
             {"recomputed": holm, "stored": stored})

    s = json.loads((R2 / "summary.json").read_text())
    for stage in ("expert", "intermediate"):
        mean, smp = pd.read_csv(R2 / f"e3_{stage}_mean" / "episodes.csv"), pd.read_csv(R2 / f"e3_{stage}_sampled" / "episodes.csv")
        spec = {"1_type1_with_minus_without_mean": ((mean, "filtered"), (mean, "unfiltered"), "type1", False),
                "2_type2_with_minus_without_mean": ((mean, "filtered"), (mean, "unfiltered"), "type2", False),
                "3_type2_filtered_sampled_minus_mean": ((smp, "filtered"), (mean, "filtered"), "type2", True),
                "4_type1_filtered_sampled_minus_mean": ((smp, "filtered"), (mean, "filtered"), "type1", True)}
        for k, (x, y, m, paired) in spec.items():
            est, ci = boot(mat(x[0], "agent", x[1], m), mat(y[0], "agent", y[1], m), paired)
            ref = s["E3"][stage]["contrasts"][k]
            au.contrast("A", f"E3 {stage} {k}", est, ci, ref["difference"], ref["ci95"])
        est, ci = boot(mat(mean, "agent", "filtered", "success"), mat(mean, "agent", "unfiltered", "success"), False)
        ref = s["E3"][stage]["success_with_minus_without_mean"]
        au.contrast("A", f"E3 {stage} success with-without", est, ci, ref["difference"], ref["ci95"])
    e4 = pd.read_csv(R2 / "e4_mean" / "episodes.csv")
    for arm in ("beta_policy", "riskfree"):
        for m in ("success", "type1", "type2", "deadlock", "collision"):
            est, ci = boot(mat(e4, "agent", arm, m), mat(e4, "agent", "control", m), False)
            ref = s["E4"]["secondary_mean"][f"{arm}_minus_control_{m}"]
            au.contrast("A", f"E4 {arm}-control {m}", est, ci, ref["difference"], ref["ci95"])
    for k, arm in (("H4a_beta_minus_control_policy_stall", "beta_policy"), ("H4b_riskfree_minus_control_policy_stall", "riskfree")):
        est, ci = boot(mat(e4, "agent", arm, "policy_stall"), mat(e4, "agent", "control", "policy_stall"), False)
        ref = s["E4"]["primary"][k]
        au.contrast("A", f"E4 {k}", est, ci, ref["difference"], ref["ci95"])
    au.check("A", "E4 decision rule applied as pre-registered",
             all((ref["difference"] <= -0.10 and ref["ci95"][1] < 0 and ref["p_holm"] < 0.05) == s["E4"]["decisions"][k]
                 for k, ref in s["E4"]["primary"].items()), s["E4"]["decisions"])
    return e1, e4, s


def seedstats(a, b):
    t = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else float("nan")
    return float(t.pvalue), float(g)


def section_b(au, e1, e4):
    def per_seed(df, col, name, metric):
        g = df[df[col] == name]
        return g.assign(_x=indicator(g, metric)).groupby("train_seed")._x.mean()

    f = e1[(e1.set == "fresh") & (e1.actions == "mean")]
    stored = json.loads((EXP / "e1a" / "eval" / "seed_level_fresh_mean.json").read_text())["metrics"]
    for m in ("success", "deadlock", "type1", "type2", "policy_stall"):
        a, b = per_seed(f, "arm", "penalty", m), per_seed(f, "arm", "control", m)
        p, g = seedstats(a, b)
        ref = stored[m]
        ok = abs(a.mean() - ref["penalty"]["mean"]) < 1e-12 and abs(a.std(ddof=1) - ref["penalty"]["sd"]) < 1e-12 \
            and abs(p - ref["welch_p"]) < 1e-9 and abs(g - ref["hedges_g"]) < 1e-9
        au.check("B", f"E1a seed-level {m}", ok, {"welch_p": [p, ref["welch_p"]], "g": [g, ref["hedges_g"]]})
    for arm in ("riskfree", "beta_policy"):
        stored = json.loads((R2 / f"e4_seed_level_{arm}_vs_control.json").read_text())["metrics"]
        for m in ("success", "deadlock", "type1", "policy_stall"):
            a, b = per_seed(e4, "agent", arm, m), per_seed(e4, "agent", "control", m)
            p, g = seedstats(a, b)
            ref = stored[m]
            ok = abs(p - ref["welch_p"]) < 1e-9 and abs(g - ref["hedges_g"]) < 1e-9
            au.check("B", f"E4 seed-level {arm} {m}", ok, {"welch_p": [p, ref["welch_p"]], "g": [g, ref["hedges_g"]]})
        # t-interval from analyze_round2 (policy-side stall)
    s = json.loads((R2 / "summary.json").read_text())["E4"]["policy_stall_seed"]
    for arm, ref in s.items():
        x = per_seed(e4, "agent", arm, "policy_stall")
        half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
        ok = abs(x.mean() - ref["mean"]) < 1e-12 and all(abs(u - v) < 1e-3 for u, v in zip([x.mean() - half, x.mean() + half], ref["t_ci95"]))
        au.check("B", f"E4 t-interval policy_stall {arm}", ok, {"recomputed": [x.mean() - half, x.mean() + half], "stored": ref["t_ci95"]})


def section_c(au):
    s = json.loads((R2 / "summary.json").read_text())
    for tag in ("dwa_test_on", "dwa_test_off", "dwa_fresh_on"):
        d = pd.read_csv(R2 / tag / "episodes.csv")
        for stage, g in d.groupby("stage"):
            k, n = int((g.outcome == "success").sum()), len(g)
            ref = s["DWA"][tag][f"dwa|{stage}"]
            ok = abs(k / n - ref["success"]) < 1e-12 and all(abs(u - v) < 1e-9 for u, v in zip(wilson(k, n), ref["success_wilson95"]))
            au.check("C", f"{tag} {stage} success + Wilson", ok, {"k": k, "n": n})
    for tag in ("e3_expert_mean", "e3_intermediate_mean", "e3_dense_mean"):
        d = pd.read_csv(R2 / tag / "episodes.csv")
        for agent in ("potential_field", "gotogoal", "dwa"):
            g = d[d.agent == agent]
            k, n = int((g.outcome == "success").sum()), len(g)
            sec = s["E3"][tag.split("_")[1]]["rates_mean"] if tag != "e3_dense_mean" else s["E3_dense_exploratory"]["rates"]
            ref = sec[f"{agent}|expert" if "expert" in tag or "dense" in tag else f"{agent}|intermediate"]
            ok = abs(k / n - ref["success"]) < 1e-12 and all(abs(u - v) < 1e-9 for u, v in zip(wilson(k, n), ref["success_wilson95"]))
            au.check("C", f"{tag} {agent} success + Wilson", ok, {"k": k, "n": n})


PURPOSE = {  # seed base -> purpose (from the protocols)
    (1_000_000, 1_300_000): "validation (training checkpoint selection; L2 kappa tuning at 1.2M)",
    (1_300_000, 1_300_100): "DWA tuning (validation)",
    (2_000_000, 2_300_000): "committed test set",
    (5_000_000, 5_000_500): "E1a primary (fresh)",
    (6_000_000, 6_000_500): "E3-dense",
    (6_500_000, 6_500_500): "E4 primary",
    (7_500_000, 7_501_000): "E3 intermediate",
    (7_600_000, 7_601_000): "E3 expert",
    (9_000_000, 9_000_500): "fresh set (L3)",
}


def section_d(au):
    rng_by_file = {}
    for f in sorted(EXP.rglob("episodes.csv")):
        d = pd.read_csv(f, usecols=lambda c: c in ("episode_seed", "set", "stage"))
        rng_by_file[str(f.relative_to(ROOT)).replace("\\", "/")] = sorted(set(d.episode_seed.astype(int)))
    out = {}
    for f, seeds in rng_by_file.items():
        purposes = set()
        for s in seeds:
            hit = [p for (lo, hi), p in PURPOSE.items() if lo <= s < hi]
            purposes.add(hit[0] if hit else "UNDECLARED")
        out[f] = sorted(purposes)
        au.check("D", f"seed ranges of {f} are declared", "UNDECLARED" not in purposes, sorted(purposes))
    # validation seeds used by the trainer: base 1,000,000 + 100,000 * stage + i, i < 20
    val = {1_000_000 + 100_000 * st + i for st in range(3) for i in range(20)}
    l2 = set(rng_by_file.get("research/experiments/lightweight/results/L2_val_k0/episodes.csv", []))
    au.check("D", "L2 kappa-tuning scenarios disjoint from the trainer's validation scenarios",
             not (val & l2), f"{len(val & l2)} shared seeds (1,200,000-1,200,019): the kappa rule used 20 scenarios "
                             "that also selected the checkpoints; both are validation uses, no test leakage")
    test_like = ["5_000_000", "6_500_000", "7_500_000", "7_600_000", "6_000_000"]
    tune = set(rng_by_file.get("research/experiments/round2/results/dwa_tune_00/episodes.csv", []))
    others = set().union(*[set(v) for k, v in rng_by_file.items() if "dwa_tune" not in k])
    au.check("D", "DWA tuning scenarios unused elsewhere", not (tune & others), len(tune & others))
    return out


def section_e(au):
    out = {}
    for rec_dir, arms in ((EXP / "e1a" / "records", {"control": {}, "penalty": {"safety.correction_penalty": 1.0}}),
                          (EXP / "round2" / "records", {"riskfree": {"reward.risk_penalty_beta": 0.0},
                                                        "beta_policy": {"network.distribution": "beta"}})):
        for f in sorted(rec_dir.glob("train_*.json")):
            inv = json.loads(f.read_text())["invocations"]
            arm = inv[0]["arm"]
            last = inv[-1]
            steps = last["chunks"][-1]["timesteps"] if last.get("chunks") else None
            ok = (inv[0]["overrides"] == arms[arm] and last["finished"] and steps == 301_056
                  and not any(i["code_modified_since_commit"] for i in inv)
                  and all(i["on_mains_power"] for i in inv))
            out[f.name] = {"commits": sorted({i["git_commit"][:7] for i in inv}), "invocations": len(inv), "steps": steps}
            au.check("E", f"run record {f.name}", ok, out[f.name] | {"overrides": inv[0]["overrides"]})
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def section_f(au):
    copies = {"control": EXP / "e1a" / "checkpoints" / "control", "penalty": EXP / "e1a" / "checkpoints" / "penalty",
              "riskfree": EXP / "round2" / "checkpoints" / "riskfree", "beta_policy": EXP / "round2" / "checkpoints" / "beta_policy"}
    for tag in ("e4_mean", "e4_sampled"):
        inv = json.loads((R2 / tag / "run_record.json").read_text())["invocations"][0]
        for key, v in inv["checkpoints"].items():
            arm, s = key.rsplit("_seed", 1)
            p = copies[arm] / f"seed_{s}" / "best.pt"
            au.check("F", f"{tag} {key} hash == committed copy", p.is_file() and sha(p) == v["sha256"], str(p.relative_to(ROOT)))
    for arm in ("control", "penalty"):
        for s in range(5):
            for c in ("best", "final"):
                p, r = copies[arm] / f"seed_{s}" / f"{c}.pt", ROOT / "runs" / "e1a" / arm / "filtered" / f"seed_{s}" / "checkpoints" / f"{c}.pt"
                if r.is_file():
                    au.check("F", f"E1a {arm} seed {s} {c}.pt copy identical to run output", sha(p) == sha(r))


def section_g(au, n):
    """Replay n scenarios per (file, agent, seed) from committed checkpoint copies with run_eval2.py."""
    if n <= 0:
        return {}
    tmp = OUT / "replay"
    jobs = [  # (stored tag, csv filter, run_eval2 args)
        ("e4_mean", ["--agent", "control=research/experiments/e1a/checkpoints/control",
                     "--agent", "beta_policy=research/experiments/round2/checkpoints/beta_policy",
                     "--agent", "riskfree=research/experiments/round2/checkpoints/riskfree",
                     "--stages", "expert", "--seed-base", "6500000", "--n", str(n)]),
        ("e3_expert_mean", ["--agent", "filtered=results/checkpoints/filtered", "--agent", "unfiltered=results/checkpoints/unfiltered",
                            "--baseline", "dwa", "--dwa", "w_heading=0.2", "--dwa", "w_clearance=0.5", "--dwa", "clearance_min=0.05",
                            "--stages", "expert", "--seed-base", "7600000", "--n", str(n), "--set", "env.goal_sampling=rejection"]),
        ("e3_expert_sampled", ["--agent", "filtered=results/checkpoints/filtered", "--sampled",
                               "--stages", "expert", "--seed-base", "7600000", "--n", str(n), "--set", "env.goal_sampling=rejection"]),
    ]
    res = {}
    for tag, args in jobs:
        subprocess.run([sys.executable, str(EXP / "round2" / "run_eval2.py"), "--tag", f"replay_{tag}",
                        "--out-root", str(tmp), "--checkpoint-name", "best", "--deadline-s", "3000"] + args,
                       cwd=ROOT, check=True, capture_output=True)
        new = pd.read_csv(tmp / f"replay_{tag}" / "episodes.csv")
        old = pd.read_csv(R2 / tag / "episodes.csv")
        key = ["agent", "train_seed", "stage", "episode_seed"]
        m = new.merge(old, on=key, suffixes=("_new", "_old"))
        same = ((m.outcome_new == m.outcome_old) & (m.steps_new == m.steps_old)
                & (m.class_new.fillna("") == m.class_old.fillna("")))
        res[tag] = {"replayed": len(new), "matched_rows": len(m), "identical": int(same.sum())}
        au.check("G", f"replay {tag}: {len(m)} episodes identical", len(m) == len(new) and bool(same.all()), res[tag])
    e1 = pd.read_csv(EXP / "e1a" / "eval" / "episodes.csv")
    # E1a rows are produced by e1_evaluate.py; replay a slice with run_eval2.py (same env, same policy class)
    subprocess.run([sys.executable, str(EXP / "round2" / "run_eval2.py"), "--tag", "replay_e1a_fresh", "--out-root", str(tmp),
                    "--agent", "penalty=research/experiments/e1a/checkpoints/penalty", "--agent", "control=research/experiments/e1a/checkpoints/control",
                    "--stages", "expert", "--seed-base", "5000000", "--n", str(n), "--deadline-s", "3000"], cwd=ROOT, check=True, capture_output=True)
    new = pd.read_csv(tmp / "replay_e1a_fresh" / "episodes.csv").rename(columns={"agent": "arm"})
    old = e1[(e1.set == "fresh") & (e1.actions == "mean")]
    m = new.merge(old, on=["arm", "train_seed", "episode_seed"], suffixes=("_new", "_old"))
    same = (m.outcome_new == m.outcome_old) & (m.steps_new == m.steps_old) & (m.class_new.fillna("") == m.class_old.fillna(""))
    res["e1a_fresh"] = {"replayed": len(new), "matched_rows": len(m), "identical": int(same.sum())}
    au.check("G", "replay E1a fresh (run_eval2 vs e1_evaluate)", len(m) == len(new) and bool(same.all()), res["e1a_fresh"])
    return res


def fmt(x, sign=True):
    v = round(x * 100, 1)
    s = f"{v:+.1f}" if sign else f"{v:.1f}"
    return s.replace("-", "−")


def section_h(au, s):
    """Quoted numbers: each expected string must occur in the named documents."""
    docs = {"README": (ROOT / "README.md").read_text(encoding="utf-8"),
            "manuscript": (ROOT / "paper_draft" / "manuscript.tex").read_text(encoding="utf-8"),
            "e1a": (EXP / "e1a" / "README.md").read_text(encoding="utf-8"),
            "round2": (EXP / "round2" / "README.md").read_text(encoding="utf-8")}
    e1 = {(r["contrast"], r["metric"], r["actions"]): r for r in json.loads((EXP / "e1a" / "eval" / "summary.json").read_text())["contrasts"]}

    def ci(lo, hi):
        return f"[{fmt(lo)}, {fmt(hi)}]"

    def tex(x):
        return f"{round(x * 100, 1):.1f}"

    exp = []
    r = e1[("penalty - control", "deadlock", "mean")]
    exp += [("README", f"−5.1 points {ci(r['ci95_low'], r['ci95_high'])}"), ("e1a", f"{ci(r['ci95_low'], r['ci95_high'])}"),
            ("manuscript", f"$-5.1$ points [$-{tex(-r['ci95_low'])}$, {tex(r['ci95_high'])}]")]
    r = e1[("penalty - control", "success", "mean")]
    exp += [("README", f"{ci(r['ci95_low'], r['ci95_high'])}"), ("manuscript", f"[$-{tex(-r['ci95_low'])}$, {tex(r['ci95_high'])}]")]
    for metric, txt in (("type2", "−19.7"), ("type1", "+14.0")):
        r = e1[("penalty - control", metric, "mean")]
        exp += [("e1a", f"{txt} points {ci(r['ci95_low'], r['ci95_high'])}"),
                ("manuscript", f"{tex(abs(r['difference']))} points [{tex(min(abs(r['ci95_low']), abs(r['ci95_high'])))}, {tex(max(abs(r['ci95_low']), abs(r['ci95_high'])))}]")]
    e3 = s["E3"]["expert"]["contrasts"]
    for k, key in (("1_type1_with_minus_without_mean", "manuscript"), ("2_type2_with_minus_without_mean", "manuscript"),
                   ("3_type2_filtered_sampled_minus_mean", "manuscript"), ("4_type1_filtered_sampled_minus_mean", "manuscript")):
        d, (lo, hi) = e3[k]["difference"], e3[k]["ci95"]
        exp.append(("round2", f"{fmt(d)} {ci(lo, hi)}"))
        a, b = sorted([abs(lo), abs(hi)])
        exp.append((key, f"{tex(abs(d))} points [{tex(a)}, {tex(b)}]"))
    for k in ("H4a_beta_minus_control_policy_stall", "H4b_riskfree_minus_control_policy_stall"):
        d, (lo, hi) = s["E4"]["primary"][k]["difference"], s["E4"]["primary"][k]["ci95"]
        exp += [("round2", f"{fmt(d)} {ci(lo, hi)}"), ("README", f"{fmt(d)} points {ci(lo, hi)}" if "beta" in k else f"{fmt(d)} points {ci(lo, hi)}"),
                ("manuscript", f"$-{tex(-d)}$ points [$-{tex(-lo)}$, $-{tex(-hi)}$]")]
    for arm, m in (("riskfree", "success"), ("beta_policy", "type1"), ("riskfree", "type1")):
        r = s["E4"]["secondary_mean"][f"{arm}_minus_control_{m}"]
        exp += [("round2", f"{fmt(r['difference'])} {ci(*r['ci95'])}"),
                ("manuscript", f"{tex(r['difference'])} points [{tex(r['ci95'][0])}, {tex(r['ci95'][1])}]")]
    res = []
    for doc, needle in exp:
        found = needle in docs[doc]
        res.append({"doc": doc, "expected": needle, "found": found})
        au.check("H", f"{doc} quotes '{needle}'", found)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=int, default=20, help="scenarios replayed per agent and file (0 = skip)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    au = Audit()
    e1, e4, s = section_a(au)
    section_b(au, e1, e4)
    section_c(au)
    seeds = section_d(au)
    records = section_e(au)
    section_f(au)
    replay = section_g(au, args.replay)
    quotes = section_h(au, s)
    report = {"n_checks": len(au.checks), "n_failed": len(au.findings), "findings": au.findings,
              "seed_purposes": seeds, "records": records, "replay": replay, "quotes": quotes, "checks": au.checks}
    (OUT / "audit_round2.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"{len(au.checks)} checks, {len(au.findings)} failed")
    for f in au.findings:
        print(" -", f[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
