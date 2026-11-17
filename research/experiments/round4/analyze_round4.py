"""
analyze_round4.py — pre-registered analysis of round 4 (PROTOCOL.md). Reads results/<tag>/ only.

    python research/experiments/round4/analyze_round4.py [--latency latency] [--expert r_expert]
        [--intermediate r_intermediate] [--dense r_dense] [--heading r_heading] [--out summary.json]

Missing inputs are skipped and listed. Writes results/<out>.
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
RES = HERE / "results"
STALLS = ("filter-blocked", "self-stop", "other stall")
TYPES = {"with_filter": "filtered", "without_filter": "unfiltered"}
N_BOOT = 10_000
BINS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]


def load(tag):
    p = RES / tag / "episodes.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    c = df["class"].fillna("")
    return df.assign(success=(df.outcome == "success").astype(float), collision=(df.outcome == "collision").astype(float),
                     deadlock=c.isin(STALLS).astype(float), filter_blocked=(c == "filter-blocked").astype(float),
                     self_stop=(c == "self-stop").astype(float), moving=(c == "moving").astype(float),
                     timeout=(df.outcome == "timeout").astype(float))


def mat(df, method, agent, metric, planner=None):
    g = df[(df.method == method) & (df.agent == agent)]
    if planner is not None:
        g = g[g.planner == planner]
    return g.pivot_table(index="train_seed", columns="episode_seed", values=metric).sort_index(axis=1)


def boot(a, b, paired_seeds=True, seed=0):
    """a - b; scenarios paired; seeds paired (same agents) or b without a seed dimension (1 row)."""
    assert list(a.columns) == list(b.columns), "scenario sets differ"
    A, B = a.to_numpy(float), b.to_numpy(float)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sc = rng.integers(0, A.shape[1], A.shape[1])
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = sa if (paired_seeds and B.shape[0] == A.shape[0]) else rng.integers(0, B.shape[0], B.shape[0])
        d[i] = A[np.ix_(sa, sc)].mean() - B[np.ix_(sb, sc)].mean()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"difference": float(A.mean() - B.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_boot": float(p)}


def seed_level(a, b):
    ra, rb = a.mean(axis=1), b.mean(axis=1)
    if len(rb) != len(ra):
        return {"per_seed": {int(k): float(v) for k, v in ra.items()}, "reference_mean": float(rb.mean())}
    diff = ra.to_numpy() - rb.to_numpy()
    n = len(diff)
    sd = diff.std(ddof=1) if n > 1 else float("nan")
    t = stats.ttest_rel(ra, rb) if n > 1 and sd > 0 else None
    sd_avg = (ra.std(ddof=1) + rb.std(ddof=1)) / 2
    g = diff.mean() / sd_avg * (1 - 3 / (4 * (n - 1) - 1)) if n > 2 and sd_avg > 0 else float("nan")
    return {"per_seed_method": {int(k): float(v) for k, v in ra.items()}, "per_seed_reference": {int(k): float(v) for k, v in rb.items()},
            "mean_difference": float(diff.mean()), "paired_t_p": float(t.pvalue) if t is not None else None,
            "d_z": float(diff.mean() / sd) if n > 1 and sd > 0 else float("nan"), "hedges_g_av": float(g),
            "same_sign_all_seeds": bool(np.all(np.sign(diff) == np.sign(diff.mean())))}


def contrast(df_a, m_a, df_b, m_b, agent, metric, planner_a=None, planner_b=None, seed=0):
    a = mat(df_a, m_a, agent, metric, planner_a)
    b = (mat(df_b, m_b, agent, metric, planner_b) if m_b not in ("planner_only", "path_follower", "path_follower_inflated")
         else mat(df_b, m_b, "map", metric))
    r = boot(a, b, seed=seed)
    r["seed_level"] = seed_level(a, b)
    return r


def holm(tests):
    order = sorted(tests, key=lambda k: tests[k]["p_boot"])
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, tests[k]["p_boot"] * (len(order) - rank)))
        tests[k]["p_holm"] = run
    return tests


def pct(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return {**{f"p{p}": float(np.percentile(x, p)) for p in (50, 90, 95, 99)}, "max": float(x.max()),
            "mean": float(x.mean()), "n": int(x.size)}


def rates(df):
    out = {}
    for (m, pl, agent), g in df.groupby(["method", "planner", "agent"]):
        per = g.groupby("train_seed")
        r = {"episodes": int(len(g))}
        for k in ("success", "deadlock", "filter_blocked", "self_stop", "moving", "collision", "timeout"):
            s = per[k].mean()
            half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")
            r[k] = {"mean": float(g[k].mean()), "seed_sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "t_ci95": [float(s.mean() - half), float(s.mean() + half)] if len(s) > 1 else None}
        r["min_clearance_m"] = float(g.min_obstacle_clearance.min())
        succ = g[g.outcome == "success"]
        r["success_steps_median"] = float(succ.steps.median()) if len(succ) else None
        r["success_path_length_median_m"] = float(succ.path_length_m.median()) if len(succ) else None
        acts = g.activations.sum()
        r["recovery"] = {"activations_per_episode": float(g.activations.mean()),
                         "recovery_success_rate": float(g.activations_success.sum() / acts) if acts else None,
                         "recovery_success_rate_08_supplementary": (float(g.activations_success_08.sum() / acts)
                                                                    if acts and "activations_success_08" in g else None),
                         "repeat_activations": int(g.repeat_activations.sum()),
                         "map_rebuilds_mean": float(pd.to_numeric(g.map_rebuilds, errors="coerce").mean())}
        kn = pd.to_numeric(g.known_obstacles, errors="coerce")
        if kn.notna().any():
            r["known_obstacles_mean"] = float(kn.mean())
        r["planner_calls_per_episode"] = float(g.planner_calls.mean())
        r["act_wall_ms_max_p99_over_episodes"] = float(np.percentile(g.act_wall_ms_max, 99))
        out[f"{m}|{pl}|{agent}"] = r
    return out


def part_p(tag, missing):
    df = load(tag)
    if df is None:
        missing.append("Part P")
        return None
    calls = pd.read_csv(RES / tag / "calls.csv")
    hist = pd.read_csv(RES / tag / "step_hist.csv")
    out = {"episodes": int(len(df)), "collisions": int(df.collision.sum()), "min_clearance_m": float(df.min_obstacle_clearance.min())}
    for pl in ("reference", "fast"):
        c = calls[(calls.planner == pl) & calls.method.isin(["A", "C"])]
        h = hist[(hist.planner == pl) & hist.method.isin(["A", "C"])]
        counts = h[[f"b{i}" for i in range(len(BINS) - 1)]].sum().to_numpy()
        tot = counts.sum()
        over100 = counts[8:].sum()
        cum = np.cumsum(counts) / tot
        p99_bin = int(np.searchsorted(cum, 0.99))
        tail = c[c.wall_ms >= 100]
        out[pl] = {"calls": int(len(c)), "call_wall_ms": pct(c.wall_ms), "call_cpu_ms": pct(c.cpu_ms),
                   "calls_over_100ms": int(len(tail)),
                   "calls_over_100ms_preempted_share": float((tail.cpu_ms < 0.5 * tail.wall_ms).mean()) if len(tail) else None,
                   "steps": int(tot), "steps_over_100ms": int(over100), "share_steps_over_100ms": float(over100 / tot),
                   "step_p99_bin_ms": [BINS[p99_bin], BINS[p99_bin + 1]],
                   "episode_act_max_ms": pct(df[(df.planner == pl)].act_wall_ms_max)}
        if pl == "fast":
            out["P1_call_p99_below_100ms"] = bool(out[pl]["call_wall_ms"]["p99"] < 100.0)
            out["P2_step_p99_below_100ms"] = bool(over100 / tot < 0.01)
    # paired per call: same episode and call index (plans are identical, so call sequences align)
    key = ["method", "agent", "train_seed", "stage", "episode_seed", "call"]
    m = calls[calls.planner == "reference"].merge(calls[calls.planner == "fast"], on=key, suffixes=("_r", "_f"))
    ep = m.groupby(["method", "agent", "train_seed", "episode_seed"])[["cpu_ms_r", "cpu_ms_f"]].sum()
    ratio = ep.cpu_ms_f / ep.cpu_ms_r
    rng = np.random.default_rng(1)
    bs = [ratio.to_numpy()[rng.integers(0, len(ratio), len(ratio))].mean() for _ in range(2000)]
    out["paired_calls"] = int(len(m))
    out["cpu_ratio_fast_over_reference_per_episode"] = {"mean": float(ratio.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                        "median_per_call": float((m.cpu_ms_f / m.cpu_ms_r).median())}
    # P3 non-inferiority and identity
    p3, ok = {}, True
    for mode in ("A", "C"):
        for tname, agent in TYPES.items():
            for metric in ("success", "deadlock"):
                r = contrast(df, mode, df, mode, agent, metric, "fast", "reference")
                p3[f"{mode}_{metric}_{tname}"] = r
                ok &= (r["ci95"][0] > -0.01) if metric == "success" else (r["ci95"][1] < 0.01)
    ok &= out["collisions"] == 0 and out["min_clearance_m"] >= 0.30 - 1e-9
    out["P3"] = p3
    out["P3_non_inferior"] = bool(ok)
    cols = ["outcome", "class", "steps", "intervention_steps", "activations", "activations_success", "exit_reasons", "end_geo_cost"]
    k2 = ["method", "agent", "train_seed", "stage", "episode_seed"]
    mm = df[df.planner == "reference"].merge(df[df.planner == "fast"], on=k2, suffixes=("_r", "_f"))
    same = np.ones(len(mm), bool)
    for c in cols:
        a, b = mm[c + "_r"], mm[c + "_f"]
        same &= (a.fillna("").astype(str) == b.fillna("").astype(str)).to_numpy()
    out["identical_episodes"] = {"matched": int(len(mm)), "identical": int(same.sum())}
    return out


CONDITIONS = {  # condition: (set key, method)
    "localisation_error": ("expert", "C_loc"), "map_error": ("expert", "C_map"), "sensed_map": ("expert", "C_sensed"),
    "intermediate": ("intermediate", "C"), "dense": ("dense", "C"), "heading": ("heading", "C")}


def part_r(sets, missing):
    data = {k: load(t) for k, t in sets.items()}
    for k, d in data.items():
        if d is None:
            missing.append(f"Part R set {k}")
    out = {"rates": {k: rates(d) for k, d in data.items() if d is not None}}
    prim = {}
    for cond, (sk, meth) in CONDITIONS.items():
        d = data.get(sk)
        if d is None or meth not in set(d.method):
            missing.append(f"Part R condition {cond}")
            continue
        for tname, agent in TYPES.items():
            for metric in ("deadlock", "success"):
                prim[f"{cond}|{metric}|{tname}"] = contrast(d, meth, d, "baseline", agent, metric)
    if prim:
        holm(prim)
        dec = {}
        for cond in CONDITIONS:
            sk, meth = CONDITIONS[cond]
            d = data.get(sk)
            for tname, agent in TYPES.items():
                kd, ks = f"{cond}|deadlock|{tname}", f"{cond}|success|{tname}"
                if kd not in prim:
                    continue
                r1 = prim[kd]["difference"] <= -0.10 and prim[kd]["ci95"][1] < 0 and prim[kd]["p_holm"] < 0.05
                r2 = prim[ks]["difference"] > 0 and prim[ks]["ci95"][0] > 0 and prim[ks]["p_holm"] < 0.05
                filt = d[(d.method != "nofilter")]
                safe = bool((filt.collision == 0).all() and filt.min_obstacle_clearance.min() >= 0.30 - 1e-9)
                dec[f"{cond}|{tname}"] = {"R1": bool(r1), "R2": bool(r2), "safe": safe, "claim": bool(r1 and r2 and safe)}
        out["primary"], out["decisions"] = prim, dec
    d = data.get("expert")
    if d is not None and "C" in set(d.method):
        deg = {}
        for meth in ("C_loc", "C_map", "C_sensed"):
            if meth in set(d.method):
                for tname, agent in TYPES.items():
                    for metric in ("deadlock", "success", "moving"):
                        deg[f"{meth}_minus_C|{metric}|{tname}"] = contrast(d, meth, d, "C", agent, metric)
        out["degradation_unadjusted"] = deg
    return out


def part_f(tag, missing):
    d = load(tag)
    if d is None or not {"planner_only", "path_follower", "C"} <= set(d.method):
        missing.append("Part F")
        return None
    out = {"rates": rates(d)}
    comp = {}
    for other in [o for o in ("planner_only", "path_follower", "path_follower_inflated") if o in set(d.method)]:
        for tname, agent in TYPES.items():
            for metric in ("success", "deadlock", "moving"):
                comp[f"C_minus_{other}|{metric}|{tname}"] = contrast(d, "C", d, other, agent, metric)
    out["comparisons_unadjusted"] = comp
    nf = d[d.method == "nofilter"]
    out["nofilter_collision_rate"] = float(nf.collision.mean()) if len(nf) else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latency", default="latency")
    ap.add_argument("--expert", default="r_expert")
    ap.add_argument("--intermediate", default="r_intermediate")
    ap.add_argument("--dense", default="r_dense")
    ap.add_argument("--heading", default="r_heading")
    ap.add_argument("--out", default="summary.json")
    args = ap.parse_args()
    missing = []
    summary = {"P": part_p(args.latency, missing),
               "R": part_r({"expert": args.expert, "intermediate": args.intermediate, "dense": args.dense,
                            "heading": args.heading}, missing),
               "F": part_f(args.expert, missing)}
    summary["missing_inputs"] = missing
    RES.mkdir(parents=True, exist_ok=True)
    (RES / args.out).write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    brief = {"missing": missing}
    if summary["P"]:
        brief["P"] = {k: summary["P"].get(k) for k in ("P1_call_p99_below_100ms", "P2_step_p99_below_100ms", "P3_non_inferior", "identical_episodes")}
    if summary["R"].get("decisions"):
        brief["R"] = summary["R"]["decisions"]
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
