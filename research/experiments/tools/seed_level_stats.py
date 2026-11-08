"""
seed_level_stats.py — supplementary seed-level statistics for an episodes CSV (added after the runs;
it does not replace any pre-registered analysis).

    python research/experiments/tools/seed_level_stats.py --csv research/experiments/e1a/eval/episodes.csv \
        --arm-col arm --a penalty --b control --filter set=fresh --filter actions=mean --out <json>

For each metric, the unit of analysis is the training seed: an episode-level rate is computed per
seed, then reported per seed, as mean +- SD over seeds and as a t-interval over seeds. The two arms
are compared with Welch's t-test on the per-seed rates (n = number of seeds per arm), with Hedges' g
as the standardised effect size. With 5 seeds per arm these tests have low power; they complement
the paired hierarchical bootstrap and should be read with it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "success": lambda d: d.outcome == "success",
    "collision": lambda d: d.outcome == "collision",
    "deadlock": lambda d: d["class"].isin(["filter-blocked", "self-stop", "other stall"]),
    "policy_stall": lambda d: d["class"].isin(["self-stop", "other stall"]),
    "type1": lambda d: d["class"] == "filter-blocked",
    "type2": lambda d: d["class"] == "self-stop",
}


def per_seed(df, metric):
    return df.assign(x=METRICS[metric](df).astype(float)).groupby("train_seed").x.mean()


def describe(s):
    n, m = len(s), float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"per_seed": {int(k): round(float(v), 4) for k, v in s.items()}, "mean": m, "sd": sd,
            "t_ci95": [m - half, m + half], "n_seeds": n}


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--arm-col", default="arm")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--filter", action="append", default=[], metavar="COL=VALUE")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    for f in args.filter:
        k, v = f.split("=", 1)
        df = df[df[k].astype(str) == v]
    res = {"csv": args.csv, "filters": args.filter, "contrast": f"{args.a} - {args.b}", "metrics": {}}
    for metric in METRICS:
        a = per_seed(df[df[args.arm_col] == args.a], metric)
        b = per_seed(df[df[args.arm_col] == args.b], metric)
        t = stats.ttest_ind(a, b, equal_var=False)
        res["metrics"][metric] = {
            args.a: describe(a), args.b: describe(b),
            "difference_of_seed_means": float(a.mean() - b.mean()),
            "welch_t": float(t.statistic) if np.isfinite(t.statistic) else None,
            "welch_p": float(t.pvalue) if np.isfinite(t.pvalue) else None,
            "hedges_g": hedges_g(a, b)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for m, r in res["metrics"].items():
        print(f"{m:13s} {args.a} {r[args.a]['mean']:.3f}±{r[args.a]['sd']:.3f}  {args.b} {r[args.b]['mean']:.3f}±{r[args.b]['sd']:.3f}"
              f"  diff {r['difference_of_seed_means']:+.3f}  p={r['welch_p']}  g={r['hedges_g']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
