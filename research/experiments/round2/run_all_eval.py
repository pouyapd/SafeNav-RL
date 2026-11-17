"""
run_all_eval.py — run every evaluation-only task of round 2 in a fixed order (README.md here).

    python research/experiments/round2/run_all_eval.py [--only e4diag dwa e3 dense]

Each task calls run_eval2.py (or tune_dwa.py) repeatedly until its CSV holds the planned number
of rows, so it can be stopped and restarted at any time. The DWA weights for every later task
come from results/dwa_selection.json (frozen by tune_dwa.py). Stops when the machine is on
battery below --min-battery-pct.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = [sys.executable, str(HERE / "run_eval2.py")]
AGENTS = ["--agent", "filtered=results/checkpoints/filtered", "--agent", "unfiltered=results/checkpoints/unfiltered"]
REJ = ["--set", "env.goal_sampling=rejection"]


def dwa_args():
    sel = json.loads((HERE / "results" / "dwa_selection.json").read_text(encoding="utf-8"))["selected_params"]
    out = ["--baseline", "dwa"]
    for k, v in sel.items():
        out += ["--dwa", f"{k}={v}"]
    return out


def tasks():
    """(group, tag, expected rows, list of argument lists (one per invocation group))."""
    t = [("e4diag", "e4_diag_committed_fresh", 5 * 500,
          [["--agent", "filtered=results/checkpoints/filtered", "--stages", "expert", "--seed-base", "9000000", "--n", "500"]])]
    for filt in ("on", "off"):
        t.append(("dwa", f"dwa_test_{filt}", 300, [
            ["DWA", "--stages", s, "--seed-base", str(b), "--n", "100", "--filter", filt]
            for s, b in (("beginner", 2_000_000), ("intermediate", 2_100_000), ("expert", 2_200_000))]))
    t.append(("dwa", "dwa_fresh_on", 500, [["DWA", "--stages", "expert", "--seed-base", "9000000", "--n", "500"]]))
    for stage, base in (("expert", 7_600_000), ("intermediate", 7_500_000)):
        t.append(("e3", f"e3_{stage}_mean", 13 * 1000,
                  [AGENTS + ["--baseline", "potential_field", "--baseline", "gotogoal", "DWA",
                             "--stages", stage, "--seed-base", str(base), "--n", "1000"] + REJ]))
        t.append(("e3", f"e3_{stage}_sampled", 10 * 1000,
                  [AGENTS + ["--sampled", "--stages", stage, "--seed-base", str(base), "--n", "1000"] + REJ]))
    t.append(("dense", "e3_dense_mean", 13 * 500,
              [AGENTS + ["--baseline", "potential_field", "--baseline", "gotogoal", "DWA", "--stages", "expert",
                         "--seed-base", "6000000", "--n", "500", "--task", "min_obstacles=15",
                         "--task", "max_obstacles=25"] + REJ]))
    return t


def rows(tag):
    p = HERE / "results" / tag / "episodes.csv"
    return len(pd.read_csv(p)) if p.is_file() else 0


def battery_low(limit):
    import psutil
    b = psutil.sensors_battery()
    return b is not None and not b.power_plugged and b.percent < limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--min-battery-pct", type=float, default=30)
    args = ap.parse_args()
    if args.only is None or "dwa" in args.only:
        if battery_low(args.min_battery_pct):
            print("paused: battery"); return 0
        subprocess.run([sys.executable, str(HERE / "tune_dwa.py")], cwd=ROOT, check=True)
    for group, tag, expected, invocations in tasks():
        if args.only is not None and group not in args.only:
            continue
        for attempt in range(40):
            if rows(tag) >= expected:
                break
            if battery_low(args.min_battery_pct):
                print("paused: battery"); return 0
            for inv in invocations:
                inv = [a for x in inv for a in (dwa_args() if x == "DWA" else [x])]
                subprocess.run(RUN + ["--tag", tag, "--deadline-s", "1200"] + inv, cwd=ROOT, check=True)
        print(tag, rows(tag), "/", expected, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
