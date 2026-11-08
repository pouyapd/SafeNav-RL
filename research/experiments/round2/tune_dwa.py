"""
tune_dwa.py — pre-registered DWA tuning on validation scenarios only (README.md, "DWA-style baseline").

    python research/experiments/round2/tune_dwa.py

Runs the 12 configurations (w_heading x w_clearance x clearance_min) on 100 expert scenarios
(seed base 1,300,000, filter on) with run_eval2.py, then writes results/dwa_selection.json:
highest success, ties broken by fewer timeouts, then by the lower configuration index.
Resumable: finished configurations are not re-run.
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRID = [dict(w_heading=h, w_clearance=c, clearance_min=m)
        for h, c, m in itertools.product((0.2, 0.5, 1.0), (0.2, 0.5), (0.05, 0.15))]
N, SEED_BASE = 100, 1_300_000


def main() -> int:
    rows = []
    for i, params in enumerate(GRID):
        tag = f"dwa_tune_{i:02d}"
        csv_path = HERE / "results" / tag / "episodes.csv"
        if not (csv_path.is_file() and len(pd.read_csv(csv_path)) >= N):
            cmd = [sys.executable, str(HERE / "run_eval2.py"), "--tag", tag, "--baseline", "dwa",
                   "--stages", "expert", "--seed-base", str(SEED_BASE), "--n", str(N), "--deadline-s", "1800"]
            for k, v in params.items():
                cmd += ["--dwa", f"{k}={v}"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        df = pd.read_csv(csv_path)
        rows.append(dict(index=i, **params, episodes=len(df), success=float((df.outcome == "success").mean()),
                         timeout=float((df.outcome == "timeout").mean()),
                         collision=float((df.outcome == "collision").mean())))
    table = pd.DataFrame(rows)
    best = table.sort_values(["success", "timeout", "index"], ascending=[False, True, True]).iloc[0]
    selection = {"rule": "max success, then min timeout, then lowest index (fixed before tuning)",
                 "validation": {"stage": "expert", "seed_base": SEED_BASE, "episodes": N, "filter": "on"},
                 "selected_index": int(best["index"]), "selected_params": GRID[int(best["index"])],
                 "table": rows}
    (HERE / "results" / "dwa_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("selected:", selection["selected_params"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
