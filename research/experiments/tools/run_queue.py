"""
run_queue.py — run a list of training jobs one after another, gently and resumably.

    python research/experiments/tools/run_queue.py --queue research/experiments/e1a/queue.json \
        --max-hours 6

Each job is trained with research/experiments/e1_pilot/train_pilot.py (one process, one torch
thread, below-normal priority, exact-resume chunks). The queue repeats a job until its run
directory contains train_summary.json, then moves to the next one. A job that stops because the
machine is busy or short of memory is retried after --retry-wait-s seconds. If the machine is on
battery below --min-battery-pct, or after --max-hours, the queue stops and can simply be started
again later: finished runs are skipped and unfinished ones resume exactly.

Queue file (JSON): {"common": [...train_pilot args...], "jobs": [{"arm": "...", "seed": 0, "set": [...]}]}
Progress is written to <queue file directory>/queue_status.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "research" / "experiments" / "e1_pilot" / "train_pilot.py"


def run_dir(common, job):
    runs_root = common[common.index("--runs-root") + 1]
    return ROOT / runs_root / job["arm"] / "filtered" / f"seed_{job['seed']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--invocation-deadline-s", type=float, default=900)
    ap.add_argument("--retry-wait-s", type=float, default=180)
    ap.add_argument("--max-retries", type=int, default=20, help="consecutive non-progressing invocations per job")
    ap.add_argument("--min-battery-pct", type=float, default=35)
    args = ap.parse_args()

    qpath = Path(args.queue)
    queue = json.loads(qpath.read_text(encoding="utf-8"))
    common, jobs = queue["common"], queue["jobs"]
    status_path = qpath.parent / "queue_status.json"
    t0 = time.time()
    status = {"queue": str(qpath), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": []}

    def save(state):
        status["state"] = state
        status["elapsed_h"] = round((time.time() - t0) / 3600, 3)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    for job in jobs:
        rd = run_dir(common, job)
        entry = {"arm": job["arm"], "seed": job["seed"], "invocations": 0, "done": False}
        status["jobs"].append(entry)
        retries = 0
        while not (rd / "train_summary.json").is_file():
            if time.time() - t0 > args.max_hours * 3600:
                save("stopped: --max-hours reached")
                return 0
            cmd = [sys.executable, str(DRIVER), "--arm", job["arm"], "--seed", str(job["seed"]),
                   "--deadline-s", str(args.invocation_deadline_s),
                   "--min-battery-pct", str(args.min_battery_pct), *common]
            for s in job.get("set", []):
                cmd += ["--set", s]
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            entry["invocations"] += 1
            last = (out.stdout.strip().splitlines() or [""])[-1]
            entry["last_result"] = last[:300]
            save("running")
            if "battery" in last:
                save("paused: on battery below --min-battery-pct")
                return 0
            progressed = '"finished": true' in last or '"stop_reason": "deadline"' in last
            if out.returncode != 0:
                entry["error"] = out.stderr[-2000:]
                save("stopped: driver error")
                return 1
            if not progressed:
                retries += 1
                if retries > args.max_retries:
                    save("stopped: too many non-progressing invocations")
                    return 1
                time.sleep(args.retry_wait_s)
            else:
                retries = 0
        entry["done"] = True
        save("running")
    save("finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
