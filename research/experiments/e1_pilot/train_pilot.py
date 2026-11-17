"""
train_pilot.py — one E1 pilot training run, in bounded chunks, gentle on the machine.

    python research/experiments/e1_pilot/train_pilot.py --arm penalty --seed 0 --timesteps 150000 \
        --deadline-s 1500

Same training code as `python -m safenav train --seed <s> --timesteps <n> --threads 1` (filter on,
default config). The penalty arm adds only `safety.correction_penalty=<alpha>`; the control arm
leaves the default (0). The run trains in chunks of --chunk-updates PPO updates. Each chunk ends
with an exact resume checkpoint (tests/test_ppo.py checks that resumed training is identical to
uninterrupted training), so the run can stop between chunks and be continued by calling the
script again. Between chunks it stops if the deadline is reached, its own peak memory exceeds
--max-rss-mb, the free system memory falls below --min-free-mb, or a chunk ran slower than
--min-steps-per-s (a sign that the machine is busy or the process is being paged out).

Run directory: runs/e1_pilot/<arm>/filtered/seed_<s> (train-many layout; git-ignored).
Each invocation appends to research/experiments/e1_pilot/results/train_<arm>_seed<s>.json.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT_ROOT = Path(__file__).resolve().parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["control", "penalty"])
    ap.add_argument("--alpha", type=float, default=1.0, help="penalty weight of the penalty arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--runs-root", default="runs/e1_pilot")
    ap.add_argument("--chunk-updates", type=int, default=2, help="PPO updates (2,048 steps each) per chunk")
    ap.add_argument("--deadline-s", type=float, default=1500)
    ap.add_argument("--max-rss-mb", type=float, default=1200)
    ap.add_argument("--min-free-mb", type=float, default=1000)
    ap.add_argument("--min-steps-per-s", type=float, default=60,
                    help="stop after a chunk slower than this (validation chunks included)")
    ap.add_argument("--tag", default=None, help="record name (default train_<arm>_seed<s>)")
    args = ap.parse_args()

    import psutil
    import torch
    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if platform.system() == "Windows" else 10)
    torch.set_num_threads(1)

    from safenav.config import load_config
    from safenav.training.ppo_trainer import PPOTrainer

    overrides = {"safety.correction_penalty": args.alpha} if args.arm == "penalty" else {}
    config = load_config(None, overrides=overrides)
    config["training"]["seed"] = args.seed
    config["ppo"]["total_timesteps"] = args.timesteps
    run_dir = ROOT / args.runs_root / args.arm / "filtered" / f"seed_{args.seed}"
    if (run_dir / "train_summary.json").is_file():
        print(f"{run_dir} already finished")
        return 0

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "safenav", "configs"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{args.tag or f'train_{args.arm}_seed{args.seed}'}.json"
    history = json.loads(path.read_text(encoding="utf-8")).get("invocations", []) if path.is_file() else []
    t0, cpu0 = time.time(), proc.cpu_times()
    chunks, stop_reason, finished = [], "deadline", False

    def save_record():
        # Rewritten after every chunk, so a process stopped from outside keeps its record.
        cpu1 = proc.cpu_times()
        record = {
            "arm": args.arm, "alpha": args.alpha if args.arm == "penalty" else 0.0, "seed": args.seed,
            "timesteps_planned": args.timesteps, "command": " ".join(sys.argv), "git_commit": commit,
            "code_modified_since_commit": dirty, "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(),
            "on_mains_power": getattr(psutil.sensors_battery(), "power_plugged", None),
            "finished": finished, "stop_reason": stop_reason,
            "wall_s_this_invocation": round(time.time() - t0, 1),
            "cpu_s_this_invocation": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 1),
            "priority": "below normal", "torch_threads": 1, "chunks": chunks,
        }
        path.write_text(json.dumps({"invocations": history + [record]}, indent=2), encoding="utf-8")
        return record

    if psutil.virtual_memory().available / 2**20 < args.min_free_mb:
        stop_reason = "not started: free system memory below --min-free-mb"
    while stop_reason == "deadline" and time.time() - t0 < args.deadline_s:
        tc = time.time()
        trainer = PPOTrainer(config, run_dir=str(run_dir), safety_filter=True, verbose=False, resume=True)
        start = int(trainer._resume_state["loop"]["update"]) if trainer._resume_state else 0
        steps0 = trainer.logger.total_timesteps
        result = trainer.train(stop_after_updates=start + args.chunk_updates)
        mem = proc.memory_info()
        peak_mb = (getattr(mem, "peak_wset", None) or mem.rss) / 2**20
        free_mb = psutil.virtual_memory().available / 2**20
        wall = time.time() - tc
        sps = (trainer.logger.total_timesteps - steps0) / max(wall, 1e-9)
        chunks.append({"from_update": start, "timesteps": trainer.logger.total_timesteps,
                       "wall_s": round(wall, 1), "steps_per_s": round(sps), "peak_memory_mb": round(peak_mb, 1),
                       "free_system_mb": round(free_mb), "cpu_percent_system": psutil.cpu_percent(interval=None)})
        print(json.dumps(chunks[-1]), flush=True)
        del trainer
        if "interrupted_after_update" not in result:
            finished, stop_reason = True, "finished"
        elif peak_mb > args.max_rss_mb:
            stop_reason = "memory limit (process)"
        elif free_mb < args.min_free_mb:
            stop_reason = "memory limit (system)"
        elif sps < args.min_steps_per_s:
            stop_reason = "throughput below --min-steps-per-s (machine busy or paging)"
        save_record()

    record = save_record()
    print(json.dumps({k: record[k] for k in ("finished", "stop_reason", "wall_s_this_invocation",
                                              "cpu_s_this_invocation")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
