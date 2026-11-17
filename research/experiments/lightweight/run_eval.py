"""
run_eval.py — resource-limited, evaluation-only runner for the lightweight experiments
(research/experiments/lightweight/README.md). No training.

One process, one torch thread, below-normal CPU priority, a hard deadline, and one CSV row
written per episode (an interrupted run keeps its rows and resumes where it stopped).

    python research/experiments/lightweight/run_eval.py --tag fresh_best_k0 \
        --agents results/checkpoints --stage expert --seed-base 9000000 --n 100 --kappa 0 --deadline-s 600

--agents: a directory in the committed layout (<dir>/{filtered,unfiltered}/seed_<k>.pt), or
          runs/benchmark with --checkpoint-name best|final (<dir>/<variant>/seed_<k>/checkpoints/<name>.pt).
Outputs: research/experiments/lightweight/results/<tag>/episodes.csv and run_record.json.
Timeout classes follow research/scripts/audit_diagnostics.py (last 100 steps, 0.10 m, 50 % rules).
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT_ROOT = Path(__file__).resolve().parent / "results"
TAIL, STALL_DISP = 100, 0.10
FIELDS = ["agent", "variant", "train_seed", "checkpoint", "actions", "kappa", "stage", "episode_seed",
          "outcome", "class", "steps", "escape_steps", "intervention_steps", "min_obstacle_clearance",
          "tail_net_displacement_m", "end_obstacle_barrier_m"]


def lower_priority():
    import psutil
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if platform.system() == "Windows" else 10)
    return p


def find_agents(path: Path, checkpoint_name: str):
    agents = []
    for variant in ("filtered", "unfiltered"):
        for s in range(5):
            committed = path / variant / f"seed_{s}.pt"
            run = path / variant / f"seed_{s}" / "checkpoints" / f"{checkpoint_name}.pt"
            ckpt = committed if committed.is_file() else run
            if ckpt.is_file():
                agents.append((variant, s, ckpt))
    return agents


def classify(traj, v_nom, interv):
    disp = float(np.linalg.norm(traj[-1] - traj[-TAIL - 1]))
    if disp >= STALL_DISP:
        return "moving", disp
    if np.mean(interv[-TAIL:]) >= 0.5:
        return "filter-blocked", disp
    if np.mean(np.asarray(v_nom[-TAIL:]) <= 1e-3) >= 0.5:
        return "self-stop", disp
    return "other stall", disp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--agents", required=True)
    ap.add_argument("--checkpoint-name", default="best", choices=["best", "final"])
    ap.add_argument("--variants", nargs="+", default=["filtered", "unfiltered"])
    ap.add_argument("--stage", default="expert")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, help="episodes per agent")
    ap.add_argument("--kappa", type=float, default=0.0, help="heading escape (0 = plain filter)")
    ap.add_argument("--sampled", action="store_true", help="sample actions instead of the policy mean")
    ap.add_argument("--deadline-s", type=float, default=600.0)
    args = ap.parse_args()

    proc = lower_priority()
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config
    from safenav.safety.heading_escape import make_escape_env

    config = load_config(ROOT / "configs" / "default_config.yaml")
    task = get_stage(config, args.stage)
    agents = [a for a in find_agents(Path(args.agents).resolve(), args.checkpoint_name) if a[0] in args.variants]
    if not agents:
        raise SystemExit(f"no checkpoints found under {args.agents}")
    out = OUT_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    done = set()
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            done = {(r["agent"], int(r["episode_seed"])) for r in csv.DictReader(f)}
    new_file = not csv_path.is_file()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0, cpu0 = time.time(), proc.cpu_times()
    seeds = [args.seed_base + i for i in range(args.n)]
    planned, completed, stopped_early = len(agents) * len(seeds), len(done), False
    env = make_escape_env(config, task, args.kappa)
    for variant, train_seed, ckpt in agents:
        name = f"{variant}_seed{train_seed}"
        policy = PPOPolicy.from_checkpoint(ckpt, deterministic=not args.sampled)
        for ep_seed in seeds:
            if (name, ep_seed) in done:
                continue
            if time.time() - t0 > args.deadline_s:
                stopped_early = True
                break
            obs, info = env.reset(seed=ep_seed)
            policy.reset(seed=ep_seed)
            base = env.unwrapped
            v_nom, interv, esc, clear = [], [], 0, [info["obstacle_clearance"]]
            term = trunc = False
            while not (term or trunc):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                v_nom.append(float(info["nominal_action"][0]))
                interv.append(bool(info["safety_intervened"]))
                esc += int(info["heading_escape"])
                clear.append(info["obstacle_clearance"])
            outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
            row = {"agent": name, "variant": variant, "train_seed": train_seed, "checkpoint": args.checkpoint_name,
                   "actions": "sampled" if args.sampled else "mean", "kappa": args.kappa, "stage": args.stage,
                   "episode_seed": ep_seed, "outcome": outcome, "class": "", "steps": int(info["step"]),
                   "escape_steps": esc, "intervention_steps": int(np.sum(interv)),
                   "min_obstacle_clearance": float(np.min(clear)),
                   "tail_net_displacement_m": "", "end_obstacle_barrier_m": ""}
            if outcome == "timeout":
                row["class"], row["tail_net_displacement_m"] = classify(np.asarray(base.trajectory), v_nom, interv)
                h, _ = env.safety_filter.barriers(base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
                row["end_obstacle_barrier_m"] = float(h[:len(base.obstacle_map.obstacles)].min())
            writer.writerow(row)
            f.flush()
            completed += 1
        if stopped_early:
            break
    f.close()
    cpu1 = proc.cpu_times()
    mem = proc.memory_info()
    peak = getattr(mem, "peak_wset", None) or mem.rss
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {
        "tag": args.tag, "command": " ".join(sys.argv), "git_commit": commit,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "episodes_planned": planned, "episodes_completed": completed,
        "stopped_at_deadline": stopped_early, "wall_s_this_invocation": round(time.time() - t0, 1),
        "cpu_s_this_invocation": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 1),
        "peak_memory_mb": round(peak / 2**20, 1), "priority": "below normal", "torch_threads": 1,
        "checkpoints": {f"{v}_seed{s}": {"path": str(c.relative_to(ROOT)).replace("\\", "/"),
                                         "sha256": hashlib.sha256(c.read_bytes()).hexdigest()}
                        for v, s, c in agents},
    }
    prev = out / "run_record.json"
    history = json.loads(prev.read_text(encoding="utf-8")).get("invocations", []) if prev.is_file() else []
    (out / "run_record.json").write_text(json.dumps({"invocations": history + [record]}, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("episodes_completed", "episodes_planned", "stopped_at_deadline",
                                              "wall_s_this_invocation", "cpu_s_this_invocation", "peak_memory_mb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
