"""
run_eval2.py — evaluation-only runner for round-2 experiments (E3, E4, DWA; README.md here).

    python research/experiments/round2/run_eval2.py --tag e3_corrected \
        --agent unfiltered=results/checkpoints/unfiltered --agent filtered=results/checkpoints/filtered \
        --baseline potential_field --baseline dwa --stages beginner intermediate expert \
        --seed-base 7000000 --n 500 --set env.goal_sampling=rejection --deadline-s 900

--agent NAME=DIR    agents in DIR/seed_<k>.pt (committed layout) or DIR/seed_<k>/checkpoints/<ckpt>.pt
--baseline KIND     gotogoal | potential_field | dwa (with --dwa KEY=VALUE overrides)
One process, one torch thread, below-normal priority, a deadline, one CSV row per episode
(resumable: rows already present are skipped). Timeout classes are those of
research/experiments/lightweight/run_eval.py (last 100 steps, 0.10 m, 50 % rules).
Extra diagnostics for E4: the policy's raw (unclipped) speed command over the last 100 steps,
the risk score and the barrier value at the end, and the start-goal distance.
Outputs: research/experiments/round2/results/<tag>/episodes.csv and run_record.json.
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
sys.path.insert(0, str(ROOT / "research" / "experiments" / "lightweight"))
from run_eval import TAIL, classify, lower_priority  # noqa: E402  (same stall rules as the lightweight runner)

OUT_ROOT = Path(__file__).resolve().parent / "results"
FIELDS = ["agent", "kind", "train_seed", "checkpoint", "actions", "filter", "stage", "episode_seed",
          "start_goal_dist_m", "outcome", "class", "steps", "intervention_steps", "min_obstacle_clearance",
          "tail_net_displacement_m", "end_obstacle_barrier_m", "end_risk_score", "tail_mean_raw_v",
          "tail_share_raw_v_nonpositive", "tail_mean_exec_v", "goal_sampling_fallback"]


def find_seed_ckpts(d: Path, ckpt_name: str):
    out = []
    for s in range(10):
        for c in (d / f"seed_{s}.pt", d / f"seed_{s}" / "checkpoints" / f"{ckpt_name}.pt"):
            if c.is_file():
                out.append((s, c))
                break
    return out


def parse_kv(items):
    from safenav.config import parse_override
    return dict(parse_override(s) for s in items)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--checkpoint-name", default="best", choices=["best", "final"])
    ap.add_argument("--baseline", action="append", default=[], choices=["gotogoal", "potential_field", "dwa"])
    ap.add_argument("--dwa", action="append", default=[], metavar="KEY=VALUE", help="DWA parameter override")
    ap.add_argument("--stages", nargs="+", default=["expert"])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, help="episodes per agent and stage")
    ap.add_argument("--filter", default="on", choices=["on", "off"])
    ap.add_argument("--sampled", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--task", action="append", default=[], metavar="KEY=VALUE",
                    help="override a stage parameter, e.g. min_obstacles=15 (E3-dense)")
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    proc = lower_priority()
    import torch
    torch.set_num_threads(1)
    from safenav.agents.baselines import DynamicWindowPolicy, GoToGoalPolicy, PotentialFieldPolicy
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config
    from safenav.factory import make_env
    from safenav.safety.cbf_filter import CBFSafetyFilter

    config = load_config(ROOT / "configs" / "default_config.yaml", overrides=parse_kv(args.set))
    agents = []  # (name, kind, train_seed, ckpt or None)
    for spec in args.agent:
        name, d = spec.split("=", 1)
        found = find_seed_ckpts((ROOT / d).resolve(), args.checkpoint_name)
        if not found:
            raise SystemExit(f"no checkpoints for {name} in {d}")
        agents += [(name, "ppo", s, c) for s, c in found]
    agents += [(b, b, -1, None) for b in args.baseline]

    out = (Path(args.out_root).resolve() if args.out_root else OUT_ROOT) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    done = set()
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            done = {(r["agent"], r["train_seed"], r["stage"], r["episode_seed"]) for r in csv.DictReader(fh)}
    new_file = not csv_path.is_file()
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0, cpu0 = time.time(), proc.cpu_times()
    seeds = [args.seed_base + i for i in range(args.n)]
    planned, completed, stopped = len(agents) * len(args.stages) * len(seeds), len(done), False
    probe = CBFSafetyFilter.from_config(config)
    for stage in args.stages:
        task = dict(get_stage(config, stage), **parse_kv(args.task))
        env = make_env(config, task=task, safety_filter=(args.filter == "on"))
        for name, kind, tseed, ckpt in agents:
            if kind == "ppo":
                policy = PPOPolicy.from_checkpoint(ckpt, deterministic=not args.sampled)
            elif kind == "dwa":
                policy = DynamicWindowPolicy.from_config(config, **parse_kv(args.dwa))
            elif kind == "potential_field":
                policy = PotentialFieldPolicy.from_config(config)
            else:
                policy = GoToGoalPolicy()
            for ep_seed in seeds:
                if (name, str(tseed), stage, str(ep_seed)) in done:
                    continue
                if time.time() - t0 > args.deadline_s:
                    stopped = True
                    break
                obs, info = env.reset(seed=ep_seed)
                policy.reset(seed=ep_seed)
                base = env.unwrapped
                start_goal = float(np.linalg.norm(base.goal_pos - base.robot_pos))
                v_nom, raw_v, exec_v, interv, clear = [], [], [], [], [info["obstacle_clearance"]]
                term = trunc = False
                while not (term or trunc):
                    action = np.asarray(policy.act(obs), dtype=float)
                    obs, _, term, trunc, info = env.step(action)
                    raw_v.append(float(action[0]))
                    v_nom.append(float(np.clip(action[0], 0.0, 1.0)))
                    exec_v.append(float(info["executed_action"][0]))
                    interv.append(bool(info.get("safety_intervened", False)))
                    clear.append(info["obstacle_clearance"])
                outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
                h, _ = probe.barriers(base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
                hn = h[:len(base.obstacle_map.obstacles)]
                row = {"agent": name, "kind": kind, "train_seed": tseed,
                       "checkpoint": args.checkpoint_name if ckpt else "", "filter": args.filter,
                       "actions": "sampled" if args.sampled else "mean", "stage": stage, "episode_seed": ep_seed,
                       "start_goal_dist_m": round(start_goal, 4), "outcome": outcome, "class": "",
                       "steps": int(info["step"]), "intervention_steps": int(np.sum(interv)),
                       "min_obstacle_clearance": float(np.min(clear)), "tail_net_displacement_m": "",
                       "end_obstacle_barrier_m": float(hn.min()) if len(hn) else "",
                       "end_risk_score": float(base.risk_score()),
                       "tail_mean_raw_v": float(np.mean(raw_v[-TAIL:])),
                       "tail_share_raw_v_nonpositive": float(np.mean(np.asarray(raw_v[-TAIL:]) <= 0.0)),
                       "tail_mean_exec_v": float(np.mean(exec_v[-TAIL:])),
                       "goal_sampling_fallback": int(getattr(base, "goal_sampling_fallback", False))}
                if outcome == "timeout":
                    row["class"], row["tail_net_displacement_m"] = classify(np.asarray(base.trajectory), v_nom, interv)
                writer.writerow(row)
                fh.flush()
                completed += 1
            if stopped:
                break
        if stopped:
            break
    fh.close()
    cpu1 = proc.cpu_times()
    mem = proc.memory_info()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {
        "tag": args.tag, "command": " ".join(sys.argv), "git_commit": commit,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "episodes_planned": planned, "episodes_completed": completed,
        "stopped_at_deadline": stopped, "wall_s_this_invocation": round(time.time() - t0, 1),
        "cpu_s_this_invocation": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 1),
        "peak_memory_mb": round((getattr(mem, "peak_wset", None) or mem.rss) / 2**20, 1),
        "checkpoints": {f"{n}_seed{s}": {"path": str(c.relative_to(ROOT)).replace("\\", "/"),
                                         "sha256": hashlib.sha256(c.read_bytes()).hexdigest()}
                        for n, _, s, c in agents if c is not None},
    }
    prev = out / "run_record.json"
    history = json.loads(prev.read_text(encoding="utf-8")).get("invocations", []) if prev.is_file() else []
    prev.write_text(json.dumps({"invocations": history + [record]}, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("episodes_completed", "episodes_planned", "stopped_at_deadline",
                                              "wall_s_this_invocation", "peak_memory_mb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
