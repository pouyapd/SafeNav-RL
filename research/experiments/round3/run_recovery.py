"""
run_recovery.py — evaluation runner for round 3 (deadlock recovery; PROTOCOL.md here).

    python research/experiments/round3/run_recovery.py --tag pilot --methods baseline A B C C_euclid \
        --agent filtered=results/checkpoints/filtered --agent unfiltered=results/checkpoints/unfiltered \
        --stages expert --seed-base 8300000 --n 40 --set env.goal_sampling=rejection

Methods: baseline (policy + filter), A, B, C (RecoveryController modes, geodesic cost) and
C_euclid / A_euclid (Euclidean cost). Every method uses the same environment, filter and
scenarios; the filter checks every action. One process, one torch thread, below-normal
priority, deadline, resumable (rows keyed by method, agent, seed, stage, scenario).
Stall classes use the round-2 rules on the action proposed to the filter. Extra columns
record the recovery activity and the time spent per episode.
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
sys.path.insert(0, str(ROOT / "research" / "experiments" / "round2"))
from run_eval import TAIL, classify, lower_priority  # noqa: E402
from run_eval2 import find_seed_ckpts  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "results"
METHODS = {"baseline": None, "A": ("A", "geodesic"), "B": ("B", "geodesic"), "C": ("C", "geodesic"),
           "C_euclid": ("C", "euclidean"), "A_euclid": ("A", "euclidean")}
FIELDS = ["method", "agent", "train_seed", "stage", "episode_seed", "start_goal_dist_m", "outcome", "class", "steps",
          "intervention_steps", "min_obstacle_clearance", "tail_net_displacement_m", "start_geo_cost", "end_geo_cost",
          "activations", "activations_success", "repeat_activations", "recovery_steps", "tail_recovery_share",
          "first_activation_step", "mean_activation_steps", "exit_reasons", "planner_calls", "planner_time_s",
          "controller_time_s", "field_setup_s", "episode_wall_s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", nargs="+", default=["baseline", "A", "B", "C", "C_euclid"], choices=list(METHODS))
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--checkpoint-name", default="best", choices=["best", "final"])
    ap.add_argument("--stages", nargs="+", default=["expert"])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--deadline-s", type=float, default=1200.0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    proc = lower_priority()
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config, parse_override
    from safenav.factory import make_env
    from safenav.recovery import GeodesicField, RecoveryConfig, RecoveryController

    config = load_config(ROOT / "configs" / "default_config.yaml", overrides=dict(parse_override(s) for s in args.set))
    agents = []
    for spec in args.agent:
        name, d = spec.split("=", 1)
        found = find_seed_ckpts((ROOT / d).resolve(), args.checkpoint_name)
        if not found:
            raise SystemExit(f"no checkpoints for {name} in {d}")
        agents += [(name, s, c) for s, c in found]

    out = (Path(args.out_root).resolve() if args.out_root else OUT_ROOT) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "episodes.csv"
    done = set()
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            done = {(r["method"], r["agent"], r["train_seed"], r["stage"], r["episode_seed"]) for r in csv.DictReader(fh)}
    new_file = not csv_path.is_file()
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0, cpu0 = time.time(), proc.cpu_times()
    seeds = [args.seed_base + i for i in range(args.n)]
    planned = len(agents) * len(args.stages) * len(seeds) * len(args.methods)
    completed, stopped = len(done), False
    for stage in args.stages:
        env = make_env(config, task=get_stage(config, stage), safety_filter=True)
        for name, tseed, ckpt in agents:
            policy = PPOPolicy.from_checkpoint(ckpt, deterministic=True)
            for method in args.methods:
                spec = METHODS[method]
                ctl = RecoveryController(policy, RecoveryConfig(mode=spec[0], cost=spec[1])) if spec else None
                for ep_seed in seeds:
                    if (method, name, str(tseed), stage, str(ep_seed)) in done:
                        continue
                    if time.time() - t0 > args.deadline_s:
                        stopped = True
                        break
                    te = time.perf_counter()
                    obs, info = env.reset(seed=ep_seed)
                    base = env.unwrapped
                    if ctl:
                        ctl.reset(seed=ep_seed)
                        ctl.bind(env)
                        geo = ctl.geo
                    else:
                        policy.reset(seed=ep_seed)
                        geo = GeodesicField(base.obstacle_map.obstacles, base.goal_pos)
                    start_goal = float(np.linalg.norm(base.goal_pos - base.robot_pos))
                    start_cost = float(geo(base.robot_pos))
                    v_nom, interv, clear, rec = [], [], [info["obstacle_clearance"]], []
                    term = trunc = False
                    while not (term or trunc):
                        a = np.asarray((ctl or policy).act(obs), dtype=float)
                        rec.append(bool(ctl.in_recovery) if ctl else False)
                        obs, _, term, trunc, info = env.step(a)
                        v_nom.append(float(np.clip(a[0], 0.0, 1.0)))
                        interv.append(bool(info.get("safety_intervened", False)))
                        clear.append(info["obstacle_clearance"])
                    if ctl:
                        ctl.notify_goal(int(info["step"]))
                    outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
                    cls, disp = classify(np.asarray(base.trajectory), v_nom, interv) if outcome == "timeout" else ("", "")
                    acts = ctl.stats.activations if ctl else []
                    starts = [np.array(x.start_pos) for x in acts]
                    repeats = sum(any(np.hypot(*(starts[i] - starts[j])) < 0.5 for j in range(i)) for i in range(len(starts)))
                    row = {"method": method, "agent": name, "train_seed": tseed, "stage": stage, "episode_seed": ep_seed,
                           "start_goal_dist_m": round(start_goal, 4), "outcome": outcome, "class": cls,
                           "steps": int(info["step"]), "intervention_steps": int(np.sum(interv)),
                           "min_obstacle_clearance": float(np.min(clear)), "tail_net_displacement_m": disp,
                           "start_geo_cost": start_cost, "end_geo_cost": float(geo(base.robot_pos)),
                           "activations": len(acts), "activations_success": sum(x.success for x in acts),
                           "repeat_activations": int(repeats),
                           "recovery_steps": ctl.stats.recovery_steps if ctl else 0,
                           "tail_recovery_share": float(np.mean(rec[-TAIL:])) if rec else 0.0,
                           "first_activation_step": acts[0].start_step if acts else "",
                           "mean_activation_steps": float(np.mean([x.end_step - x.start_step for x in acts])) if acts else "",
                           "exit_reasons": ";".join(x.reason for x in acts),
                           "planner_calls": ctl.stats.planner_calls if ctl else 0,
                           "planner_time_s": round(ctl.stats.planner_time_s, 5) if ctl else 0.0,
                           "controller_time_s": round(ctl.stats.controller_time_s, 5) if ctl else 0.0,
                           "field_setup_s": round(ctl.field_setup_s, 5) if ctl else 0.0,
                           "episode_wall_s": round(time.perf_counter() - te, 4)}
                    writer.writerow(row)
                    fh.flush()
                    completed += 1
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break
    fh.close()
    cpu1, mem = proc.cpu_times(), proc.memory_info()
    import psutil
    bat = psutil.sensors_battery()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "safenav", "research/experiments/round3/run_recovery.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"tag": args.tag, "command": " ".join(sys.argv), "git_commit": commit, "code_modified_since_commit": bool(dirty),
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "on_mains_power": bool(bat.power_plugged) if bat else None,
              "episodes_planned": planned, "episodes_completed": completed, "stopped_at_deadline": stopped,
              "wall_s_this_invocation": round(time.time() - t0, 1),
              "cpu_s_this_invocation": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 1),
              "peak_memory_mb": round((getattr(mem, "peak_wset", None) or mem.rss) / 2**20, 1),
              "checkpoints": {f"{n}_seed{s}": {"path": str(c.relative_to(ROOT)).replace("\\", "/"),
                                               "sha256": hashlib.sha256(c.read_bytes()).hexdigest()} for n, s, c in agents}}
    prev = out / "run_record.json"
    history = json.loads(prev.read_text(encoding="utf-8")).get("invocations", []) if prev.is_file() else []
    prev.write_text(json.dumps({"invocations": history + [record]}, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("episodes_completed", "episodes_planned", "stopped_at_deadline",
                                              "wall_s_this_invocation", "on_mains_power")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
