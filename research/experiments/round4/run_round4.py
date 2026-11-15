"""
run_round4.py — evaluation runner for round 4 (PROTOCOL.md here). The round-3 runner is unchanged.

    python research/experiments/round4/run_round4.py --tag latency --methods C:reference C:fast A:reference A:fast \
        --agent filtered=results/checkpoints/filtered --agent unfiltered=results/checkpoints/unfiltered \
        --stages expert --seed-base 4200000 --n 100 --set env.goal_sampling=rejection --record-calls

Methods (a ":reference" / ":fast" suffix selects the planner; default --planner):
  nofilter        the policy without the filter (reference only; collisions possible)
  baseline        the policy with the filter
  A, B, C, C_euclid   RecoveryController modes with exact information (round 3)
  C_loc, C_map, C_sensed   C with imperfect information for the controller only (perception.py);
                   the filter always uses the true state and map
  planner_only, path_follower   policy-free full-map controllers (global_baselines.py), run once per
                   scenario (agent "map", train_seed -1)
Scenario options: --task KEY=VALUE (stage overrides, e.g. dense obstacles), --heading-perturb (initial
heading replaced by a uniformly random one, seeded by the scenario; the same for every method).
Outputs (resumable; rows keyed by method, agent, seed, stage, scenario): results/<tag>/episodes.csv,
calls.csv (every planner call, with --record-calls), step_hist.csv (per-episode histogram of
controller-step latency), run_record.json.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "experiments" / "lightweight"))
sys.path.insert(0, str(ROOT / "research" / "experiments" / "round2"))
from run_eval import TAIL, classify  # noqa: E402
from run_eval2 import find_seed_ckpts  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "results"
AGENT_FREE = ("planner_only", "path_follower")
BINS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf")]   # ms
FIELDS = ["method", "planner", "info", "agent", "train_seed", "stage", "episode_seed", "heading_offset",
          "start_goal_dist_m", "outcome", "class", "steps", "intervention_steps", "min_obstacle_clearance",
          "tail_net_displacement_m", "start_geo_cost", "end_geo_cost", "path_length_m", "mean_exec_speed",
          "activations", "activations_success", "repeat_activations", "recovery_steps", "tail_recovery_share",
          "first_activation_step", "mean_activation_steps", "exit_reasons", "map_rebuilds", "known_obstacles",
          "planner_calls", "call_wall_ms_p50", "call_wall_ms_p99", "call_wall_ms_max", "call_cpu_ms_max",
          "act_wall_ms_p50", "act_wall_ms_p99", "act_wall_ms_max", "act_cpu_ms_max", "act_steps_over_100ms",
          "field_setup_s", "episode_wall_s"]


def parse_method(m, default_planner):
    name, _, pl = m.partition(":")
    return name, (pl or default_planner)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--planner", default="fast", choices=["reference", "fast"])
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--stages", nargs="+", default=["expert"])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--task", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--heading-perturb", action="store_true")
    ap.add_argument("--record-calls", action="store_true")
    ap.add_argument("--priority", default="below_normal", choices=["below_normal", "normal"])
    ap.add_argument("--deadline-s", type=float, default=1200.0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    import psutil
    proc = psutil.Process()
    if sys.platform == "win32":
        proc.nice(psutil.NORMAL_PRIORITY_CLASS if args.priority == "normal" else psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config, parse_override
    from safenav.factory import make_env
    from safenav.recovery import GeodesicField, RecoveryConfig, RecoveryController
    from safenav.recovery.global_baselines import PathFollower, PlannerOnly
    from safenav.recovery.perception import INFO
    from safenav.recovery.timing import calibrate, thread_cpu_s

    clock = calibrate()
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides=dict(parse_override(s) for s in args.set))
    task_over = dict(parse_override(s) for s in args.task)
    agents = []
    for spec in args.agent:
        name, d = spec.split("=", 1)
        found = find_seed_ckpts((ROOT / d).resolve(), "best")
        if not found:
            raise SystemExit(f"no checkpoints for {name} in {d}")
        agents += [(name, s, c) for s, c in found]
    methods = [parse_method(m, args.planner) for m in args.methods]

    out = (Path(args.out_root).resolve() if args.out_root else OUT_ROOT) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ep_path, call_path, hist_path = out / "episodes.csv", out / "calls.csv", out / "step_hist.csv"
    done = set()
    if ep_path.is_file():
        with open(ep_path, newline="", encoding="utf-8") as fh:
            done = {(r["method"], r["planner"], r["agent"], r["train_seed"], r["stage"], r["episode_seed"]) for r in csv.DictReader(fh)}
    # drop call/histogram rows of episodes that were interrupted before their episode row was written
    for path in (call_path, hist_path):
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            keep = [r for r in rows if (r["method"], r["planner"], r["agent"], r["train_seed"], r["stage"], r["episode_seed"]) in done]
            if len(keep) != len(rows):
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(keep)

    def opener(path, fields):
        new = not path.is_file()
        fh = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        return fh, w

    key_fields = ["method", "planner", "agent", "train_seed", "stage", "episode_seed"]
    fe, we = opener(ep_path, FIELDS)
    fc, wc = opener(call_path, key_fields + ["call", "wall_ms", "cpu_ms"]) if args.record_calls else (None, None)
    hist_fields = key_fields + [f"b{i}" for i in range(len(BINS) - 1)]
    fhh, wh = opener(hist_path, hist_fields)

    t0, cpu0 = time.time(), proc.cpu_times()
    psutil.cpu_percent(None)
    seeds = [args.seed_base + i for i in range(args.n)]
    completed, stopped = len(done), False
    # order: per agent, scenario by scenario, every method on the same scenario in turn, so that
    # methods (and the two planner implementations) alternate under the same background load
    blocks = []
    for stage in args.stages:
        for name, tseed, ckpt in agents:
            ms = [(m, pl) for m, pl in methods if m not in AGENT_FREE]
            if ms:
                blocks.append((stage, name, tseed, ckpt, ms))
        ms = [(m, pl) for m, pl in methods if m in AGENT_FREE]
        if ms:
            blocks.append((stage, "map", -1, None, ms))
    envs = {}
    jobs = []
    for stage, name, tseed, ckpt, ms in blocks:
        policy = PPOPolicy.from_checkpoint(ckpt, deterministic=True) if ckpt else None
        for ep_seed in seeds:
            for method, pl in ms:
                jobs.append((stage, name, tseed, policy, method, pl, ep_seed))
    for stage, name, tseed, policy, method, pl, ep_seed in jobs:
        if stopped:
            break
        filt = method != "nofilter"
        if (stage, filt) not in envs:
            envs[(stage, filt)] = make_env(config, task=dict(get_stage(config, stage), **task_over), safety_filter=filt)
        env = envs[(stage, filt)]
        if True:
            k = (method, pl, name, str(tseed), stage, str(ep_seed))
            if k in done:
                continue
            if time.time() - t0 > args.deadline_s:
                stopped = True
                break
            te = time.perf_counter()
            obs, info = env.reset(seed=ep_seed)
            base = env.unwrapped
            offset = 0.0
            if args.heading_perturb:
                rng = np.random.default_rng([ep_seed, 7])
                new_th = float(rng.uniform(-np.pi, np.pi))
                offset = float((new_th - base.robot_theta + np.pi) % (2 * np.pi) - np.pi)
                base.robot_theta = new_th
                base.headings = [new_th] if hasattr(base, "headings") else None
                obs = base._get_obs()
            if method in ("nofilter", "baseline"):
                ctl, info_name = None, "-"
                policy.reset(seed=ep_seed)
            elif method in AGENT_FREE:
                ctl = PlannerOnly(planner=pl) if method == "planner_only" else PathFollower()
                ctl.bind(env, ep_seed)
                info_name = "exact"
            else:
                mode = {"A": "A", "B": "B", "C": "C", "C_euclid": "C", "C_loc": "C", "C_map": "C", "C_sensed": "C"}[method]
                cost = "euclidean" if method == "C_euclid" else "geodesic"
                info_name = {"C_loc": "localisation_error", "C_map": "map_error", "C_sensed": "sensed_map"}.get(method, "exact")
                ctl = RecoveryController(policy, RecoveryConfig(mode=mode, cost=cost, planner=pl), info=INFO[info_name]())
                ctl.reset(seed=ep_seed)
                ctl.bind(env, ep_seed)
            geo = getattr(ctl, "geo", None) or GeodesicField(base.obstacle_map.obstacles, base.goal_pos)
            start_goal = float(np.linalg.norm(base.goal_pos - base.robot_pos))
            start_cost = float(geo(base.robot_pos))
            v_nom, interv, clear, rec, act_w, act_c, vexec = [], [], [info["obstacle_clearance"]], [], [], [], []
            term = trunc = False
            while not (term or trunc):
                w0, c0 = time.perf_counter(), thread_cpu_s()
                a = np.asarray((ctl or policy).act(obs), dtype=float)
                act_w.append(1000 * (time.perf_counter() - w0))
                act_c.append(1000 * (thread_cpu_s() - c0))
                rec.append(bool(getattr(ctl, "in_recovery", False)))
                obs, _, term, trunc, info = env.step(a)
                v_nom.append(float(np.clip(a[0], 0.0, 1.0)))
                interv.append(bool(info.get("safety_intervened", False)))
                clear.append(info["obstacle_clearance"])
                vexec.append(float(info["executed_action"][0]))
            if hasattr(ctl, "notify_goal"):
                ctl.notify_goal(int(info["step"]))
            outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
            cls, disp = classify(np.asarray(base.trajectory), v_nom, interv) if outcome == "timeout" else ("", "")
            st = getattr(ctl, "stats", None)
            acts = st.activations if st else []
            starts = [np.array(x.start_pos) for x in acts]
            repeats = sum(any(np.hypot(*(starts[i] - starts[j])) < 0.5 for j in range(i)) for i in range(len(starts)))
            calls_w = (st.call_wall_ms if st else getattr(ctl, "call_wall_ms", [])) if ctl else []
            calls_c = (st.call_cpu_ms if st else getattr(ctl, "call_cpu_ms", [])) if ctl else []
            tr = np.asarray(base.trajectory)
            row = {"method": method, "planner": pl, "info": info_name, "agent": name, "train_seed": tseed, "stage": stage,
                   "episode_seed": ep_seed, "heading_offset": round(offset, 6), "start_goal_dist_m": round(start_goal, 4),
                   "outcome": outcome, "class": cls, "steps": int(info["step"]), "intervention_steps": int(np.sum(interv)),
                   "min_obstacle_clearance": float(np.min(clear)), "tail_net_displacement_m": disp,
                   "start_geo_cost": start_cost, "end_geo_cost": float(geo(base.robot_pos)),
                   "path_length_m": float(np.sum(np.hypot(*np.diff(tr, axis=0).T))) if len(tr) > 1 else 0.0,
                   "mean_exec_speed": float(np.mean(vexec)) if vexec else 0.0,
                   "activations": len(acts), "activations_success": sum(x.success for x in acts),
                   "repeat_activations": int(repeats), "recovery_steps": st.recovery_steps if st else 0,
                   "tail_recovery_share": float(np.mean(rec[-TAIL:])) if rec else 0.0,
                   "first_activation_step": acts[0].start_step if acts else "",
                   "mean_activation_steps": float(np.mean([x.end_step - x.start_step for x in acts])) if acts else "",
                   "exit_reasons": ";".join(x.reason for x in acts),
                   "map_rebuilds": getattr(ctl, "map_rebuilds", 0) if ctl else 0,
                   "known_obstacles": int(ctl.info.known.sum()) if ctl is not None and hasattr(getattr(ctl, "info", None), "known") else "",
                   "planner_calls": len(calls_w),
                   "call_wall_ms_p50": float(np.percentile(calls_w, 50)) if calls_w else "",
                   "call_wall_ms_p99": float(np.percentile(calls_w, 99)) if calls_w else "",
                   "call_wall_ms_max": float(np.max(calls_w)) if calls_w else "",
                   "call_cpu_ms_max": float(np.max(calls_c)) if calls_c else "",
                   "act_wall_ms_p50": float(np.percentile(act_w, 50)), "act_wall_ms_p99": float(np.percentile(act_w, 99)),
                   "act_wall_ms_max": float(np.max(act_w)), "act_cpu_ms_max": float(np.max(act_c)),
                   "act_steps_over_100ms": int(np.sum(np.asarray(act_w) > 100.0)),
                   "field_setup_s": round(getattr(ctl, "field_setup_s", 0.0), 5) if ctl else 0.0,
                   "episode_wall_s": round(time.perf_counter() - te, 4)}
            kd = {"method": method, "planner": pl, "agent": name, "train_seed": tseed, "stage": stage, "episode_seed": ep_seed}
            if wc is not None:
                for i, (w, c) in enumerate(zip(calls_w, calls_c)):
                    wc.writerow({**kd, "call": i, "wall_ms": round(w, 4), "cpu_ms": round(c, 4)})
                fc.flush()
            counts, _ = np.histogram(act_w, bins=BINS)
            wh.writerow({**kd, **{f"b{i}": int(c) for i, c in enumerate(counts)}})
            fhh.flush()
            we.writerow(row)       # written last: an episode counts as done only when all its rows exist
            fe.flush()
            completed += 1
    for f in (fe, fc, fhh):
        if f is not None:
            f.close()
    cpu1, mem = proc.cpu_times(), proc.memory_info()
    bat = psutil.sensors_battery()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "safenav", "research/experiments/round4/run_round4.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"tag": args.tag, "command": " ".join(sys.argv), "git_commit": commit, "code_modified_since_commit": bool(dirty),
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "on_mains_power": bool(bat.power_plugged) if bat else None,
              "priority": args.priority, "system_cpu_percent_mean": psutil.cpu_percent(None), "cpu_clock": clock,
              "episodes_completed": completed, "stopped_at_deadline": stopped,
              "wall_s_this_invocation": round(time.time() - t0, 1),
              "cpu_s_this_invocation": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 1),
              "peak_memory_mb": round((getattr(mem, "peak_wset", None) or mem.rss) / 2**20, 1),
              "checkpoints": {f"{n}_seed{s}": {"path": str(c.relative_to(ROOT)).replace("\\", "/"),
                                               "sha256": hashlib.sha256(c.read_bytes()).hexdigest()} for n, s, c in agents}}
    prev = out / "run_record.json"
    history = json.loads(prev.read_text(encoding="utf-8")).get("invocations", []) if prev.is_file() else []
    prev.write_text(json.dumps({"invocations": history + [record]}, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("episodes_completed", "stopped_at_deadline", "wall_s_this_invocation",
                                              "on_mains_power", "system_cpu_percent_mean")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
