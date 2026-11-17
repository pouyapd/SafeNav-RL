"""
profile_planner.py — Phase 2 of round 4: where does the recovery planner spend its time?

    python research/experiments/round4/profile_planner.py [--n-scen 20] [--calls 4000]

Reproducible profiling on a dedicated scenario set (expert, corrected generator, seeds 4,100,000+i;
not used anywhere else). Steps:
  1. Episodes: the 10 committed agents x n_scen scenarios x modes A and C, once with the reference
     planner (round 3) and once with the fast planner (round 4). Every planner call's input is
     recorded (reference run); every controller step is timed (wall clock and thread CPU time).
     The two runs must give identical episodes (checked).
  2. Isolated calls: a seeded sample of the recorded planner inputs is re-timed with both planners,
     in alternating order, 3 repetitions each (wall clock and thread CPU time).
  3. Counts per call: candidates, rollout row-steps, barrier evaluations (reference: two passes over
     all rows every step; fast: one pass over the moving rows).
  4. Memory: tracemalloc peak per call (subsample). 5. cProfile breakdown (subsample).
  6. Cost-to-go field construction time per scenario (once per episode, outside the control loop).
Configuration (process priority normal, one thread, power state, CPU, background load) is recorded.
Outputs: research/experiments/round4/profile/{calls.csv, steps.csv, summary.json, cprofile_*.txt}.
"""

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from safenav.recovery.timing import thread_cpu_s  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "profile"
PCT = (50, 90, 95, 99)


class Recording:
    """Wraps a planner and records the input of every plan() call."""

    def __init__(self, inner, log, scen):
        self.inner, self.log, self.scen = inner, log, scen
        self.library, self.horizon = inner.library, inner.horizon

    def plan(self, model, field, pos, theta, extra=None):
        self.log.append({"scenario": self.scen, "pos": np.asarray(pos, float).copy(), "theta": float(theta),
                         "extra": None if extra is None else np.asarray(extra).copy(), "field": field.name})
        return self.inner.plan(model, field, pos, theta, extra)

    def predicted_cost(self, *a, **k):
        return self.inner.predicted_cost(*a, **k)


def pct(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {}
    return {**{f"p{p}": float(np.percentile(x, p)) for p in PCT}, "max": float(x.max()), "mean": float(x.mean()), "n": int(x.size)}


def config_record():
    proc = psutil.Process()
    bat = psutil.sensors_battery()
    try:
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        cpu = platform.processor()
    import torch
    return {"cpu": cpu, "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "priority": str(proc.nice()), "on_mains_power": bool(bat.power_plugged) if bat else None,
            "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def run_episodes(planner_name, agents, seeds, config, log=None):
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage
    from safenav.factory import make_env
    from safenav.recovery import RecoveryConfig, RecoveryController
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    rows, steps = [], []
    for name, tseed, ckpt in agents:
        policy = PPOPolicy.from_checkpoint(ckpt, deterministic=True)
        for mode in ("A", "C"):
            for s in seeds:
                ctl = RecoveryController(policy, RecoveryConfig(mode=mode, planner=planner_name))
                if log is not None:
                    ctl.planner = Recording(ctl.planner, log, s)
                obs, _ = env.reset(seed=s)
                ctl.reset(seed=s)
                ctl.bind(env)
                term = trunc = False
                traj_hash = 0.0
                while not (term or trunc):
                    w0, c0 = time.perf_counter_ns(), int(thread_cpu_s() * 1e9)
                    a = ctl.act(obs)
                    w1, c1 = time.perf_counter_ns(), int(thread_cpu_s() * 1e9)
                    steps.append((planner_name, mode, name, tseed, s, (w1 - w0) / 1e6, (c1 - c0) / 1e6, ctl.in_recovery))
                    obs, _, term, trunc, info = env.step(a)
                ctl.notify_goal(int(info["step"]))
                tr = np.asarray(env.unwrapped.trajectory)
                rows.append({"planner": planner_name, "mode": mode, "agent": name, "train_seed": tseed, "episode_seed": s,
                             "goal": bool(info["goal_reached"]), "steps": int(info["step"]),
                             "end_x": float(tr[-1, 0]), "end_y": float(tr[-1, 1]), "path_sum": float(tr.sum()),
                             "calls": ctl.stats.planner_calls, "activations": len(ctl.stats.activations),
                             "field_setup_ms": 1000 * ctl.field_setup_s})
    return pd.DataFrame(rows), pd.DataFrame(steps, columns=["planner", "mode", "agent", "train_seed", "episode_seed",
                                                            "act_wall_ms", "act_cpu_ms", "in_recovery"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scen", type=int, default=20)
    ap.add_argument("--calls", type=int, default=4000)
    ap.add_argument("--seed-base", type=int, default=4_100_000)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS if hasattr(psutil, "NORMAL_PRIORITY_CLASS") else 0)
    from safenav.config import load_config
    from safenav.factory import make_env
    from safenav.config import get_stage
    from safenav.recovery import EscapePlanner, EuclideanField, FilteredUnicycleModel, GeodesicField
    from safenav.recovery.fast import FastEscapePlanner, rollout_end
    from safenav.recovery.timing import calibrate
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides={"env.goal_sampling": "rejection"})
    agents = [(v, s, ROOT / "results" / "checkpoints" / v / f"seed_{s}.pt") for v in ("filtered", "unfiltered") for s in range(5)]
    seeds = [args.seed_base + i for i in range(args.n_scen)]
    summary = {"config": config_record(), "seed_base": args.seed_base, "n_scenarios": args.n_scen,
               "cpu_clock": calibrate()}
    cpu0 = psutil.cpu_percent(None)

    # 1. episodes
    log = []
    t0 = time.time()
    ep_ref, st_ref = run_episodes("reference", agents, seeds, config, log)
    load_ref = psutil.cpu_percent(None)
    ep_fast, st_fast = run_episodes("fast", agents, seeds, config)
    load_fast = psutil.cpu_percent(None)
    key = ["mode", "agent", "train_seed", "episode_seed"]
    m = ep_ref.merge(ep_fast, on=key, suffixes=("_r", "_f"))
    ident = ((m.goal_r == m.goal_f) & (m.steps_r == m.steps_f) & (m.end_x_r == m.end_x_f) & (m.end_y_r == m.end_y_f)
             & (m.path_sum_r == m.path_sum_f) & (m.calls_r == m.calls_f) & (m.activations_r == m.activations_f))
    steps = pd.concat([st_ref, st_fast])
    steps.to_csv(OUT / "steps.csv", index=False)
    summary["episodes"] = {"per_planner": int(len(ep_ref)), "identical_reference_vs_fast": int(ident.sum()),
                           "planner_calls_recorded": len(log), "wall_s": round(time.time() - t0, 1),
                           "system_cpu_percent_during": {"reference": load_ref, "fast": load_fast, "before": cpu0}}
    summary["controller_step_latency_ms"] = {
        p: {mode: {"wall": pct(g.act_wall_ms), "cpu": pct(g.act_cpu_ms),
                   "wall_recovery_steps": pct(g[g.in_recovery].act_wall_ms)}
            for mode, g in steps[steps.planner == p].groupby("mode")} for p in ("reference", "fast")}
    summary["field_setup_ms"] = pct(ep_ref.field_setup_ms)

    # 2-3. isolated calls
    rng = np.random.default_rng(2026)
    pick = sorted(rng.choice(len(log), size=min(args.calls, len(log)), replace=False))
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    ref, fast = EscapePlanner(), FastEscapePlanner()
    cache, rows = {}, []
    for j, i in enumerate(pick):
        c = log[i]
        if c["scenario"] not in cache:
            env.reset(seed=c["scenario"])
            b = env.unwrapped
            model = FilteredUnicycleModel.from_env(env)
            tf = []
            for _ in range(5):
                w0 = time.perf_counter_ns()
                g = GeodesicField(b.obstacle_map.obstacles, b.goal_pos)
                tf.append((time.perf_counter_ns() - w0) / 1e6)
            cache = {c["scenario"]: (model, {"geodesic": g, "euclidean": EuclideanField(b.goal_pos)}, tf)}
        model, fields, _ = cache[c["scenario"]]
        field = fields[c["field"]]
        order = (ref, fast) if j % 2 == 0 else (fast, ref)
        res = {}
        for _ in range(3):
            for pl in order:
                w0, c0 = time.perf_counter_ns(), int(thread_cpu_s() * 1e9)
                p = pl.plan(model, field, c["pos"], c["theta"], c["extra"])
                w1, c1 = time.perf_counter_ns(), int(thread_cpu_s() * 1e9)
                nm = "fast" if pl is fast else "reference"
                res.setdefault(nm, []).append(((w1 - w0) / 1e6, (c1 - c0) / 1e6, p.index))
        cands = ref.library if c["extra"] is None else np.concatenate([ref.library, c["extra"]])
        n, H = cands.shape[:2]
        _, reached = rollout_end(model, c["pos"], c["theta"], cands)
        stop = np.where(reached >= 0, reached, H)
        moving = sum(int(np.sum(cands[k, :stop[k], 0] > 0)) for k in range(n))
        rows.append({"call": int(i), "scenario": c["scenario"], "field": c["field"], "candidates": n, "horizon": H,
                     "obstacles": len(model.ox), "row_steps": n * H, "barrier_evals_reference": 2 * n * H + n,
                     "barrier_evals_fast": moving, "same_plan": res["reference"][0][2] == res["fast"][0][2],
                     **{f"{nm}_wall_ms_{k}": res[nm][k][0] for nm in res for k in range(3)},
                     **{f"{nm}_cpu_ms_{k}": res[nm][k][1] for nm in res for k in range(3)}})
    calls = pd.DataFrame(rows)
    calls.to_csv(OUT / "calls.csv", index=False)
    summary["isolated_calls"] = {"n": int(len(calls)), "same_plan": int(calls.same_plan.sum()),
                                 "candidates": calls.candidates.value_counts().to_dict(),
                                 "obstacles": pct(calls.obstacles), "row_steps_mean": float(calls.row_steps.mean()),
                                 "barrier_evals": {"reference_mean": float(calls.barrier_evals_reference.mean()),
                                                   "fast_mean": float(calls.barrier_evals_fast.mean())}}
    for nm in ("reference", "fast"):
        wall = calls[[f"{nm}_wall_ms_{k}" for k in range(3)]].to_numpy().ravel()
        cpu = calls[[f"{nm}_cpu_ms_{k}" for k in range(3)]].to_numpy().ravel()
        best = calls[[f"{nm}_cpu_ms_{k}" for k in range(3)]].min(axis=1)
        summary["isolated_calls"][nm] = {"wall_ms_all_reps": pct(wall), "cpu_ms_all_reps": pct(cpu), "cpu_ms_best_of_3": pct(best)}
    ratio = calls[[f"fast_cpu_ms_{k}" for k in range(3)]].min(axis=1) / calls[[f"reference_cpu_ms_{k}" for k in range(3)]].min(axis=1)
    summary["isolated_calls"]["fast_over_reference_cpu_best_of_3"] = pct(ratio)
    summary["field_construction_ms_isolated"] = pct([x for v in cache.values() for x in v[2]])

    # 4. memory (subsample of 200)
    mem = {"reference": [], "fast": []}
    for i in pick[:200]:
        c = log[i]
        env.reset(seed=c["scenario"])
        b = env.unwrapped
        model = FilteredUnicycleModel.from_env(env)
        field = GeodesicField(b.obstacle_map.obstacles, b.goal_pos) if c["field"] == "geodesic" else EuclideanField(b.goal_pos)
        for nm, pl in (("reference", ref), ("fast", fast)):
            tracemalloc.start()
            pl.plan(model, field, c["pos"], c["theta"], c["extra"])
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem[nm].append(peak / 1024)
    summary["memory_peak_kib_per_call"] = {k: pct(v) for k, v in mem.items()}

    # 5. cProfile (subsample of 300)
    for nm, pl in (("reference", ref), ("fast", fast)):
        pr = cProfile.Profile()
        sub = pick[:300]
        prepared = []
        for i in sub:
            c = log[i]
            env.reset(seed=c["scenario"])
            b = env.unwrapped
            model = FilteredUnicycleModel.from_env(env)
            field = GeodesicField(b.obstacle_map.obstacles, b.goal_pos) if c["field"] == "geodesic" else EuclideanField(b.goal_pos)
            prepared.append((model, field, c))
        pr.enable()
        for model, field, c in prepared:
            pl.plan(model, field, c["pos"], c["theta"], c["extra"])
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
        (OUT / f"cprofile_{nm}.txt").write_text(s.getvalue(), encoding="utf-8")
    summary["system_cpu_percent_end"] = psutil.cpu_percent(None)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    ic = summary["isolated_calls"]
    print(json.dumps({"identical_episodes": summary["episodes"]["identical_reference_vs_fast"], "of": summary["episodes"]["per_planner"],
                      "reference_cpu": ic["reference"]["cpu_ms_all_reps"], "fast_cpu": ic["fast"]["cpu_ms_all_reps"],
                      "ratio": ic["fast_over_reference_cpu_best_of_3"]["p50"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
