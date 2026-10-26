"""
audit_diagnostics.py

Read-only diagnostics for research/REPOSITORY_AUDIT.md. It loads the committed
checkpoints and result files, never writes to results/, and stores its own
outputs in research/audit_outputs/.

    python research/scripts/audit_diagnostics.py            # about 3-5 minutes on 8 cores

Diagnostics:
  1. scenario generator: requested vs actual start-goal distances (test seeds and 3,000 fresh seeds)
  2. timeout attribution for the PPO agents with the filter on (expert stage): is the robot
     stationary at the end, and is it the filter or the policy itself that stops it?
  3. range-sensor coverage: share of nearby obstacles that no beam hits
  4. statistics: variance between training seeds vs between test scenarios, and paired
     hierarchical-bootstrap intervals for the main comparisons
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safenav.config import get_stage, load_config  # noqa: E402
from safenav.env.navigation_env import NavigationEnv  # noqa: E402
from safenav.evaluation.runner import episode_seeds  # noqa: E402

OUT = ROOT / "research" / "audit_outputs"
REQUESTED = {"beginner": (3.0, 5.0), "intermediate": (5.0, 8.0), "expert": (5.0, 12.0)}
STAGES = ("beginner", "intermediate", "expert")
TAIL = 100          # steps at the end of a timeout episode that are inspected
STALL_DISP = 0.10   # metres: net displacement over the tail below which the robot counts as stalled


# ── 1. scenario generator ────────────────────────────────────────────────────

def goal_distances(config):
    rows = []
    for idx, stage in enumerate(STAGES):
        env = NavigationEnv(config, task=get_stage(config, stage))
        lo, _ = REQUESTED[stage]
        for name, seeds in (("test", episode_seeds(config["evaluation"]["test_seed"], idx, 100)),
                            ("fresh_3000", range(10_000_000, 10_003_000))):
            d = []
            for s in seeds:
                env.reset(seed=int(s))
                d.append(float(np.linalg.norm(env.goal_pos - env.start_pos)))
            d = np.asarray(d)
            rows.append({"stage": stage, "seed_set": name, "n": len(d), "requested_min": lo,
                         "share_below_requested_min": float((d < lo).mean()),
                         "share_start_inside_goal": float((d < env.goal_radius).mean()),
                         "mean": float(d.mean()), "median": float(np.median(d)), "max": float(d.max())})
    return pd.DataFrame(rows)


# ── 2. timeout attribution ───────────────────────────────────────────────────

def _attribution_job(args):
    variant, seed, deterministic, stage = args
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.factory import make_env

    config = load_config()
    stage_idx = STAGES.index(stage)
    ckpt = ROOT / "results" / "checkpoints" / variant / f"seed_{seed}.pt"
    policy = PPOPolicy.from_checkpoint(ckpt, deterministic=deterministic)
    env = make_env(config, task=get_stage(config, stage), safety_filter=True)
    flt = env.safety_filter
    rows = []
    for ep_seed in episode_seeds(config["evaluation"]["test_seed"], stage_idx, 100):
        obs, _ = env.reset(seed=ep_seed)
        policy.reset(seed=ep_seed)
        base = env.unwrapped
        v_nom, v_lim, omega, interv = [], [], [], []
        done = False
        while not done:
            a = policy.act(obs)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            v_nom.append(float(info["nominal_action"][0]))
            v_lim.append(float(info["speed_limit"]))
            omega.append(float(info["executed_action"][1]))
            interv.append(bool(info["safety_intervened"]))
        if not (trunc and not term):
            continue  # only timeouts are attributed
        traj = np.asarray(base.trajectory)
        tail = slice(-TAIL, None)
        disp = float(np.linalg.norm(traj[-1] - traj[-TAIL - 1]))
        path_tail = float(np.sum(np.linalg.norm(np.diff(traj[-TAIL - 1:], axis=0), axis=1)))
        vn, vl = np.asarray(v_nom[tail]), np.asarray(v_lim[tail])
        om = np.asarray(omega[tail])
        h_end = flt.min_barrier(base.robot_pos, base.obstacle_map.obstacles)
        stalled = disp < STALL_DISP
        filter_blocked = float(np.mean(np.asarray(interv[tail])))
        policy_stop = float(np.mean(vn <= 1e-3))
        if not stalled:
            cls = "moving (no stall)"
        elif filter_blocked >= 0.5:
            cls = "stall: filter blocks the commanded speed"
        elif policy_stop >= 0.5:
            cls = "stall: policy commands v = 0 itself"
        else:
            cls = "stall: mixed"
        signs = np.sign(om[np.abs(om) > 1e-3])
        # Geometry at the end: is the robot facing the nearest obstacle, with the goal behind it?
        near = base.obstacle_map.nearest_obstacle(*base.robot_pos)
        to_obs = np.array([near.x, near.y]) - base.robot_pos
        to_goal = base.goal_pos - base.robot_pos
        heading = np.array([np.cos(base.robot_theta), np.sin(base.robot_theta)])
        cos_heading_obs = float(heading @ to_obs / np.linalg.norm(to_obs))
        cos_goal_obs = float(to_goal @ to_obs / (np.linalg.norm(to_goal) * np.linalg.norm(to_obs) + 1e-12))
        rows.append({
            "variant": variant, "train_seed": seed, "actions": "mean" if deterministic else "sampled",
            "stage": stage, "episode_seed": ep_seed, "class": cls,
            "tail_net_displacement_m": disp, "tail_path_length_m": path_tail,
            "tail_share_filter_intervened": filter_blocked, "tail_share_nominal_v_zero": policy_stop,
            "tail_mean_abs_omega": float(np.mean(np.abs(om))),
            "tail_omega_sign_changes": int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0,
            "tail_mean_nominal_v": float(vn.mean()), "tail_mean_speed_limit": float(vl.mean()),
            "end_min_barrier_m": float(h_end),
            "end_dist_to_goal_m": float(np.linalg.norm(base.goal_pos - base.robot_pos)),
            "end_cos_heading_to_nearest_obstacle": cos_heading_obs,
            "end_cos_goal_dir_to_nearest_obstacle_dir": cos_goal_obs,
        })
    return rows


def timeout_attribution(jobs=8, stage="expert"):
    tasks = [(v, s, det, stage) for v in ("filtered", "unfiltered") for s in range(5) for det in (True, False)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = [r for part in pool.map(_attribution_job, tasks) for r in part]
    return pd.DataFrame(rows)


# ── 3. range-sensor coverage ─────────────────────────────────────────────────

def sensor_coverage(config, n_scenarios=200, poses_per_scenario=20, seed=0):
    """For random free poses in expert scenarios: share of obstacles (by surface distance
    from the robot body) that no beam of the noise-free scan hits."""
    rng = np.random.default_rng(seed)
    env = NavigationEnv(config, task=get_stage(config, "expert"))
    bins = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.7]
    hits = {b: [0, 0] for b in bins[:-1]}
    for s in range(n_scenarios):
        env.reset(seed=20_000_000 + s)
        om = env.obstacle_map
        k = 0
        while k < poses_per_scenario:
            p = rng.uniform(0.5, env.workspace_size - 0.5, 2)
            if om.min_surface_distance(*p) < env.robot_radius:
                continue
            k += 1
            theta = rng.uniform(-np.pi, np.pi)
            angles = theta + env.beam_offsets
            ux, uy = np.cos(angles), np.sin(angles)
            for o in om.obstacles:
                surf = o.distance_to_point(*p) - env.robot_radius   # body-to-surface distance
                if surf < 0 or surf >= bins[-1]:   # bins[-1] = max_range - robot_radius
                    continue
                # is this obstacle the first hit along any beam within max_range?
                ocx, ocy = p[0] - o.x, p[1] - o.y
                b = ux * ocx + uy * ocy
                disc = b ** 2 - (ocx ** 2 + ocy ** 2 - o.radius ** 2)
                t = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
                t = np.where(t >= 0, t, np.inf)
                first = om.ray_cast(p, angles, env.max_range)
                seen = bool(np.any(np.isfinite(t) & (t <= env.max_range) & (np.abs(t - first) < 1e-9)))
                for lo, hi in zip(bins[:-1], bins[1:]):
                    if lo <= surf < hi:
                        hits[lo][0] += int(seen)
                        hits[lo][1] += 1
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        seen, n = hits[lo]
        rows.append({"body_to_surface_m": f"[{lo}, {hi})", "obstacles": n,
                     "share_missed_by_all_beams": 1 - seen / n if n else float("nan")})
    return pd.DataFrame(rows)


# ── 4. statistics ────────────────────────────────────────────────────────────

def variance_components(df, value="success"):
    """Two-way (seed x scenario) decomposition of a binary outcome for one condition:
    share of variance explained by seed main effect, scenario main effect, and residual."""
    m = df.pivot_table(index="train_seed", columns="episode_seed", values=value, aggfunc="mean").to_numpy(float)
    grand = m.mean()
    seed_eff = m.mean(axis=1) - grand
    scen_eff = m.mean(axis=0) - grand
    resid = m - grand - seed_eff[:, None] - scen_eff[None, :]
    ss_seed = m.shape[1] * np.sum(seed_eff ** 2)
    ss_scen = m.shape[0] * np.sum(scen_eff ** 2)
    ss_res = np.sum(resid ** 2)
    tot = ss_seed + ss_scen + ss_res
    return {"seeds": m.shape[0], "scenarios": m.shape[1], "mean": float(grand),
            "share_ss_seed": ss_seed / tot, "share_ss_scenario": ss_scen / tot, "share_ss_residual": ss_res / tot}


def paired_hier_bootstrap(a, b, n_boot=10_000, seed=0):
    """
    Difference of means a - b for two conditions evaluated on the same scenarios.
    a, b: arrays (seeds_a, scenarios) and (seeds_b, scenarios). Resamples scenarios
    (shared, paired) and, independently, training seeds within each condition.
    Returns (diff, low, high) of a 95% percentile interval.
    """
    rng = np.random.default_rng(seed)
    n_sc = a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sc = rng.integers(0, n_sc, n_sc)
        sa = rng.integers(0, a.shape[0], a.shape[0])
        sb = rng.integers(0, b.shape[0], b.shape[0])
        diffs[i] = a[np.ix_(sa, sc)].mean() - b[np.ix_(sb, sc)].mean()
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def matrix(df, **sel):
    g = df
    for k, v in sel.items():
        g = g[g[k] == v]
    if g["train_seed"].isna().all():
        g = g.assign(train_seed=0)
    return g.pivot_table(index="train_seed", columns="episode_seed", values="success", aggfunc="mean").to_numpy(float)


def statistics():
    bench = pd.read_csv(ROOT / "results" / "benchmark" / "episodes.csv")
    stoch = pd.read_csv(ROOT / "results" / "stochastic_eval" / "episodes.csv")
    ex_b, ex_s = bench[bench.stage == "expert"], stoch[stoch.stage == "expert"]

    comps = []
    for name, sub in [
        ("PPO no filter in training | filter off", dict(method="PPO (no filter in training)", safety_filter=False)),
        ("PPO no filter in training | filter on", dict(method="PPO (no filter in training)", safety_filter=True)),
        ("PPO filter in training | filter on", dict(method="PPO (filter in training)", safety_filter=True)),
    ]:
        g = ex_b
        for k, v in sub.items():
            g = g[g[k] == v]
        comps.append(dict(condition=name, **variance_components(g)))
    for variant in ("filtered", "unfiltered"):
        g = ex_s[(ex_s.variant == variant) & (ex_s.actions == "sampled")]
        comps.append(dict(condition=f"PPO {variant} training | filter on | sampled", **variance_components(g)))
    var_df = pd.DataFrame(comps)

    A = {
        "unf_off": matrix(ex_b, method="PPO (no filter in training)", safety_filter=False),
        "unf_on": matrix(ex_b, method="PPO (no filter in training)", safety_filter=True),
        "fil_on": matrix(ex_b, method="PPO (filter in training)", safety_filter=True),
        "unf_on_sampled": matrix(ex_s, variant="unfiltered", actions="sampled"),
        "fil_on_sampled": matrix(ex_s, variant="filtered", actions="sampled"),
        "pf_on": matrix(ex_b, method="PotentialField", safety_filter=True),
    }
    rows = []
    for label, a, b in [
        ("cost of adding the filter at test time (unfiltered agent): off - on", "unf_off", "unf_on"),
        ("filter-trained vs unfiltered-trained, filter on, mean actions", "fil_on", "unf_on"),
        ("filter-trained vs unfiltered-trained, filter on, sampled actions", "fil_on_sampled", "unf_on_sampled"),
        ("sampled - mean actions, filter-trained agent", "fil_on_sampled", "fil_on"),
        ("sampled - mean actions, unfiltered-trained agent", "unf_on_sampled", "unf_on"),
        ("filter-trained (sampled) vs potential field, filter on", "fil_on_sampled", "pf_on"),
    ]:
        d, lo, hi = paired_hier_bootstrap(A[a], A[b])
        rows.append({"comparison (expert, success rate)": label, "difference": d, "ci95_low": lo, "ci95_high": hi})
    return var_df, pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    summary = {}

    gd = goal_distances(config)
    gd.to_csv(OUT / "goal_distances.csv", index=False)
    print(gd.round(3).to_string(index=False), "\n")

    ta = timeout_attribution()
    ta.to_csv(OUT / "timeout_attribution_expert.csv", index=False)
    counts = ta.groupby(["variant", "actions", "class"]).size().unstack(fill_value=0)
    print(counts, "\n")
    print(ta.groupby(["variant", "actions"])[["tail_net_displacement_m", "end_min_barrier_m",
                                              "end_dist_to_goal_m", "tail_mean_abs_omega",
                                              "end_cos_heading_to_nearest_obstacle",
                                              "end_cos_goal_dir_to_nearest_obstacle_dir"]].median().round(3), "\n")
    summary["timeout_classes"] = {f"{k[0]}|{k[1]}": v for k, v in counts.to_dict(orient="index").items()}

    sc = sensor_coverage(config)
    sc.to_csv(OUT / "sensor_coverage.csv", index=False)
    print(sc.round(3).to_string(index=False), "\n")

    var_df, comp_df = statistics()
    var_df.to_csv(OUT / "variance_components_expert.csv", index=False)
    comp_df.to_csv(OUT / "paired_comparisons_expert.csv", index=False)
    print(var_df.round(3).to_string(index=False), "\n")
    print(comp_df.round(3).to_string(index=False))

    (OUT / "README.md").write_text(
        "Outputs of `python research/scripts/audit_diagnostics.py` (read-only analysis of the committed\n"
        "checkpoints and results; see research/REPOSITORY_AUDIT.md). Nothing in results/ is modified.\n",
        encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
