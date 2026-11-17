"""
mechanism.py — Phase 2 of round 3: what do self-stops and filter-blocked stalls have in common?
(exploratory and descriptive; analysis plan fixed in this docstring before the run)

    python research/experiments/round3/mechanism.py [--n-e3 300] [--n-e4 200]

Data: replays of scenarios that round 2 already evaluated (so this is exploratory, not a
fresh test). All agents run with the policy mean and the filter on.
  * E3 expert, corrected generator, seeds 7,600,000 + i (i < n_e3): the 10 committed agents;
  * E4 expert, seeds 6,500,000 + i (i < n_e4): control, riskfree and beta_policy (5 seeds each).
Every replayed episode is checked against the stored round-2 row (outcome, steps, class).

Units of analysis:
  * stall: a timeout classified filter-blocked, self-stop or other stall (round-2 rules),
    features taken at the final state;
  * close passage (reference): a successful episode whose smallest obstacle barrier was
    below 0.30 m, features taken at the step of that smallest barrier.

Features at a state (p, theta):
  goal_obstacle_angle  angle between the goal direction and the direction of the nearest obstacle (deg)
  heading_obstacle_angle, heading_goal_angle  (deg)
  free_beams           share of the 16 noise-free range readings longer than 1.0 m
  geo_reachable        the goal is reachable through the filter's safe set (oracle field)
  detour_ratio         geodesic distance to the goal region / (straight-line distance - goal radius)
  path_deviation       angle between the first 0.5 m of the shortest safe path and the goal direction (deg)
  euclid_local_min     no admissible straight move of 0.30 m (72 directions; the end point and the
                       midpoint keep the margin) reduces the straight-line distance by >= 0.05 m
  escape_geodesic      the best of the 65 manoeuvres (4 s, through the filter model) reduces the
                       geodesic cost-to-go by >= 1.0 m
  escape_euclidean     the best manoeuvre reduces the straight-line distance by >= 0.5 m
  tail_interv_share, tail_omega_sign_changes   filter interventions and turn-rate reversals, last 100 steps

Hypotheses (exploratory; stated before the run):
  M1  stalls need a large deviation from the goal direction (path_deviation > 45 deg) more often than
      close passages of successful episodes do;
  M2  in at least 80 % of stalls a feasible manoeuvre exists that reduces the geodesic cost-to-go by
      >= 1 m (escape_geodesic), i.e. the stall is escapable with lookahead and map knowledge;
  M3  the straight-line distance is locally minimal (euclid_local_min) in most stalls, and much more
      often than in close passages, i.e. a myopic controller cannot make progress there;
  M4  the two stall types share these conditions (rates of M1-M3 within 15 points of each other).
Intervals: 95 % bootstrap over scenarios (episodes of the same scenario are resampled together).
Outputs: research/experiments/round3/mechanism/{episodes.csv,states.csv,summary.json}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "experiments" / "lightweight"))
from run_eval import TAIL, classify, lower_priority  # noqa: E402

OUT = Path(__file__).resolve().parent / "mechanism"
R2 = ROOT / "research" / "experiments" / "round2" / "results"


def angle(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1))))


def features(env, model, geo, euc, planner, pos, theta):
    base = env.unwrapped
    obs = base.obstacle_map.obstacles
    g = base.goal_pos - pos
    h, _ = model.barriers(np.array([pos[0]]), np.array([pos[1]]), np.array([theta]))
    m = len(obs)
    k = int(np.argmin(h[0, :m])) if m else -1
    n = np.array([obs[k].x - pos[0], obs[k].y - pos[1]]) if m else np.array([np.nan, np.nan])
    u = np.array([np.cos(theta), np.sin(theta)])
    saved = (base.robot_pos.copy(), base.robot_theta)
    base.robot_pos, base.robot_theta = np.asarray(pos, dtype=float).copy(), float(theta)
    ranges = base.range_readings(noisy=False)
    base.robot_pos, base.robot_theta = saved
    gd, ed = float(geo(pos)), float(euc(pos))
    pdir = geo.path_direction(pos) if np.isfinite(gd) else None
    # myopic local-minimum test
    dirs = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    cand = pos[None, :] + 0.30 * np.column_stack([np.cos(dirs), np.sin(dirs)])
    mid = pos[None, :] + 0.15 * np.column_stack([np.cos(dirs), np.sin(dirs)])
    hc, _ = model.barriers(cand[:, 0], cand[:, 1], dirs)
    hm, _ = model.barriers(mid[:, 0], mid[:, 1], dirs)
    ok = (hc.min(axis=1) >= 0) & (hm.min(axis=1) >= 0)
    local_min = not np.any(ok & (euc(cand) <= ed - 0.05))
    pg = planner.plan(model, geo, pos, theta)
    pe = planner.plan(model, euc, pos, theta)
    return {"h_obstacle": float(h[0, :m].min()) if m else np.nan,
            "goal_obstacle_angle": angle(g, n), "heading_obstacle_angle": angle(u, n), "heading_goal_angle": angle(u, g),
            "free_beams": float(np.mean(ranges > 1.0)), "geo_reachable": bool(np.isfinite(gd)),
            "detour_ratio": gd / (ed - base.goal_radius) if np.isfinite(gd) and ed - base.goal_radius > 1e-9 else np.nan,
            "path_deviation": angle(pdir, g) if pdir is not None else np.nan,
            "euclid_local_min": bool(local_min),
            "escape_geodesic": bool(pg.reaches_goal or pg.gain >= 1.0), "escape_geodesic_gain": float(pg.gain),
            "escape_euclidean": bool(pe.reaches_goal or pe.gain >= 0.5), "escape_euclidean_gain": float(pe.gain)}


def replay(config, agents, stage, seeds, task_over, tag):
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage
    from safenav.factory import make_env
    from safenav.recovery import EscapePlanner, EuclideanField, FilteredUnicycleModel, GeodesicField
    planner = EscapePlanner()
    env = make_env(config, task=dict(get_stage(config, stage), **task_over), safety_filter=True)
    ep_rows, st_rows = [], []
    for name, tseed, ckpt in agents:
        policy = PPOPolicy.from_checkpoint(ckpt, deterministic=True)
        for s in seeds:
            obs, info = env.reset(seed=s)
            policy.reset(seed=s)
            base = env.unwrapped
            v_nom, interv, om, h_hist, states = [], [], [], [], []
            term = trunc = False
            model = FilteredUnicycleModel.from_env(env)
            while not (term or trunc):
                a = np.asarray(policy.act(obs), dtype=float)
                states.append((base.robot_pos.copy(), float(base.robot_theta)))
                obs, _, term, trunc, info = env.step(a)
                v_nom.append(float(np.clip(a[0], 0, 1)))
                om.append(float(info["executed_action"][1]))
                interv.append(bool(info["safety_intervened"]))
                hh, _ = model.barriers(np.array([base.robot_pos[0]]), np.array([base.robot_pos[1]]), np.array([base.robot_theta]))
                h_hist.append(float(hh[0, :len(base.obstacle_map.obstacles)].min()))
            outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
            cls = classify(np.asarray(base.trajectory), v_nom, interv)[0] if outcome == "timeout" else ""
            ep_rows.append({"set": tag, "agent": name, "train_seed": tseed, "episode_seed": s, "outcome": outcome,
                            "class": cls, "steps": int(info["step"])})
            unit = None
            if cls in ("filter-blocked", "self-stop", "other stall"):
                unit, pos, th = "stall", base.robot_pos.copy(), float(base.robot_theta)
            elif outcome == "success" and min(h_hist) < 0.30:
                k = int(np.argmin(h_hist))
                unit = "close passage"
                pos, th = (states[k + 1] if k + 1 < len(states) else (base.robot_pos.copy(), float(base.robot_theta)))
            if unit:
                geo = GeodesicField(base.obstacle_map.obstacles, base.goal_pos)
                f = features(env, model, geo, EuclideanField(base.goal_pos), planner, pos, th)
                o = np.sign(np.asarray(om[-TAIL:]))
                f.update({"set": tag, "agent": name, "train_seed": tseed, "episode_seed": s, "unit": unit, "class": cls,
                          "tail_interv_share": float(np.mean(interv[-TAIL:])) if unit == "stall" else np.nan,
                          "tail_omega_sign_changes": int(np.sum(o[1:] * o[:-1] < 0)) if unit == "stall" else np.nan})
                st_rows.append(f)
    return ep_rows, st_rows


def boot_prop(df, col, n=2000, seed=3):
    if len(df) == 0:
        return {"n": 0}
    x = df[col].astype(float)
    groups = df.groupby(["set", "episode_seed"]).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        vals.append(np.nanmean(x.to_numpy()[idx]))
    return {"n": int(x.notna().sum()), "share": float(np.nanmean(x)), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-e3", type=int, default=300)
    ap.add_argument("--n-e4", type=int, default=200)
    args = ap.parse_args()
    lower_priority()
    import torch
    torch.set_num_threads(1)
    from safenav.config import load_config
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg_rej = load_config(ROOT / "configs" / "default_config.yaml", overrides={"env.goal_sampling": "rejection"})
    cfg_def = load_config(ROOT / "configs" / "default_config.yaml")
    exp = ROOT / "research" / "experiments"
    a3 = [(v, s, ROOT / "results" / "checkpoints" / v / f"seed_{s}.pt") for v in ("filtered", "unfiltered") for s in range(5)]
    a4 = [(arm, s, d / arm / f"seed_{s}" / "best.pt") for arm, d in (("control", exp / "e1a" / "checkpoints"),
          ("riskfree", exp / "round2" / "checkpoints"), ("beta_policy", exp / "round2" / "checkpoints")) for s in range(5)]
    e1, s1 = replay(cfg_rej, a3, "expert", [7_600_000 + i for i in range(args.n_e3)], {}, "E3")
    e2, s2 = replay(cfg_def, a4, "expert", [6_500_000 + i for i in range(args.n_e4)], {}, "E4")
    ep, st = pd.DataFrame(e1 + e2), pd.DataFrame(s1 + s2)
    # consistency with the stored round-2 rows
    old = pd.concat([pd.read_csv(R2 / "e3_expert_mean" / "episodes.csv").assign(set="E3"),
                     pd.read_csv(R2 / "e4_mean" / "episodes.csv").assign(set="E4")])
    m = ep.merge(old, on=["set", "agent", "train_seed", "episode_seed"], suffixes=("", "_r2"))
    same = (m.outcome == m.outcome_r2) & (m.steps == m.steps_r2) & (m["class"].fillna("") == m["class_r2"].fillna(""))
    ep.to_csv(OUT / "episodes.csv", index=False)
    st.to_csv(OUT / "states.csv", index=False)
    st["dev45"] = st.path_deviation > 45
    summ = {"replayed_episodes": len(ep), "identical_to_round2": int(same.sum()), "matched": len(m),
            "wall_s": round(time.time() - t0, 1), "units": {f"{u}|{c}": int(n) for (u, c), n in st.groupby(["unit", "class"]).size().items()}}
    groups = {"all stalls": st[st.unit == "stall"], "filter-blocked": st[st["class"] == "filter-blocked"],
              "self-stop": st[st["class"] == "self-stop"], "close passages": st[st.unit == "close passage"]}
    for gname, g in groups.items():
        summ[gname] = {c: boot_prop(g, c) for c in ("dev45", "escape_geodesic", "escape_euclidean", "euclid_local_min",
                                                    "geo_reachable")}
        summ[gname]["median"] = {c: float(g[c].median()) for c in ("goal_obstacle_angle", "heading_obstacle_angle",
                                                                    "heading_goal_angle", "free_beams", "detour_ratio",
                                                                    "path_deviation", "h_obstacle") if len(g)}
    stalls, passes = groups["all stalls"], groups["close passages"]
    fb, ss = groups["filter-blocked"], groups["self-stop"]
    summ["hypotheses"] = {
        "M1_dev45_stalls_vs_passages": [summ["all stalls"]["dev45"].get("share"), summ["close passages"]["dev45"].get("share")],
        "M2_escape_geodesic_share_stalls": summ["all stalls"]["escape_geodesic"].get("share"),
        "M3_local_min_stalls_vs_passages": [summ["all stalls"]["euclid_local_min"].get("share"),
                                            summ["close passages"]["euclid_local_min"].get("share")],
        "M4_type_difference_points": {c: 100 * (fb[c].astype(float).mean() - ss[c].astype(float).mean())
                                      for c in ("dev45", "escape_geodesic", "euclid_local_min")}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ["hypotheses"], indent=1), summ["identical_to_round2"], "/", summ["matched"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
