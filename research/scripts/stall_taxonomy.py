"""
stall_taxonomy.py

Read-only extension of audit_diagnostics.py: classifies every timeout with the
filter on, for all three stages, for the 10 committed PPO agents (policy mean
and sampled actions) and for the two baseline controllers. Uses only the
committed checkpoints and the stored test seeds; writes only to
research/audit_outputs/stall_taxonomy_all.csv.

    python research/scripts/stall_taxonomy.py          # about 4 minutes on 8 cores

Classes (same rules as audit_diagnostics.py, last TAIL steps of a timeout):
  filter-blocked  net displacement < STALL_DISP and the filter cut the speed in >= 50 % of the steps
  self-stop       net displacement < STALL_DISP and the policy commanded v <= 0.001 in >= 50 % of the steps
  other stall     net displacement < STALL_DISP, neither of the above
  moving          net displacement >= STALL_DISP
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAGES = ("beginner", "intermediate", "expert")
TAIL, STALL_DISP = 100, 0.10


def _policy(kind, variant, seed, deterministic, config):
    from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy
    from safenav.agents.policy import PPOPolicy
    if kind == "ppo":
        return PPOPolicy.from_checkpoint(ROOT / "results" / "checkpoints" / variant / f"seed_{seed}.pt",
                                         deterministic=deterministic)
    if kind == "gotogoal":
        return GoToGoalPolicy()
    return PotentialFieldPolicy.from_config(config)


def _job(args):
    kind, variant, seed, deterministic = args
    import torch
    torch.set_num_threads(1)
    from safenav.config import get_stage, load_config
    from safenav.evaluation.runner import episode_seeds
    from safenav.factory import make_env

    config = load_config(ROOT / "configs" / "default_config.yaml")
    policy = _policy(kind, variant, seed, deterministic, config)
    rows = []
    for stage_idx, stage in enumerate(STAGES):
        env = make_env(config, task=get_stage(config, stage), safety_filter=True)
        flt = env.safety_filter
        for ep_seed in episode_seeds(config["evaluation"]["test_seed"], stage_idx, 100):
            obs, _ = env.reset(seed=ep_seed)
            policy.reset(seed=ep_seed)
            base = env.unwrapped
            v_nom, interv, omega = [], [], []
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy.act(obs))
                done = term or trunc
                v_nom.append(float(info["nominal_action"][0]))
                interv.append(bool(info["safety_intervened"]))
                omega.append(float(info["executed_action"][1]))
            outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
            row = {"policy": kind, "variant": variant, "train_seed": seed,
                   "actions": "mean" if deterministic else "sampled", "stage": stage,
                   "episode_seed": ep_seed, "outcome": outcome, "class": ""}
            if outcome == "timeout":
                traj = np.asarray(base.trajectory)
                disp = float(np.linalg.norm(traj[-1] - traj[-TAIL - 1]))
                blocked = float(np.mean(interv[-TAIL:]))
                selfstop = float(np.mean(np.asarray(v_nom[-TAIL:]) <= 1e-3))
                if disp >= STALL_DISP:
                    cls = "moving"
                elif blocked >= 0.5:
                    cls = "filter-blocked"
                elif selfstop >= 0.5:
                    cls = "self-stop"
                else:
                    cls = "other stall"
                h, _ = flt.barriers(base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
                n_obs = len(base.obstacle_map.obstacles)
                near = base.obstacle_map.nearest_obstacle(*base.robot_pos)
                to_obs = np.array([near.x, near.y]) - base.robot_pos
                heading = np.array([np.cos(base.robot_theta), np.sin(base.robot_theta)])
                to_goal = base.goal_pos - base.robot_pos
                om = np.asarray(omega[-TAIL:])
                signs = np.sign(om[np.abs(om) > 1e-3])
                row.update({
                    "class": cls, "tail_net_displacement_m": disp,
                    "tail_share_filter_intervened": blocked, "tail_share_nominal_v_zero": selfstop,
                    "end_obstacle_barrier_m": float(h[:n_obs].min()) if n_obs else float("inf"),
                    "end_wall_barrier_m": float(h[n_obs:].min()),
                    "end_heading_to_obstacle_deg": float(np.degrees(np.arccos(np.clip(
                        heading @ to_obs / np.linalg.norm(to_obs), -1, 1)))),
                    "end_goal_to_obstacle_deg": float(np.degrees(np.arccos(np.clip(
                        to_goal @ to_obs / (np.linalg.norm(to_goal) * np.linalg.norm(to_obs) + 1e-12), -1, 1)))),
                    "tail_mean_abs_omega": float(np.mean(np.abs(om))),
                    "tail_omega_sign_changes": int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0,
                })
            rows.append(row)
        env.close()
    return rows


def main() -> int:
    tasks = [("ppo", v, s, det) for v in ("filtered", "unfiltered") for s in range(5) for det in (True, False)]
    tasks += [("gotogoal", "baseline", None, True), ("potential_field", "baseline", None, True)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = [r for part in pool.map(_job, tasks) for r in part]
    df = pd.DataFrame(rows)
    out = ROOT / "research" / "audit_outputs" / "stall_taxonomy_all.csv"
    df.to_csv(out, index=False)
    t = df[df.outcome == "timeout"]
    print(t.groupby(["policy", "variant", "actions", "stage", "class"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
