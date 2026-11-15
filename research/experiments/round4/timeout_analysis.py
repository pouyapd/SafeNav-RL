"""
timeout_analysis.py — Part T of round 4 (exploratory): why do ~14 % of C's episodes time out while moving?

    python research/experiments/round4/timeout_analysis.py [--sample 200]

Subject: the episodes in which method C ended in a "moving" timeout in the round-3 main experiment
(research/experiments/round3/results/main, seeds 8,500,000+; already observed data).
1. Decomposition: every such episode is replayed with C (fast planner; identical plans to the
   round-3 reference planner) and per-step instrumentation.
2. Counterfactuals (one factor each) on a seeded random sample of `sample` episodes:
   control (unchanged; must reproduce the moving timeout), max_steps_800, window_15, no_handback,
   planner_only, gamma_1.0. Diagnostics only; nothing here is a proposed change of the method.
Outputs: research/experiments/round4/timeouts/{decomposition.csv, counterfactuals.csv, summary.json}.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "experiments" / "lightweight"))
from run_eval import classify  # noqa: E402

OUT = Path(__file__).resolve().parent / "timeouts"
R3 = ROOT / "research" / "experiments" / "round3" / "results" / "main" / "episodes.csv"


def make(config_over, task_over):
    from safenav.config import get_stage, load_config
    from safenav.factory import make_env
    cfg = load_config(ROOT / "configs" / "default_config.yaml", overrides={"env.goal_sampling": "rejection", **config_over})
    return make_env(cfg, task=dict(get_stage(cfg, "expert"), **task_over), safety_filter=True)


def episode(env, ctl, seed, instrument=False):
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    ctl.reset(seed=seed)
    ctl.bind(env, seed)
    geo = ctl.geo
    rows, v_nom, interv = [], [], []
    term = trunc = False
    while not (term or trunc):
        a = np.asarray(ctl.act(obs), dtype=float)
        rec = bool(getattr(ctl, "in_recovery", False))
        obs, _, term, trunc, info = env.step(a)
        v_nom.append(float(np.clip(a[0], 0, 1)))
        interv.append(bool(info["safety_intervened"]))
        if instrument:
            rows.append((rec, float(np.clip(a[0], 0, 1)), float(info["executed_action"][0]), bool(info["safety_intervened"]),
                         float(geo(base.robot_pos))))
    if hasattr(ctl, "notify_goal"):
        ctl.notify_goal(int(info["step"]))
    outcome = "success" if info["goal_reached"] else ("collision" if info["collision"] else "timeout")
    cls = classify(np.asarray(base.trajectory), v_nom, interv)[0] if outcome == "timeout" else ""
    return outcome, cls, int(info["step"]), rows, float(geo(base.robot_pos)), float(min_clear(base))


def min_clear(base):
    return base.clearance()[0]


def decompose(rows, ctl, start_cost, window):
    rec = np.array([r[0] for r in rows])
    vcmd = np.array([r[1] for r in rows])
    vexe = np.array([r[2] for r in rows])
    lim = np.array([r[3] for r in rows])
    cost = np.array([start_cost] + [r[4] for r in rows])
    acts = ctl.stats.activations
    first = acts[0].start_step if acts else len(rows)
    # policy segments after the first activation: cost regained (sum of increases over each segment)
    regress, seg_start = 0.0, None
    for t in range(first, len(rows)):
        if not rec[t] and seg_start is None:
            seg_start = t
        if (rec[t] or t == len(rows) - 1) and seg_start is not None:
            regress += max(0.0, cost[t + 1] - cost[seg_start])
            seg_start = None
    waiting = sum(min(window, a.start_step - (acts[i - 1].end_step if i else 0)) for i, a in enumerate(acts))
    moving = vcmd > 0
    return {"steps_before_first_activation": int(first), "activations": len(acts), "trigger_wait_steps": int(waiting),
            "recovery_steps": int(rec.sum()), "recovery_turn_steps": int((rec & (vcmd == 0)).sum()),
            "recovery_drive_steps": int((rec & (vcmd > 0)).sum()), "policy_steps": int((~rec).sum()),
            "policy_regress_m": float(regress),
            "filter_limited_share": float(lim.mean()),
            "speed_ratio_exec_over_cmd": float(vexe[moving].sum() / max(vcmd[moving].sum(), 1e-9)),
            "mean_exec_speed": float(vexe.mean()), "remaining_geo_m": float(cost[-1]), "start_geo_m": float(cost[0]),
            "progress_m": float(cost[0] - cost[-1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    import psutil
    if sys.platform == "win32":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.recovery import RecoveryConfig, RecoveryController
    from safenav.recovery.global_baselines import PlannerOnly
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r3 = pd.read_csv(R3)
    sel = r3[(r3.method == "C") & (r3["class"] == "moving")].sort_values(["agent", "train_seed", "episode_seed"])
    pols = {}

    def policy(agent, seed):
        if (agent, seed) not in pols:
            pols[(agent, seed)] = PPOPolicy.from_checkpoint(ROOT / "results" / "checkpoints" / agent / f"seed_{seed}.pt", deterministic=True)
        return pols[(agent, seed)]

    env = make({}, {})
    dec = []
    for r in sel.itertuples():
        ctl = RecoveryController(policy(r.agent, r.train_seed), RecoveryConfig(mode="C", planner="fast"))
        outcome, cls, steps, rows, end_cost, _ = episode(env, ctl, int(r.episode_seed), instrument=True)
        d = decompose(rows, ctl, r.start_geo_cost, ctl.cfg.window)
        dec.append({"agent": r.agent, "train_seed": r.train_seed, "episode_seed": r.episode_seed, "outcome": outcome,
                    "class": cls, "steps": steps, "reproduces_round3": bool(outcome == "timeout" and cls == "moving"),
                    **d})
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "decomposition.csv", index=False)

    rng = np.random.default_rng(2026)
    sample = sel.iloc[sorted(rng.choice(len(sel), size=min(args.sample, len(sel)), replace=False))]
    variants = {
        "control": ({}, {}, {}),
        "max_steps_800": ({}, {"max_steps": 800}, {}),
        "window_15": ({}, {}, {"window": 15}),
        "no_handback": ({}, {}, {"exit_gain": 1e9, "min_gain": -1e9, "max_steps": 10**6}),
        "planner_only": ({}, {}, None),
        "gamma_1.0": ({"safety.cbf_gamma": 1.0}, {}, {}),
    }
    cf = []
    for vname, (cfg_over, task_over, rc) in variants.items():
        venv = make(cfg_over, task_over)
        for r in sample.itertuples():
            if rc is None:
                ctl = PlannerOnly(planner="fast")
            else:
                ctl = RecoveryController(policy(r.agent, r.train_seed), RecoveryConfig(mode="C", planner="fast", **rc))
            outcome, cls, steps, _, end_cost, clear = episode(venv, ctl, int(r.episode_seed))
            cf.append({"variant": vname, "agent": r.agent, "train_seed": r.train_seed, "episode_seed": r.episode_seed,
                       "outcome": outcome, "class": cls, "steps": steps, "end_geo_cost": end_cost})
    cf = pd.DataFrame(cf)
    cf.to_csv(OUT / "counterfactuals.csv", index=False)

    summ = {"episodes": int(len(dec)), "reproduced_moving_timeouts": int(((dec.outcome == "timeout") & (dec["class"] == "moving")).sum()),
            "wall_s": round(time.time() - t0, 1),
            "decomposition_median": dec.drop(columns=["agent", "train_seed", "episode_seed", "outcome", "class", "reproduces_round3"]).median().to_dict(),
            "decomposition_mean": dec.drop(columns=["agent", "train_seed", "episode_seed", "outcome", "class", "reproduces_round3"]).mean().to_dict(),
            "share_policy_regress_over_1m": float((dec.policy_regress_m > 1.0).mean()),
            "share_remaining_over_2m": float((dec.remaining_geo_m > 2.0).mean()),
            "share_start_geo_over_10m": float((dec.start_geo_m > 10.0).mean()),
            "counterfactuals": {v: {"n": int(len(g)), "success": float((g.outcome == "success").mean()),
                                    "moving_timeout": float((g["class"] == "moving").mean()),
                                    "deadlock": float(g["class"].isin(["filter-blocked", "self-stop", "other stall"]).mean()),
                                    "collision": float((g.outcome == "collision").mean())}
                                for v, g in cf.groupby("variant")}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ["counterfactuals"], indent=1), summ["reproduced_moving_timeouts"], "/", summ["episodes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
