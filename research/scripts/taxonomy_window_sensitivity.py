"""
taxonomy_window_sensitivity.py

Read-only check of how the stall taxonomy depends on the window length K (last K steps of a
timeout; K = 100 in the paper). Re-runs the expert test episodes of the 10 committed PPO
agents (policy mean, filter on) and classifies every timeout for K in (50, 100, 150, 200),
with the displacement threshold scaled to the window (0.10 m per 100 steps).

    python research/scripts/taxonomy_window_sensitivity.py      # about 2-3 minutes on 8 cores

Output: research/audit_outputs/taxonomy_window_sensitivity_expert.csv
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WINDOWS = (50, 100, 150, 200)


def _job(args):
    variant, seed = args
    import torch
    torch.set_num_threads(1)
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config
    from safenav.evaluation.runner import episode_seeds
    from safenav.factory import make_env

    config = load_config(ROOT / "configs" / "default_config.yaml")
    policy = PPOPolicy.from_checkpoint(ROOT / "results" / "checkpoints" / variant / f"seed_{seed}.pt")
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    rows = []
    for ep_seed in episode_seeds(config["evaluation"]["test_seed"], 2, 100):
        obs, _ = env.reset(seed=ep_seed)
        policy.reset(seed=ep_seed)
        v_nom, interv, done = [], [], False
        while not done:
            obs, _, term, trunc, info = env.step(policy.act(obs))
            done = term or trunc
            v_nom.append(float(info["nominal_action"][0]))
            interv.append(bool(info["safety_intervened"]))
        if not (trunc and not term):
            continue
        traj = np.asarray(env.unwrapped.trajectory)
        for k in WINDOWS:
            disp = float(np.linalg.norm(traj[-1] - traj[-k - 1]))
            if disp >= 0.10 * k / 100:
                cls = "moving"
            elif np.mean(interv[-k:]) >= 0.5:
                cls = "filter-blocked"
            elif np.mean(np.asarray(v_nom[-k:]) <= 1e-3) >= 0.5:
                cls = "self-stop"
            else:
                cls = "other stall"
            rows.append({"variant": variant, "train_seed": seed, "episode_seed": ep_seed, "window": k, "class": cls})
    return rows


def main() -> int:
    tasks = [(v, s) for v in ("filtered", "unfiltered") for s in range(5)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = [r for part in pool.map(_job, tasks) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "research" / "audit_outputs" / "taxonomy_window_sensitivity_expert.csv", index=False)
    print(df.groupby(["variant", "window", "class"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
