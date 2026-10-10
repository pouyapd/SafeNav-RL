"""
visualize_trajectory.py — plot or animate one seeded test episode.

Usage:
    python scripts/visualize_trajectory.py --checkpoint runs/default/checkpoints/best.pt
    python scripts/visualize_trajectory.py --baseline potential_field --stage expert --seed 2200003
    python scripts/visualize_trajectory.py --checkpoint ... --animate --save episode.gif
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy  # noqa: E402
from safenav.agents.policy import PPOPolicy, load_checkpoint  # noqa: E402
from safenav.config import get_stage, load_config  # noqa: E402
from safenav.evaluation.runner import run_episode  # noqa: E402
from safenav.factory import make_env  # noqa: E402
from safenav.viz.plots import animate_episode, plot_trajectory  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a single seeded evaluation episode.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--baseline", choices=["gotogoal", "potential_field"])
    p.add_argument("--config", default=None)
    p.add_argument("--stage", default="expert")
    p.add_argument("--seed", type=int, default=2_200_000, help="episode seed (default: first expert test seed)")
    p.add_argument("--safety-filter", choices=["on", "off"], default="on")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--save", default=None, help=".png for a still image; .gif/.mp4 with --animate")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint:
        config = load_config(args.config) if args.config else load_checkpoint(args.checkpoint)["config"]
        policy = PPOPolicy.from_checkpoint(args.checkpoint)
    else:
        config = load_config(args.config)
        policy = GoToGoalPolicy() if args.baseline == "gotogoal" else PotentialFieldPolicy.from_config(config)
    env = make_env(config, task=get_stage(config, args.stage), safety_filter=args.safety_filter == "on")
    result = run_episode(env, policy, args.seed, stage=args.stage, keep_trajectory=True)
    print(f"{result.method}: {result.outcome} after {result.steps} steps, path {result.path_length:.2f} m, "
          f"min clearance {result.min_clearance:.2f} m, filter interventions {result.interventions}")
    margin = config["safety"]["safe_distance"]
    if args.animate:
        animate_episode(result, margin=margin, save_path=args.save or "episode.gif")
    else:
        plot_trajectory(result, margin=margin, save_path=args.save or "episode.png")


if __name__ == "__main__":
    main()
