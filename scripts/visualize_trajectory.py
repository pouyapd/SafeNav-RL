"""
visualize_trajectory.py — Visualize and animate a single episode.

Usage:
    python scripts/visualize_trajectory.py --checkpoint checkpoints/best_model.pt
    python scripts/visualize_trajectory.py --checkpoint checkpoints/best_model.pt --animate --save trajectory.gif
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from env.navigation_env import NavigationEnv
from models.actor_critic import ActorCritic
from models.safety_layer import SafetyLayer
from evaluation.evaluator import PolicyEvaluator
from evaluation.visualizer import plot_trajectory, animate_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a single evaluation episode.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--save", type=str, default=None, help="Save path (.png / .gif / .mp4)")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    if "curriculum" in config:
        config["curriculum"]["enabled"] = False

    device = torch.device("cpu")
    model = ActorCritic(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    safety_layer = SafetyLayer(config)
    env = NavigationEnv(config)
    evaluator = PolicyEvaluator(env, model, safety_layer, device)

    # Run a single episode
    summary = evaluator.evaluate(n_episodes=1)
    result = summary.results[0]

    outcome = "SUCCESS" if result.success else ("COLLISION" if result.collision else "TIMEOUT")
    print(f"\nEpisode: {outcome}")
    print(f"  Steps: {result.steps}")
    print(f"  Reward: {result.total_reward:.3f}")
    print(f"  Path length: {result.path_length:.2f} m")
    print(f"  Mean risk: {result.mean_risk:.4f}")

    if args.animate:
        save_path = args.save if args.save else None
        anim = animate_episode(
            result,
            workspace_size=config["env"]["workspace_size"],
            save_path=save_path,
        )
        import matplotlib.pyplot as plt
        plt.show()
    else:
        save_path = args.save if args.save and args.save.endswith(".png") else None
        plot_trajectory(
            result,
            title=f"Navigation Trajectory — {outcome}",
            workspace_size=config["env"]["workspace_size"],
            save_path=save_path,
            show=True,
        )


if __name__ == "__main__":
    main()
