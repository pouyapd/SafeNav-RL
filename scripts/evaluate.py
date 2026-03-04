"""
evaluate.py — SafeNav-RL evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --episodes 100
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from env.navigation_env import NavigationEnv
from models.actor_critic import ActorCritic
from models.safety_layer import SafetyLayer
from evaluation.evaluator import PolicyEvaluator
from evaluation.visualizer import plot_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SafeNav-RL policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: from checkpoint)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--plot", action="store_true", help="Plot example trajectories")
    parser.add_argument("--save-results", type=str, default=None, help="Save summary JSON to this path")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Disable curriculum during evaluation
    if "curriculum" in config:
        config["curriculum"]["enabled"] = False

    device = torch.device("cpu")

    # Load model
    model = ActorCritic(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded (trained for {checkpoint.get('timestep', '?'):,} steps)")

    safety_layer = SafetyLayer(config)
    env = NavigationEnv(config)

    evaluator = PolicyEvaluator(env, model, safety_layer, device)

    print(f"\nRunning evaluation: {args.episodes} episodes...")
    summary = evaluator.evaluate(n_episodes=args.episodes, deterministic=args.deterministic)
    summary.print()

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Results saved to: {args.save_results}")

    if args.plot and summary.results:
        # Plot a few example trajectories
        for i, result in enumerate(summary.results[:3]):
            outcome = "success" if result.success else ("collision" if result.collision else "timeout")
            plot_trajectory(
                result,
                title=f"Episode {i+1} — {outcome.upper()}",
                workspace_size=config["env"]["workspace_size"],
                save_path=f"trajectory_ep{i+1}.png",
                show=True,
            )


if __name__ == "__main__":
    main()
