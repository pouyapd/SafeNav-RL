"""
train.py — SafeNav-RL training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default_config.yaml
    python scripts/train.py --config configs/default_config.yaml --timesteps 500000
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from training.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SafeNav-RL PPO agent for safe robot navigation."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to YAML config file (default: configs/default_config.yaml)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps from config"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--no-curriculum", action="store_true",
        help="Disable curriculum learning (train on expert difficulty)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading config: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.timesteps is not None:
        config["ppo"]["total_timesteps"] = args.timesteps
        print(f"  → Overriding total_timesteps: {args.timesteps:,}")

    if args.seed is not None:
        config["training"]["seed"] = args.seed
        print(f"  → Overriding seed: {args.seed}")

    if args.no_curriculum:
        config["curriculum"]["enabled"] = False
        print("  → Curriculum learning DISABLED")

    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
