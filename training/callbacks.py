"""
callbacks.py

Training callbacks for logging, checkpointing, and early stopping.

Using callbacks keeps the main training loop clean and modular.
"""

import os
import json
import torch
import numpy as np
from collections import deque
from typing import Dict, Any, Optional


class TrainingLogger:
    """
    Logs episode metrics and training stats to console and JSON.
    Also supports TensorBoard if available.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_episodes = 0
        self.total_timesteps = 0

        self._history = []

        # TensorBoard (optional)
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"[Logger] TensorBoard logging to: {log_dir}")
        except ImportError:
            print("[Logger] TensorBoard not available. Install tensorboard for live plots.")

    def log_episode(self, reward: float, length: int, info: Dict[str, Any]):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes += 1

        entry = {
            "episode": self.total_episodes,
            "timestep": self.total_timesteps,
            "reward": reward,
            "length": length,
            "goal_reached": info.get("goal_reached", False),
            "collision": info.get("collision", False),
        }
        self._history.append(entry)

        if self._tb_writer:
            self._tb_writer.add_scalar("episode/reward", reward, self.total_timesteps)
            self._tb_writer.add_scalar("episode/length", length, self.total_timesteps)

    def log_update(self, update_info: Dict[str, float]):
        if self._tb_writer:
            for key, value in update_info.items():
                self._tb_writer.add_scalar(f"train/{key}", value, self.total_timesteps)

    def print_summary(self, timestep: int, curriculum_info: Optional[str] = None):
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        success_rate = np.mean([
            e["goal_reached"] for e in self._history[-100:]
        ]) if self._history else 0.0

        print(
            f"[Step {timestep:>8}] "
            f"Episodes: {self.total_episodes:>5} | "
            f"Reward(100): {mean_reward:>7.2f} | "
            f"Length(100): {mean_length:>6.1f} | "
            f"Success: {success_rate:.2%}"
            + (f" | {curriculum_info}" if curriculum_info else "")
        )

    def save_history(self):
        path = os.path.join(self.log_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        self.save_history()


class CheckpointCallback:
    """
    Saves model checkpoints periodically and tracks the best model
    based on mean episode reward.
    """

    def __init__(self, checkpoint_dir: str, save_every: int):
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.best_reward = -float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._last_save = 0

    def should_save(self, timestep: int) -> bool:
        return timestep - self._last_save >= self.save_every

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        timestep: int,
        mean_reward: float,
        config: Dict,
        tag: str = "",
    ):
        self._last_save = timestep

        checkpoint = {
            "timestep": timestep,
            "mean_reward": mean_reward,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        }

        filename = f"checkpoint_{timestep:08d}{('_' + tag) if tag else ''}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

        # Track best model
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] New best model saved (reward={mean_reward:.2f})")

        print(f"[Checkpoint] Saved: {path}")
        return path
