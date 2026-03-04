"""
visualizer.py

Trajectory visualization and animation for SafeNav-RL.

Produces:
  1. Static trajectory plot (robot path, obstacles, goal)
  2. Animated episode replay (MP4 or GIF)
  3. Training curves (reward, success rate over time)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from typing import List, Optional
import os

from evaluation.evaluator import EpisodeResult, EvaluationSummary


def plot_trajectory(
    result: EpisodeResult,
    title: str = "Navigation Trajectory",
    workspace_size: float = 10.0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a single episode trajectory with obstacles and goal.
    
    Color coding:
      - Blue line: robot path
      - Green marker: goal position
      - Black circles: obstacles
      - Red dot: robot start
      - Orange dot: robot final position
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(0, workspace_size)
    ax.set_ylim(0, workspace_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Draw obstacles
    for obs in result.obstacles:
        circle = patches.Circle(
            (obs.x, obs.y), obs.radius,
            linewidth=1.5, edgecolor="black", facecolor="gray", alpha=0.5,
        )
        ax.add_patch(circle)

    # Draw trajectory
    traj = np.array(result.trajectory)
    if len(traj) > 1:
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="Robot path", alpha=0.8)

    # Start and end
    ax.plot(*traj[0], "ro", markersize=10, label="Start", zorder=5)
    ax.plot(*traj[-1], "o", color="orange", markersize=8, label="End", zorder=5)

    # Goal
    goal_circle = patches.Circle(
        result.goal_pos, 0.5,
        linewidth=2, edgecolor="green", facecolor="lime", alpha=0.6,
    )
    ax.add_patch(goal_circle)
    ax.plot(*result.goal_pos, "g*", markersize=14, label="Goal", zorder=5)

    # Status indicator
    status = "✓ SUCCESS" if result.success else ("✗ COLLISION" if result.collision else "⏱ TIMEOUT")
    color = "green" if result.success else "red"
    ax.text(
        0.02, 0.98, status,
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        verticalalignment="top", color=color,
    )

    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def animate_episode(
    result: EpisodeResult,
    workspace_size: float = 10.0,
    save_path: Optional[str] = None,
    interval_ms: int = 50,
) -> animation.FuncAnimation:
    """
    Create an animated replay of an episode.
    
    Args:
        result: EpisodeResult from PolicyEvaluator
        save_path: Optional path (.mp4 or .gif)
        interval_ms: Animation frame interval in milliseconds
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, workspace_size)
    ax.set_ylim(0, workspace_size)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Static elements
    for obs in result.obstacles:
        circle = patches.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.5)
        ax.add_patch(circle)

    goal_circle = patches.Circle(result.goal_pos, 0.5, color="lime", alpha=0.6)
    ax.add_patch(goal_circle)
    ax.plot(*result.goal_pos, "g*", markersize=14)

    # Dynamic elements
    traj = np.array(result.trajectory)
    path_line, = ax.plot([], [], "b-", linewidth=1.5, alpha=0.7)
    robot_dot, = ax.plot([], [], "ro", markersize=12)
    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=10, va="top")

    def init():
        path_line.set_data([], [])
        robot_dot.set_data([], [])
        return path_line, robot_dot, step_text

    def update(frame):
        path_line.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])
        robot_dot.set_data([traj[frame, 0]], [traj[frame, 1]])
        step_text.set_text(f"Step: {frame}/{len(traj)-1}")
        return path_line, robot_dot, step_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(traj),
        init_func=init, interval=interval_ms, blit=True
    )

    if save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=20)
        print(f"[Visualizer] Animation saved: {save_path}")

    return anim


def plot_training_curves(
    history_path: str,
    save_dir: Optional[str] = None,
):
    """
    Plot training reward and success rate from training_history.json.
    
    Args:
        history_path: Path to logs/training_history.json
        save_dir: Directory to save plots
    """
    import json
    import pandas as pd

    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    window = 50

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Reward
    axes[0].plot(df["timestep"], df["reward"], alpha=0.3, color="steelblue", label="Raw")
    axes[0].plot(
        df["timestep"],
        df["reward"].rolling(window, min_periods=1).mean(),
        color="steelblue", linewidth=2, label=f"Rolling mean ({window})"
    )
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Success rate
    success_rolling = df["goal_reached"].rolling(window, min_periods=1).mean()
    axes[1].plot(df["timestep"], success_rolling, color="green", linewidth=2)
    axes[1].axhline(y=0.7, color="red", linestyle="--", alpha=0.7, label="70% target")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Success Rate (rolling)")
    axes[1].set_title("Success Rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(path, dpi=150)
        print(f"[Visualizer] Training curves saved: {path}")
    plt.show()
    plt.close()
