"""
plots.py

Single-episode visualization for SafeNav-RL:
  1. Static trajectory plot (robot path, obstacles, safety margin, goal)
  2. Animated episode replay (GIF via Pillow, or MP4 if ffmpeg is installed)
"""

from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from safenav.evaluation.metrics import EpisodeResult
from safenav.viz.figures import _outcome_text, draw_path, draw_scenario
from safenav.viz.style import INK_2, apply_style, method_color


def plot_trajectory(
    result: EpisodeResult,
    title: Optional[str] = None,
    workspace_size: float = 10.0,
    margin: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot one episode (needs a result created with keep_trajectory=True)."""
    if result.trajectory is None:
        raise ValueError("EpisodeResult has no trajectory; evaluate with keep_trajectory=True")
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_scenario(ax, result.obstacles, result.goal, result.trajectory[0], workspace=workspace_size,
                  margin=margin if result.safety_filter else None)
    draw_path(ax, result.trajectory, method_color(result.method), result.outcome,
              interventions=result.interventions_mask)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{result.method}: {_outcome_text(result.outcome)} after {result.steps} steps")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[plots] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def animate_episode(
    result: EpisodeResult,
    workspace_size: float = 10.0,
    margin: Optional[float] = None,
    save_path: Optional[str] = None,
    fps: int = 20,
) -> animation.FuncAnimation:
    """Animated replay of one episode. `save_path` ending in .gif uses Pillow, .mp4 uses ffmpeg."""
    if result.trajectory is None:
        raise ValueError("EpisodeResult has no trajectory; evaluate with keep_trajectory=True")
    apply_style()
    traj = np.asarray(result.trajectory)
    headings = np.asarray(result.headings) if result.headings is not None else np.zeros(len(traj))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_scenario(ax, result.obstacles, result.goal, traj[0], workspace=workspace_size,
                  margin=margin if result.safety_filter else None)
    color = method_color(result.method)
    path_line, = ax.plot([], [], color=color, linewidth=2.0)
    robot = plt.Circle((traj[0, 0], traj[0, 1]), 0.3, color=color, alpha=0.35)
    ax.add_patch(robot)
    heading_line, = ax.plot([], [], color=color, linewidth=1.5)
    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top", color=INK_2)

    def update(frame):
        path_line.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])
        robot.center = (traj[frame, 0], traj[frame, 1])
        th = headings[frame]
        heading_line.set_data([traj[frame, 0], traj[frame, 0] + 0.45 * np.cos(th)],
                              [traj[frame, 1], traj[frame, 1] + 0.45 * np.sin(th)])
        step_text.set_text(f"step {frame}/{len(traj) - 1}")
        return path_line, robot, heading_line, step_text

    anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=1000 // fps, blit=True)
    if save_path:
        writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
        anim.save(save_path, writer=writer, fps=fps)
        print(f"[plots] animation saved: {save_path}")
    return anim
