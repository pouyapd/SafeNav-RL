"""
feasibility.py

Geometric oracle for a scenario (start, goal, obstacles): is the goal region
reachable while keeping a given clearance, and how long is the shortest such
path?

The robot can turn on the spot (v = 0, ω ≠ 0), so any path through the free
space is executable by the unicycle; reachability is therefore a question of
connectivity of the free space. The free space is rasterised on a grid
(default 5 cm) and searched with Dijkstra on 8-connected cells. Grid paths
over-estimate the true shortest path by at most about 8 %, and narrow passages
close to the resolution may be misjudged; both effects are small at 5 cm.

Used for two things:
  * to separate timeouts in blocked scenarios from timeouts the policy could
    have avoided (failure analysis);
  * as the reference length l in SPL instead of the straight-line distance.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


@dataclass
class FeasibilityResult:
    reachable: bool
    shortest_path: float       # metres (inf if unreachable)


def free_space_mask(
    obstacles: Sequence[Tuple[float, float, float]],
    workspace_size: float,
    robot_radius: float,
    margin: float = 0.0,
    wall_margin: float = 0.0,
    resolution: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Boolean grid of robot-centre positions that keep `margin` from every
    obstacle surface and `wall_margin` from the walls.

    Returns:
        (free, centres) with free of shape (n, n) and centres the 1-D cell
        centre coordinates (same for x and y).
    """
    n = int(np.ceil(workspace_size / resolution))
    centres = (np.arange(n) + 0.5) * resolution
    xx, yy = np.meshgrid(centres, centres, indexing="xy")   # yy[i, j] = row i, xx = column j
    inset = robot_radius + wall_margin
    free = (xx >= inset) & (xx <= workspace_size - inset) & (yy >= inset) & (yy <= workspace_size - inset)
    for (ox, oy, r) in obstacles:
        free &= np.hypot(xx - ox, yy - oy) >= r + robot_radius + margin
    return free, centres


def _cell(point: Sequence[float], resolution: float, n: int) -> Tuple[int, int]:
    j = int(np.clip(np.floor(point[0] / resolution), 0, n - 1))
    i = int(np.clip(np.floor(point[1] / resolution), 0, n - 1))
    return i, j


def check_feasibility(
    start: Sequence[float],
    goal: Sequence[float],
    obstacles: Sequence[Tuple[float, float, float]],
    workspace_size: float,
    robot_radius: float,
    goal_radius: float,
    margin: float = 0.0,
    wall_margin: float = 0.0,
    resolution: float = 0.05,
) -> FeasibilityResult:
    """Reachability of the goal region and shortest path length (see module docstring)."""
    free, centres = free_space_mask(obstacles, workspace_size, robot_radius, margin, wall_margin, resolution)
    n = len(centres)
    si, sj = _cell(start, resolution, n)
    if not free[si, sj]:
        return FeasibilityResult(False, float("inf"))

    idx = -np.ones((n, n), dtype=np.int64)
    flat = np.flatnonzero(free.ravel())
    idx.ravel()[flat] = np.arange(len(flat))

    rows, cols, weights = [], [], []
    for di, dj, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        src = (slice(0, n - di), slice(max(0, -dj), n - max(0, dj)))
        dst = (slice(di, n), slice(max(0, dj), n - max(0, -dj)))
        both = free[src] & free[dst]
        rows.append(idx[src][both])
        cols.append(idx[dst][both])
        weights.append(np.full(int(both.sum()), w * resolution))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    graph = sparse.coo_matrix((weights, (rows, cols)), shape=(len(flat), len(flat))).tocsr()

    dist = dijkstra(graph, directed=False, indices=int(idx[si, sj]))
    xx, yy = np.meshgrid(centres, centres, indexing="xy")
    in_goal = free & (np.hypot(xx - goal[0], yy - goal[1]) < goal_radius)
    if not np.any(in_goal):
        return FeasibilityResult(False, float("inf"))
    best = float(np.min(dist[idx[in_goal]]))
    return FeasibilityResult(bool(np.isfinite(best)), best)
