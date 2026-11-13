r"""
costmap.py — cost-to-go fields for the recovery planner (round 3; opt-in).

GeodesicField: shortest-path distance to the goal region through the filter's safe set
(the free space that keeps `safe_distance` from every obstacle surface and `wall_margin`
from the walls). It uses the same 5 cm grid and 8-connected Dijkstra as the feasibility
oracle (safenav/evaluation/feasibility.py), with all goal cells as sources. Positions
that fall in a blocked cell (possible at the boundary because of the grid) take the value
of the nearest free cell plus the distance to it. Unreachable positions get `inf`.
This field uses the full obstacle map: it is global information, like the map the filter
already uses. EuclideanField is the local alternative (straight-line distance).
"""

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from safenav.evaluation.feasibility import free_space_mask


class EuclideanField:
    name = "euclidean"

    def __init__(self, goal):
        self.goal = np.asarray(goal, dtype=float)

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.hypot(xy[..., 0] - self.goal[0], xy[..., 1] - self.goal[1])


class GeodesicField:
    name = "geodesic"

    def __init__(self, obstacles, goal, *, workspace_size=10.0, robot_radius=0.3, safe_distance=0.3,
                 wall_margin=0.1, goal_radius=0.5, resolution=0.05):
        self.goal = np.asarray(goal, dtype=float)
        self.res = float(resolution)
        obs = [(o.x, o.y, o.radius) for o in obstacles]
        free, centres = free_space_mask(obs, workspace_size, robot_radius, safe_distance, wall_margin, resolution)
        n = len(centres)
        self.n, self.centres = n, centres
        idx = -np.ones((n, n), dtype=np.int64)
        flat = np.flatnonzero(free.ravel())
        idx.ravel()[flat] = np.arange(len(flat))
        rows, cols, w = [], [], []
        for di, dj, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
            src = (slice(0, n - di), slice(max(0, -dj), n - max(0, dj)))
            dst = (slice(di, n), slice(max(0, dj), n - max(0, -dj)))
            both = free[src] & free[dst]
            rows.append(idx[src][both])
            cols.append(idx[dst][both])
            w.append(np.full(int(both.sum()), step * resolution))
        graph = sparse.coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(len(flat), len(flat))).tocsr()
        xx, yy = np.meshgrid(centres, centres, indexing="xy")
        in_goal = free & (np.hypot(xx - self.goal[0], yy - self.goal[1]) < goal_radius)
        field = np.full((n, n), np.inf)
        if np.any(in_goal):
            d = dijkstra(graph, directed=False, indices=idx[in_goal], min_only=True)
            field.ravel()[flat] = d
        # blocked cells: value of the nearest free cell + distance to it
        if np.any(free):
            dist, (ii, jj) = ndimage.distance_transform_edt(~free, return_indices=True)
            filled = field[ii, jj] + dist * resolution
            field = np.where(free, field, filled)
        self.free, self.field = free, field

    def cell(self, xy):
        xy = np.asarray(xy, dtype=float)
        j = np.clip(np.floor(xy[..., 0] / self.res).astype(int), 0, self.n - 1)
        i = np.clip(np.floor(xy[..., 1] / self.res).astype(int), 0, self.n - 1)
        return i, j

    def __call__(self, xy):
        i, j = self.cell(xy)
        return self.field[i, j]

    def path_direction(self, xy, lookahead=0.5):
        """Unit vector of the initial direction of the shortest path from xy (steepest descent), or None."""
        p = np.asarray(xy, dtype=float).copy()
        start = p.copy()
        for _ in range(int(4 * lookahead / self.res) + 4):
            i, j = self.cell(p)
            best, best_v = None, self.field[i, j]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < self.n and 0 <= b < self.n and self.field[a, b] < best_v:
                        best, best_v = (a, b), self.field[a, b]
            if best is None:
                break
            p = np.array([self.centres[best[1]], self.centres[best[0]]])
            if np.hypot(*(p - start)) >= lookahead:
                break
        d = p - start
        nrm = float(np.hypot(*d))
        return d / nrm if nrm > 1e-9 else None
