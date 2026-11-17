r"""
model.py — vectorised model of the filtered robot, used for planning (round 3; opt-in).

It predicts what the environment would execute for N candidate control sequences at once:
the CBF speed filter of safenav/safety/cbf_filter.py, followed by the forward-Euler unicycle
update and the goal test of NavigationEnv.step. The arithmetic follows those two files
operation by operation, so a rollout reproduces SafetyFilterWrapper + NavigationEnv up to
floating-point rounding (tests/test_recovery.py checks this to 1e-9).

The model uses the same information as the filter: the exact robot state and the exact
obstacle geometry. It does not use the observation, so sensor noise does not enter.
Nothing here can execute an action: the environment's own filter still checks every
action that a recovery controller proposes.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_EPS = 1e-9


@dataclass
class Rollout:
    pos: np.ndarray        # (N, H+1, 2) positions, including the start
    theta: np.ndarray      # (N, H+1)
    v_exec: np.ndarray     # (N, H) executed speed (0 after the goal is reached)
    limited: np.ndarray    # (N, H) bool, the filter lowered the commanded speed
    reached_at: np.ndarray # (N,) first step index (1..H) at which the goal is reached, or -1
    h_min: np.ndarray      # (N,) smallest barrier value along the rollout (>= 0 if safe)


class FilteredUnicycleModel:
    """Exact, vectorised copy of CBFSafetyFilter + NavigationEnv dynamics (see module docstring)."""

    def __init__(self, obstacles: Sequence, goal, *, dt=0.1, robot_radius=0.3, safe_distance=0.3, gamma=0.5,
                 workspace_size=10.0, wall_margin=0.1, v_max=1.0, omega_max=1.0, goal_radius=0.5):
        self.ox = np.array([o.x for o in obstacles], dtype=float)
        self.oy = np.array([o.y for o in obstacles], dtype=float)
        self.orad = np.array([o.radius for o in obstacles], dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.dt, self.r, self.d, self.gamma = float(dt), float(robot_radius), float(safe_distance), float(gamma)
        self.W, self.wall_margin = float(workspace_size), float(wall_margin)
        self.v_max, self.omega_max, self.goal_radius = float(v_max), float(omega_max), float(goal_radius)

    @classmethod
    def from_env(cls, env) -> "FilteredUnicycleModel":
        """Build the model from a (wrapped) environment that uses the default CBF filter parameters."""
        base = env.unwrapped
        flt = getattr(env, "safety_filter", None)
        kw = {}
        if flt is not None:
            kw = dict(dt=flt.dt, robot_radius=flt.robot_radius, safe_distance=flt.safe_distance, gamma=flt.gamma,
                      workspace_size=flt.workspace_size, wall_margin=flt.wall_margin, v_max=flt.v_max,
                      omega_max=flt.omega_max)
        else:
            kw = dict(dt=base.dt, robot_radius=base.robot_radius)
        return cls(base.obstacle_map.obstacles, base.goal_pos, goal_radius=base.goal_radius, **kw)

    def barriers(self, px, py, theta):
        """Barrier values and closing rates, shape (N, M+4), for positions (N,) and headings (N,)."""
        ux, uy = np.cos(theta), np.sin(theta)
        dx = self.ox[None, :] - px[:, None]
        dy = self.oy[None, :] - py[:, None]
        dist = np.hypot(dx, dy)
        h_obs = dist - self.orad[None, :] - self.r - self.d
        safe_dist = np.where(dist > _EPS, dist, 1.0)
        a_obs = np.where(dist > _EPS, (ux[:, None] * dx + uy[:, None] * dy) / safe_dist, 1.0)
        inset = self.r + self.wall_margin
        h_w = np.stack([px - inset, self.W - px - inset, py - inset, self.W - py - inset], axis=1)
        a_w = np.stack([-ux, ux, -uy, uy], axis=1)
        return np.concatenate([h_obs, h_w], axis=1), np.concatenate([a_obs, a_w], axis=1)

    def speed_limit(self, px, py, theta):
        h, a = self.barriers(px, py, theta)
        closing = a > _EPS
        lim = np.where(closing, self.gamma * h / np.where(closing, a, 1.0), np.inf)
        lim = lim.min(axis=1)
        return np.where(np.isfinite(lim), np.clip(lim, 0.0, self.v_max), self.v_max)

    def rollout(self, pos, theta, controls) -> Rollout:
        """controls: (N, H, 2) nominal [v, omega]; pos: (2,); theta: float."""
        controls = np.asarray(controls, dtype=float)
        n, horizon = controls.shape[:2]
        px = np.full(n, float(pos[0]))
        py = np.full(n, float(pos[1]))
        th = np.full(n, float(theta))
        P = np.empty((n, horizon + 1, 2))
        T = np.empty((n, horizon + 1))
        V = np.zeros((n, horizon))
        L = np.zeros((n, horizon), dtype=bool)
        P[:, 0, 0], P[:, 0, 1], T[:, 0] = px, py, th
        reached = np.full(n, -1)
        h_min = np.full(n, np.inf)
        h0, _ = self.barriers(px, py, th)
        h_min = np.minimum(h_min, h0.min(axis=1))
        for t in range(horizon):
            active = reached < 0
            v_nom = np.clip(controls[:, t, 0], 0.0, self.v_max)
            om = np.clip(controls[:, t, 1], -self.omega_max, self.omega_max)
            lim = self.speed_limit(px, py, th)
            v = np.minimum(v_nom, lim)
            L[:, t] = active & (v < v_nom - 1e-9)
            v = np.where(active, np.clip(v, 0.0, 1.0), 0.0)
            om = np.where(active, np.clip(om, -1.0, 1.0), 0.0)
            px = px + v * np.cos(th) * self.dt
            py = py + v * np.sin(th) * self.dt
            th = th + om * self.dt
            th = (th + np.pi) % (2 * np.pi) - np.pi
            V[:, t] = v
            P[:, t + 1, 0], P[:, t + 1, 1], T[:, t + 1] = px, py, th
            h, _ = self.barriers(px, py, th)
            h_min = np.minimum(h_min, h.min(axis=1))
            dg = np.hypot(self.goal[0] - px, self.goal[1] - py)
            reached = np.where(active & (dg < self.goal_radius), t + 1, reached)
        return Rollout(P, T, V, L, reached, h_min)
