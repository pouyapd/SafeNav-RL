"""
baselines.py

Non-learned controllers used as reference points in the benchmark. They read
the same observation vector as the PPO agent (no privileged map access):

  GoToGoalPolicy       — turns toward the goal and drives; ignores obstacles.
  PotentialFieldPolicy — classic artificial potential field: attraction to the
                         goal plus repulsion from range-sensor hits.
  RandomPolicy         — uniform random actions (a floor for comparison).

Observation layout (see NavigationEnv):
  [x_n, y_n, sin θ, cos θ, d_goal_n, angle_error_n, risk, range_1/max_range, ...]
"""

from typing import Any, Dict, Optional

import numpy as np

ANGLE_ERROR_IDX = 5
FIRST_RANGE_IDX = 7


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class GoToGoalPolicy:
    """Proportional heading controller toward the goal (no obstacle handling)."""

    def __init__(self, k_omega: float = 2.0, v_max: float = 1.0, name: str = "GoToGoal"):
        self.k_omega = k_omega
        self.v_max = v_max
        self.name = name

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        err = float(obs[ANGLE_ERROR_IDX]) * np.pi
        omega = float(np.clip(self.k_omega * err, -1.0, 1.0))
        v = self.v_max * max(0.0, np.cos(err))
        return np.array([v, omega])


class PotentialFieldPolicy:
    """
    Artificial potential field (Khatib, 1986) on range-sensor readings.

    The desired direction (in the robot frame) is a unit attraction toward the
    goal plus, for every beam whose clearance d is below `influence` metres,
    a repulsion k_rep * (1/d - 1/influence) / d^2 pointing away from the hit.
    The clearance d is the reading minus the robot radius minus `margin`
    (default 0; a margin-aware field was tried and got stuck in local minima
    more often). The robot turns toward the desired
    direction and drives at a speed that drops with the heading error.
    Known failure mode: local minima (the robot stalls or oscillates).
    """

    def __init__(
        self,
        beam_offsets: np.ndarray,
        max_range: float,
        robot_radius: float = 0.3,
        margin: float = 0.0,
        influence: float = 1.2,
        k_rep: float = 0.15,
        k_omega: float = 2.0,
        v_max: float = 1.0,
        name: str = "PotentialField",
    ):
        self.beam_offsets = np.asarray(beam_offsets, dtype=float)
        self.max_range = float(max_range)
        self.robot_radius = float(robot_radius)
        self.margin = float(margin)
        self.influence = float(influence)
        self.k_rep = float(k_rep)
        self.k_omega = float(k_omega)
        self.v_max = float(v_max)
        self.name = name

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "PotentialFieldPolicy":
        sensor = config.get("sensor", {}) or {}
        n = int(sensor.get("num_beams", 0))
        if n == 0:
            raise ValueError("PotentialFieldPolicy needs range readings (sensor.num_beams > 0)")
        offsets = np.linspace(-np.pi, np.pi, n, endpoint=False)
        return cls(
            beam_offsets=offsets,
            max_range=float(sensor.get("max_range", 3.0)),
            robot_radius=float(config.get("env", {}).get("robot_radius", 0.3)),
            **kwargs,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def desired_direction(self, obs: np.ndarray) -> np.ndarray:
        """Resultant force vector in the robot frame (x forward, y left)."""
        err = float(obs[ANGLE_ERROR_IDX]) * np.pi
        force = np.array([np.cos(err), np.sin(err)])
        ranges = np.asarray(obs[FIRST_RANGE_IDX:FIRST_RANGE_IDX + len(self.beam_offsets)], dtype=float)
        ranges = ranges * self.max_range
        d = np.maximum(ranges - self.robot_radius - self.margin, 0.05)
        active = (ranges - self.robot_radius - self.margin) < self.influence
        if np.any(active):
            mag = self.k_rep * (1.0 / d[active] - 1.0 / self.influence) / d[active] ** 2
            ang = self.beam_offsets[active]
            force -= np.array([np.sum(mag * np.cos(ang)), np.sum(mag * np.sin(ang))])
        return force

    def act(self, obs: np.ndarray) -> np.ndarray:
        force = self.desired_direction(obs)
        heading_err = float(np.arctan2(force[1], force[0]))
        omega = float(np.clip(self.k_omega * heading_err, -1.0, 1.0))
        v = self.v_max * max(0.0, np.cos(heading_err))
        return np.array([v, omega])


class RandomPolicy:
    """Uniformly random actions in [0, 1] x [-1, 1]."""

    def __init__(self, seed: Optional[int] = None, name: str = "Random"):
        self._rng = np.random.default_rng(seed)
        self.name = name

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self._rng.uniform(0.0, 1.0), self._rng.uniform(-1.0, 1.0)])


class DynamicWindowPolicy:
    """
    DWA-style sampling local planner (after Fox, Burgard & Thrun, 1997) on range readings.

    The simulated robot has no inertia (any (v, ω) in the action box is reachable in one
    step), so the "dynamic window" is the whole action box. Each candidate (v, ω) on a
    grid is rolled out with the environment's own forward-Euler unicycle model for
    `horizon` steps in the robot frame. Obstacles are the points where the 16 beams hit
    something (no map knowledge). A rollout is admissible if its smallest distance to any
    hit point, minus the robot radius, stays above `clearance_min`. Among admissible
    rollouts the planner maximises
        w_progress * (distance gained toward the goal) / (v_max * horizon * dt)
      + w_heading * (1 - |heading error to the goal at the end| / π)
      + w_clearance * min(clearance, 1 m)
      + w_speed * v / v_max.
    If no rollout is admissible, it turns in place toward the side with more clearance.
    Weights are tuned on validation scenarios only (research/experiments/round2/).
    Known limits: sparse beams can miss obstacles between beams; it plans 1.5 s ahead only.
    """

    def __init__(
        self,
        beam_offsets: np.ndarray,
        max_range: float,
        workspace_size: float = 10.0,
        dt: float = 0.1,
        robot_radius: float = 0.3,
        horizon: int = 15,
        n_v: int = 6,
        n_omega: int = 11,
        clearance_min: float = 0.1,
        w_progress: float = 1.0,
        w_heading: float = 0.5,
        w_clearance: float = 0.5,
        w_speed: float = 0.1,
        v_max: float = 1.0,
        omega_max: float = 1.0,
        name: str = "DWA",
    ):
        self.beam_offsets = np.asarray(beam_offsets, dtype=float)
        self.max_range = float(max_range)
        self.goal_scale = float(workspace_size) * np.sqrt(2)
        self.dt = float(dt)
        self.robot_radius = float(robot_radius)
        self.horizon = int(horizon)
        self.clearance_min = float(clearance_min)
        self.w = (float(w_progress), float(w_heading), float(w_clearance), float(w_speed))
        self.v_max = float(v_max)
        self.name = name
        v, w = np.meshgrid(np.linspace(0.0, v_max, n_v), np.linspace(-omega_max, omega_max, n_omega))
        self.cand_v, self.cand_w = v.ravel(), w.ravel()
        # Rollouts in the robot frame do not depend on the observation: precompute them.
        n = len(self.cand_v)
        pos = np.zeros((n, self.horizon + 1, 2))
        th = np.zeros(n)
        for k in range(self.horizon):
            pos[:, k + 1, 0] = pos[:, k, 0] + self.cand_v * self.dt * np.cos(th)
            pos[:, k + 1, 1] = pos[:, k, 1] + self.cand_v * self.dt * np.sin(th)
            th = th + self.cand_w * self.dt
        self.rollout_pos, self.rollout_end_heading = pos, th

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "DynamicWindowPolicy":
        sensor = config.get("sensor", {}) or {}
        n = int(sensor.get("num_beams", 0))
        if n == 0:
            raise ValueError("DynamicWindowPolicy needs range readings (sensor.num_beams > 0)")
        env = config.get("env", {})
        return cls(
            beam_offsets=np.linspace(-np.pi, np.pi, n, endpoint=False),
            max_range=float(sensor.get("max_range", 3.0)),
            workspace_size=float(env.get("workspace_size", 10.0)),
            dt=float(env.get("dt", 0.1)),
            robot_radius=float(env.get("robot_radius", 0.3)),
            **kwargs,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        err = float(obs[ANGLE_ERROR_IDX]) * np.pi
        d_goal = float(obs[ANGLE_ERROR_IDX - 1]) * self.goal_scale
        goal = np.array([d_goal * np.cos(err), d_goal * np.sin(err)])
        ranges = np.asarray(obs[FIRST_RANGE_IDX:FIRST_RANGE_IDX + len(self.beam_offsets)], dtype=float) * self.max_range
        hit = ranges < 0.999 * self.max_range
        pts = np.stack([ranges[hit] * np.cos(self.beam_offsets[hit]), ranges[hit] * np.sin(self.beam_offsets[hit])], 1)

        if len(pts):
            diff = self.rollout_pos[:, :, None, :] - pts[None, None, :, :]
            clearance = np.sqrt((diff ** 2).sum(-1)).min(axis=(1, 2)) - self.robot_radius
        else:
            clearance = np.full(len(self.cand_v), self.max_range)
        end = self.rollout_pos[:, -1, :]
        progress = (d_goal - np.linalg.norm(goal[None, :] - end, axis=1)) / (self.v_max * self.horizon * self.dt)
        to_goal = np.arctan2(goal[1] - end[:, 1], goal[0] - end[:, 0])
        head_err = np.abs((to_goal - self.rollout_end_heading + np.pi) % (2 * np.pi) - np.pi)
        wp, wh, wc, wv = self.w
        score = (wp * progress + wh * (1.0 - head_err / np.pi) + wc * np.minimum(clearance, 1.0)
                 + wv * self.cand_v / self.v_max)
        ok = clearance > self.clearance_min
        if np.any(ok):
            i = int(np.argmax(np.where(ok, score, -np.inf)))
            return np.array([self.cand_v[i], self.cand_w[i]])
        # Nothing admissible: turn in place toward the side with more free space.
        left = ranges[(self.beam_offsets > 0) & (self.beam_offsets < np.pi)].mean() if len(ranges) else 0.0
        right = ranges[(self.beam_offsets < 0) & (self.beam_offsets > -np.pi)].mean() if len(ranges) else 0.0
        return np.array([0.0, 1.0 if left >= right else -1.0])
