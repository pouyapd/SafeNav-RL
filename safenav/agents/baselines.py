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
