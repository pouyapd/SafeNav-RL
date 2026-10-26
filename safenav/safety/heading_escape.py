r"""
heading_escape.py

Opt-in, test-time heading modification for the CBF speed filter (experiment E2;
not used anywhere by default).

Whenever the filter reduces the commanded speed, the turn rate is shifted by
kappa * omega_max in the direction that rotates the heading *away* from the
binding barrier (towards its tangent):

    phi   = signed angle from n_k to u(theta)      n_k: direction of the binding barrier
                                                   (towards the obstacle centre, or towards the wall)
    s     = sign(phi); if phi == 0: the side of the goal
    omega = clip(omega_nom + kappa * s * omega_max, -omega_max, omega_max)

The speed is filtered exactly as in CBFSafetyFilter. Because the one-step barrier
condition does not depend on the turn rate (p_{t+1} depends only on p_t, theta_t
and v_t), the forward-invariance guarantee of the filter is unchanged for any
kappa (manuscript, Proposition 1). Nothing is claimed about liveness.
"""

from typing import Any, Dict

import numpy as np

from safenav.safety.cbf_filter import _EPS, CBFSafetyFilter, SafetyFilterWrapper

# Directions towards the four walls, in the order used by CBFSafetyFilter.barriers().
_WALL_DIRECTIONS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


def binding_direction(flt: CBFSafetyFilter, robot_pos, robot_theta, obstacles):
    """Unit vector towards the barrier that sets the speed limit (None if no barrier is closing)."""
    h, a = flt.barriers(robot_pos, robot_theta, obstacles)
    closing = a > _EPS
    if not np.any(closing):
        return None
    limits = np.full(h.shape, np.inf)
    limits[closing] = flt.gamma * h[closing] / a[closing]
    k = int(np.argmin(limits))
    if k < len(obstacles):
        d = np.array([obstacles[k].x - robot_pos[0], obstacles[k].y - robot_pos[1]], dtype=float)
        norm = float(np.linalg.norm(d))
        return d / norm if norm > _EPS else None
    return _WALL_DIRECTIONS[k - len(obstacles)]


def escape_turn_sign(n: np.ndarray, heading: np.ndarray, to_goal: np.ndarray) -> float:
    """+1 (counter-clockwise) or -1: the turn that increases the angle between heading and n."""
    cross = n[0] * heading[1] - n[1] * heading[0]
    if abs(cross) > 1e-9:
        return float(np.sign(cross))
    goal_side = n[0] * to_goal[1] - n[1] * to_goal[0]
    return float(np.sign(goal_side)) if abs(goal_side) > 1e-9 else 1.0


class HeadingEscapeWrapper(SafetyFilterWrapper):
    """SafetyFilterWrapper plus the heading escape above. kappa = 0 behaves exactly like the parent."""

    def __init__(self, env, safety_filter: CBFSafetyFilter, kappa: float = 0.0):
        super().__init__(env, safety_filter)
        if not 0.0 <= kappa <= 2.0:
            raise ValueError("kappa must be in [0, 2]")
        self.kappa = float(kappa)

    def step(self, action):
        base = self.env.unwrapped
        nominal = np.asarray(action, dtype=np.float64)
        safe, finfo = self.safety_filter.filter(nominal, base.robot_pos, base.robot_theta,
                                                base.obstacle_map.obstacles)
        escaped = False
        if self.kappa > 0 and finfo["intervened"]:
            n = binding_direction(self.safety_filter, base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
            if n is not None:
                heading = np.array([np.cos(base.robot_theta), np.sin(base.robot_theta)])
                s = escape_turn_sign(n, heading, base.goal_pos - base.robot_pos)
                w_max = self.safety_filter.omega_max
                safe = safe.copy()
                safe[1] = float(np.clip(safe[1] + self.kappa * s * w_max, -w_max, w_max))
                escaped = True
        obs, reward, terminated, truncated, info = self.env.step(safe)
        info["nominal_action"] = np.clip(nominal, base.action_space.low, base.action_space.high)
        info["safe_action"] = safe
        info["safety_intervened"] = finfo["intervened"]
        info["speed_limit"] = finfo["v_limit"]
        info["heading_escape"] = escaped
        return obs, reward, terminated, truncated, info


def make_escape_env(config: Dict[str, Any], task: Dict[str, Any], kappa: float):
    """Evaluation environment with the CBF filter and the heading escape (kappa = 0: plain filter)."""
    from safenav.env.navigation_env import NavigationEnv
    flt = CBFSafetyFilter.from_config(config)
    flt.enabled = True
    return HeadingEscapeWrapper(NavigationEnv(config, task=task), flt, kappa=kappa)
