r"""
cbf_filter.py

Discrete-time control-barrier-function (CBF) safety filter for the unicycle
robot in NavigationEnv. It limits the linear speed so that the robot can never
enter a safety margin around obstacles or leave the workspace.

Barrier functions (one per obstacle i and one per workspace wall):

    h_i(p) = ||p - c_i|| - (R_i + r + d_safe)          obstacle i (centre c_i, radius R_i)
    h_W(p) = p_x - (r + m),  W - p_x - (r + m),         walls (workspace [0, W]^2,
             p_y - (r + m),  W - p_y - (r + m)          wall margin m)

The safe set is S = {p : h(p) >= 0 for every barrier}.

Dynamics (exactly as implemented in NavigationEnv.step):

    p_{t+1} = p_t + v_t * dt * u(theta_t),   u(theta) = (cos theta, sin theta),   v_t in [0, v_max]

Filter. For each barrier let a_i be the rate at which it closes per unit speed
(a_i = u . n_i with n_i the unit vector from the robot to the obstacle centre;
a = -u_x, u_x, -u_y, u_y for the four walls). The filter imposes

    a_i * v <= gamma * h_i(p_t)        for every barrier with a_i > 0

and returns the largest admissible speed not above the policy's speed:

    v_safe = clip( min(v_nom, min_i gamma * h_i / a_i), 0, v_max ),   omega unchanged.

Because every constraint is an upper bound on the single variable v, this is
the exact solution of the quadratic program
    min_v (v - v_nom)^2   s.t.  a_i v <= gamma h_i,  0 <= v <= v_max.

Guarantee (forward invariance in the simulator). If h_i(p_t) >= 0 for all i and
gamma * dt <= 1, then h_i(p_{t+1}) >= (1 - gamma * dt) * h_i(p_t) >= 0.
Proof: the Euclidean norm is convex, so
    ||p_{t+1} - c|| >= ||p_t - c|| - dt * v * a_i,
hence h_i(p_{t+1}) >= h_i(p_t) - dt * v * a_i >= (1 - gamma * dt) h_i(p_t) when
a_i > 0, and h_i(p_{t+1}) >= h_i(p_t) when a_i <= 0. Wall barriers are linear,
so the same bound holds with equality.

Assumptions behind the guarantee:
  1. the robot follows the forward-Euler unicycle model above (no slip, delay or
     actuation error; v >= 0);
  2. obstacles are static circles whose positions and radii are known exactly,
     and the filter uses the true robot state (observation noise does not enter);
  3. the initial state is safe (the scenario generator ensures this);
  4. the filter is applied at every step.
It does not guarantee that the goal is reached: the robot may stop in front of
an obstacle (deadlock). It says nothing about moving obstacles, sensing errors
or a physical robot. If a barrier is already negative (an assumption was
broken), the constraint cannot be met with v >= 0 and the filter falls back to
v = 0 for that approach direction (stop; omega can still turn the robot).

Reference: A. D. Ames et al., "Control Barrier Functions: Theory and
Applications", ECC 2019 (continuous-time CBF condition dh/dt >= -gamma h).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from safenav.env.obstacle_map import Obstacle

_EPS = 1e-9


@dataclass
class CBFSafetyFilter:
    """Closed-form discrete-time CBF speed filter (see module docstring)."""

    dt: float = 0.1
    robot_radius: float = 0.3
    safe_distance: float = 0.8
    gamma: float = 0.5
    workspace_size: Optional[float] = 10.0
    wall_margin: float = 0.1
    v_max: float = 1.0
    omega_max: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.gamma * self.dt > 1.0:
            raise ValueError("gamma * dt must be <= 1 for the discrete-time barrier condition")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CBFSafetyFilter":
        env_cfg = config.get("env", {})
        safety_cfg = config.get("safety", {})
        return cls(
            dt=float(env_cfg.get("dt", 0.1)),
            robot_radius=float(env_cfg.get("robot_radius", 0.3)),
            safe_distance=float(safety_cfg.get("safe_distance", 0.8)),
            gamma=float(safety_cfg.get("cbf_gamma", 0.5)),
            workspace_size=float(env_cfg.get("workspace_size", 10.0)),
            wall_margin=float(safety_cfg.get("wall_margin", 0.1)),
            enabled=bool(safety_cfg.get("enabled", True)),
        )

    # ── Barrier values ──────────────────────────────────────────────────────

    def barriers(
        self,
        robot_pos: Sequence[float],
        robot_theta: float,
        obstacles: Sequence[Obstacle],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Barrier values h_i and closing rates a_i for every obstacle and wall.

        Returns:
            (h, a): arrays of equal length; constraint i is a_i * v <= gamma * h_i.
        """
        px, py = float(robot_pos[0]), float(robot_pos[1])
        ux, uy = np.cos(robot_theta), np.sin(robot_theta)
        h_list: List[float] = []
        a_list: List[float] = []

        for obs in obstacles:
            dx, dy = obs.x - px, obs.y - py
            dist = float(np.hypot(dx, dy))
            h_list.append(dist - obs.radius - self.robot_radius - self.safe_distance)
            if dist > _EPS:
                a_list.append((ux * dx + uy * dy) / dist)
            else:
                a_list.append(1.0)  # degenerate: robot centre on obstacle centre

        if self.workspace_size is not None:
            w = self.workspace_size
            inset = self.robot_radius + self.wall_margin
            h_list += [px - inset, w - px - inset, py - inset, w - py - inset]
            a_list += [-ux, ux, -uy, uy]

        return np.asarray(h_list, dtype=float), np.asarray(a_list, dtype=float)

    def speed_limit(
        self,
        robot_pos: Sequence[float],
        robot_theta: float,
        obstacles: Sequence[Obstacle],
    ) -> float:
        """Largest speed in [0, v_max] that satisfies every barrier constraint."""
        h, a = self.barriers(robot_pos, robot_theta, obstacles)
        closing = a > _EPS
        if not np.any(closing):
            return self.v_max
        limits = self.gamma * h[closing] / a[closing]
        return float(np.clip(limits.min(), 0.0, self.v_max))

    def filter(
        self,
        action: Sequence[float],
        robot_pos: Sequence[float],
        robot_theta: float,
        obstacles: Sequence[Obstacle],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Project a nominal action [v, omega] to the closest safe action.

        Returns:
            (safe_action, info) where safe_action is a float64 array [v_safe, omega]
            (omega clipped to the action bounds but otherwise unchanged) and info
            holds the speed limit and whether the filter changed the speed.
        """
        nominal = np.asarray(action, dtype=np.float64).copy()
        v_nom = float(np.clip(nominal[0], 0.0, self.v_max))
        omega = float(np.clip(nominal[1], -self.omega_max, self.omega_max))

        if not self.enabled:
            return np.array([v_nom, omega]), {"v_limit": self.v_max, "intervened": False, "v_nominal": v_nom}

        v_limit = self.speed_limit(robot_pos, robot_theta, obstacles)
        v_safe = min(v_nom, v_limit)
        intervened = v_safe < v_nom - 1e-9
        return np.array([v_safe, omega]), {"v_limit": v_limit, "intervened": bool(intervened), "v_nominal": v_nom}

    def min_barrier(self, robot_pos: Sequence[float], obstacles: Sequence[Obstacle]) -> float:
        """Smallest barrier value at a position (>= 0 means inside the safe set)."""
        h, _ = self.barriers(robot_pos, 0.0, obstacles)
        return float(h.min()) if h.size else float("inf")


class SafetyLayer(nn.Module):
    """
    Backwards-compatible wrapper around CBFSafetyFilter with the original
    `project_action` interface.

    Note: the projection is computed in NumPy; gradients do not flow through
    `forward`. In training the filter is treated as part of the environment
    (see SafetyFilterWrapper), which is the standard way to use a safety shield
    with policy-gradient methods.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.filter_ = CBFSafetyFilter.from_config(config)
        self.enabled = self.filter_.enabled
        self.safe_distance = self.filter_.safe_distance
        self.cbf_gamma = self.filter_.gamma
        self.dt = self.filter_.dt
        self.robot_radius = self.filter_.robot_radius

    def project_action(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        robot_theta: float,
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """
        Project nominal action [v, omega] to the closest safe action.

        Returns:
            Projected action [v_safe, omega] as a float64 array.
        """
        safe, _ = self.filter_.filter(action, robot_pos, robot_theta, obstacles)
        return safe

    def forward(
        self,
        action: torch.Tensor,
        robot_pos: np.ndarray,
        robot_theta: float,
        obstacles: List[Obstacle],
    ) -> torch.Tensor:
        """Torch convenience wrapper (no gradient through the projection)."""
        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 1:
            safe = self.project_action(action_np, robot_pos, robot_theta, obstacles)
        else:
            safe = np.stack([self.project_action(a, robot_pos, robot_theta, obstacles) for a in action_np])
        return torch.as_tensor(safe, dtype=action.dtype, device=action.device)


class SafetyFilterWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies the CBF filter to every action before it
    reaches NavigationEnv. The agent keeps its own (nominal) action, so policy
    log-probabilities stay consistent; the filter is part of the environment
    from the agent's point of view.

    Extra info keys: nominal_action, safe_action, safety_intervened, speed_limit.
    """

    def __init__(self, env: gym.Env, safety_filter: CBFSafetyFilter):
        super().__init__(env)
        self.safety_filter = safety_filter

    def step(self, action):
        base = self.env.unwrapped
        nominal = np.asarray(action, dtype=np.float64)
        safe, finfo = self.safety_filter.filter(
            nominal, base.robot_pos, base.robot_theta, base.obstacle_map.obstacles
        )
        obs, reward, terminated, truncated, info = self.env.step(safe)
        info["nominal_action"] = np.clip(nominal, base.action_space.low, base.action_space.high)
        info["safe_action"] = safe
        info["safety_intervened"] = finfo["intervened"]
        info["speed_limit"] = finfo["v_limit"]
        return obs, reward, terminated, truncated, info
