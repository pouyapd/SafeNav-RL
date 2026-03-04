"""
safety_layer.py

Safety layer implementing a Control Barrier Function (CBF)-inspired
action projection for collision avoidance.

Concept:
  Given a nominal action from the policy, project it to the nearest
  safe action that satisfies a distance constraint from all obstacles.

This turns an unconstrained RL policy into a constrained one,
guaranteeing (approximately) that the robot maintains safe distance
from obstacles at all times.

Reference:
  Ames et al. (2019), "Control Barrier Functions: Theory and Applications"
  Dalal et al. (2018), "Safe Exploration in Continuous Action Spaces"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from env.obstacle_map import Obstacle


class SafetyLayer(nn.Module):
    """
    Differentiable safety layer that projects actions to satisfy
    CBF-based safety constraints.

    For each obstacle within sensing range, it computes a constraint:
        h(s) = ||p - p_obs|| - (r_obs + r_robot + d_safe)

    The action is projected to ensure h does not decrease below zero.
    This is a simplified analytical projection (no QP solver required).
    """

    def __init__(self, config: dict):
        super().__init__()
        safety_cfg = config.get("safety", {})
        self.enabled = safety_cfg.get("enabled", True)
        self.safe_distance = safety_cfg.get("safe_distance", 0.8)
        self.cbf_gamma = safety_cfg.get("cbf_gamma", 0.5)
        self.dt = config.get("env", {}).get("dt", 0.1)
        self.robot_radius = config.get("env", {}).get("robot_radius", 0.3)

    def project_action(
        self,
        action: np.ndarray,
        robot_pos: np.ndarray,
        robot_theta: float,
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """
        Project nominal action to be safe w.r.t. nearby obstacles.

        Algorithm:
          1. For each nearby obstacle, compute CBF value h(s)
          2. Compute the CBF constraint on velocity
          3. If constraint is violated, scale down linear velocity
          4. Return the projected (safe) action

        Args:
            action: Nominal action [v, omega] from the policy
            robot_pos: Current robot position [x, y]
            robot_theta: Current robot heading angle
            obstacles: List of Obstacle objects in the scene

        Returns:
            Projected safe action [v_safe, omega] (same shape as input)
        """
        if not self.enabled or not obstacles:
            return action

        v, omega = float(action[0]), float(action[1])
        safe_v = v

        for obs in obstacles:
            # Distance from robot surface to obstacle surface
            dist_centers = np.sqrt(
                (robot_pos[0] - obs.x) ** 2 + (robot_pos[1] - obs.y) ** 2
            )
            dist_surface = dist_centers - obs.radius - self.robot_radius

            # CBF value: h(s) = dist_surface - safe_distance
            h = dist_surface - self.safe_distance

            if h >= 0:
                # Already safe for this obstacle
                continue

            # Direction from robot to obstacle
            dx = obs.x - robot_pos[0]
            dy = obs.y - robot_pos[1]
            direction = np.array([dx, dy])
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)

            # Component of velocity pointing toward obstacle
            vel_direction = np.array([np.cos(robot_theta), np.sin(robot_theta)])
            approaching_component = np.dot(vel_direction, direction)

            if approaching_component <= 0:
                # Robot is moving away from this obstacle, no constraint needed
                continue

            # CBF constraint: Ḣ + γ·h ≥ 0
            # Ḣ ≈ -v · approaching_component / dt (simplified)
            # Constraint: v · approaching_component ≤ γ · h · dt / ... 
            # Solve for max safe v:
            max_safe_v = self.cbf_gamma * max(h, 0.0) / (approaching_component + 1e-6)
            safe_v = min(safe_v, max_safe_v)

        safe_v = float(np.clip(safe_v, 0.0, 1.0))
        return np.array([safe_v, omega], dtype=np.float32)

    def forward(
        self,
        action: torch.Tensor,
        robot_pos: np.ndarray,
        robot_theta: float,
        obstacles: List[Obstacle],
    ) -> torch.Tensor:
        """
        Torch-compatible wrapper for use in neural network pipelines.
        Projects each action in the batch (or single action) through
        the safety layer.
        """
        if not self.enabled:
            return action

        action_np = action.detach().cpu().numpy()

        if action_np.ndim == 1:
            safe_action = self.project_action(action_np, robot_pos, robot_theta, obstacles)
            return torch.tensor(safe_action, dtype=action.dtype, device=action.device)

        # Batch case — typically used during rollout collection
        safe_actions = np.stack([
            self.project_action(a, robot_pos, robot_theta, obstacles)
            for a in action_np
        ])
        return torch.tensor(safe_actions, dtype=action.dtype, device=action.device)
