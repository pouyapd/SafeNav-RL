"""
navigation_env.py

Custom Gymnasium-compatible environment for 2D differential-drive robot navigation.

This implements the core MDP:
  - State: robot pose + goal info + risk score
  - Action: linear and angular velocity
  - Dynamics: differential-drive kinematics
  - Reward: shaped reward encouraging safe, efficient navigation

Compatible with the standard gym.Env API so it works with any
RL library (Stable-Baselines3, CleanRL, custom PPO, etc.).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from env.obstacle_map import ObstacleMap
from env.curriculum import CurriculumScheduler


class NavigationEnv(gym.Env):
    """
    2D Navigation environment with safety-aware observations.

    Observation vector (6D):
        [x_norm, y_norm, sin(θ), cos(θ), d_goal_norm, angle_error_norm, risk_score]
        Note: sin/cos encoding avoids angle discontinuity at ±π

    Action vector (2D, continuous):
        [v, ω] — linear velocity ∈ [0, 1], angular velocity ∈ [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict[str, Any], curriculum: Optional[CurriculumScheduler] = None):
        super().__init__()

        # Load config sections
        env_cfg = config.get("env", {})
        reward_cfg = config.get("reward", {})
        safety_cfg = config.get("safety", {})

        self.workspace_size = env_cfg.get("workspace_size", 10.0)
        self.dt = env_cfg.get("dt", 0.1)
        self.robot_radius = env_cfg.get("robot_radius", 0.3)
        self.goal_radius = env_cfg.get("goal_radius", 0.5)
        self.max_steps = env_cfg.get("max_steps", 200)
        self.sensor_noise_std = env_cfg.get("sensor_noise_std", 0.02)

        self.reward_cfg = reward_cfg
        self.safe_distance = safety_cfg.get("safe_distance", 0.8)

        self.curriculum = curriculum

        # Observation: [x_n, y_n, sin(θ), cos(θ), d_goal_n, angle_err_n, risk]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0,  1.0, 1.0,  1.0, 1.0], dtype=np.float32),
        )

        # Action: [v, ω]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
        )

        self.obstacle_map = ObstacleMap(workspace_size=self.workspace_size)
        self._rng = np.random.default_rng(seed=None)

        # State
        self.robot_pos = np.zeros(2)
        self.robot_theta = 0.0
        self.goal_pos = np.zeros(2)
        self._step_count = 0
        self._prev_dist_to_goal = 0.0

        # Trajectory storage (for evaluation and visualization)
        self.trajectory: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Get current curriculum parameters
        env_params = {}
        if self.curriculum is not None:
            env_params = self.curriculum.get_env_params()
            self.max_steps = env_params.get("max_steps", self.max_steps)

        min_goal_dist = env_params.get("min_goal_dist", 3.0)
        max_goal_dist = env_params.get("max_goal_dist", 6.0)

        # Place robot randomly near the center-left region
        margin = 1.5
        self.robot_pos = self._rng.uniform(margin, self.workspace_size / 2, size=2)
        self.robot_theta = self._rng.uniform(-np.pi, np.pi)

        # Place goal at a distance from the robot
        goal_dist = self._rng.uniform(min_goal_dist, max_goal_dist)
        goal_angle = self._rng.uniform(-np.pi, np.pi)
        self.goal_pos = self.robot_pos + goal_dist * np.array([np.cos(goal_angle), np.sin(goal_angle)])
        # Clip goal inside workspace
        self.goal_pos = np.clip(self.goal_pos, margin, self.workspace_size - margin)

        # Generate obstacles
        self.obstacle_map.min_obstacles = env_params.get("min_obstacles", 2)
        self.obstacle_map.max_obstacles = env_params.get("max_obstacles", 6)
        self.obstacle_map.generate(self.robot_pos, self.goal_pos, self._rng)

        self._step_count = 0
        self._prev_dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        self.trajectory = [self.robot_pos.copy()]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: [v, ω] — linear and angular velocity
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        v = float(np.clip(action[0], 0.0, 1.0))
        omega = float(np.clip(action[1], -1.0, 1.0))

        # ── Differential-drive kinematics ──────────────────────────
        self.robot_pos[0] += v * np.cos(self.robot_theta) * self.dt
        self.robot_pos[1] += v * np.sin(self.robot_theta) * self.dt
        self.robot_theta += omega * self.dt
        # Normalize angle to [-π, π]
        self.robot_theta = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi

        self._step_count += 1
        self.trajectory.append(self.robot_pos.copy())

        # ── Termination checks ─────────────────────────────────────
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        collision = self.obstacle_map.check_collision(*self.robot_pos, self.robot_radius)
        out_of_bounds = self.obstacle_map.is_out_of_bounds(*self.robot_pos, self.robot_radius)
        goal_reached = dist_to_goal < self.goal_radius

        terminated = goal_reached or collision or out_of_bounds
        truncated = self._step_count >= self.max_steps

        # ── Reward ─────────────────────────────────────────────────
        reward = self._compute_reward(
            dist_to_goal=dist_to_goal,
            collision=collision or out_of_bounds,
            goal_reached=goal_reached,
        )
        self._prev_dist_to_goal = dist_to_goal

        obs = self._get_obs()
        info = self._get_info(
            goal_reached=goal_reached,
            collision=collision or out_of_bounds,
            dist_to_goal=dist_to_goal,
        )

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        dist_to_goal: float,
        collision: bool,
        goal_reached: bool,
    ) -> float:
        r = self.reward_cfg

        reward = r.get("step_penalty", -0.01)

        # Distance shaping: reward for moving closer to goal
        alpha = r.get("distance_shaping_alpha", 1.0)
        reward += alpha * (self._prev_dist_to_goal - dist_to_goal)

        # Risk penalty
        beta = r.get("risk_penalty_beta", 0.5)
        risk = self.obstacle_map.get_risk_score(*self.robot_pos, self.safe_distance)
        reward -= beta * risk

        if goal_reached:
            reward += r.get("goal_reward", 10.0)
        elif collision:
            reward += r.get("collision_penalty", -5.0)

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation vector with optional sensor noise
        for domain randomization.
        """
        noise = self._rng.normal(0, self.sensor_noise_std, size=2)
        noisy_pos = self.robot_pos + noise

        x_norm = noisy_pos[0] / self.workspace_size
        y_norm = noisy_pos[1] / self.workspace_size
        sin_theta = np.sin(self.robot_theta)
        cos_theta = np.cos(self.robot_theta)

        dx = self.goal_pos[0] - noisy_pos[0]
        dy = self.goal_pos[1] - noisy_pos[1]
        max_dist = self.workspace_size * np.sqrt(2)
        d_goal_norm = np.linalg.norm([dx, dy]) / max_dist

        angle_to_goal = np.arctan2(dy, dx)
        angle_error = angle_to_goal - self.robot_theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # normalize
        angle_error_norm = angle_error / np.pi  # normalize to [-1, 1]

        risk = self.obstacle_map.get_risk_score(*self.robot_pos, self.safe_distance)

        obs = np.array([
            x_norm, y_norm, sin_theta, cos_theta,
            d_goal_norm, angle_error_norm, risk
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "robot_pos": self.robot_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "obstacles": self.obstacle_map.obstacles,
            "risk_score": self.obstacle_map.get_risk_score(*self.robot_pos, self.safe_distance),
            **kwargs,
        }

    def render(self):
        """Basic rendering is handled by visualizer.py."""
        pass

    def close(self):
        pass
