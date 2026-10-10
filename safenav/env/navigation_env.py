"""
navigation_env.py

Gymnasium environment for 2D differential-drive (unicycle) robot navigation
among static circular obstacles.

This implements the core MDP:
  - State: robot pose, goal, obstacle layout (hidden) and a simulated range sensor
  - Action: linear and angular velocity
  - Dynamics: forward-Euler unicycle kinematics
  - Reward: shaped reward encouraging safe, efficient navigation

It follows the standard gymnasium.Env API, so it works with any RL library.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from safenav.env.curriculum import CurriculumScheduler
from safenav.env.obstacle_map import ObstacleMap

# Task parameters used when no curriculum stage is given.
DEFAULT_TASK = {
    "min_obstacles": 2,
    "max_obstacles": 6,
    "min_goal_dist": 3.0,
    "max_goal_dist": 6.0,
}

BASE_OBS_DIM = 7


class NavigationEnv(gym.Env):
    """
    2D navigation environment with safety-aware observations.

    Observation vector (7 + K values, K = sensor.num_beams):
        [x_norm, y_norm, sin(θ), cos(θ), d_goal_norm, angle_error_norm, risk,
         range_1 / max_range, ..., range_K / max_range]
        sin/cos encoding avoids the angle discontinuity at ±π. The K range
        readings come from a simulated 360° range sensor (rays evenly spaced
        around the robot heading) that detects obstacles and walls.
        With K = 0 the observation is the original 7-D vector.

    Action vector (2D, continuous):
        [v, ω] — linear velocity ∈ [0, 1] m/s, angular velocity ∈ [-1, 1] rad/s.
        Out-of-range actions are clipped.

    Episode end:
        terminated — goal reached, obstacle collision or robot leaves the workspace
        truncated  — max_steps reached (time limit)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Dict[str, Any],
        curriculum: Optional[CurriculumScheduler] = None,
        task: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        env_cfg = config.get("env", {})
        reward_cfg = config.get("reward", {})
        safety_cfg = config.get("safety", {})
        sensor_cfg = config.get("sensor", {}) or {}

        self.workspace_size = float(env_cfg.get("workspace_size", 10.0))
        self.dt = float(env_cfg.get("dt", 0.1))
        self.robot_radius = float(env_cfg.get("robot_radius", 0.3))
        self.goal_radius = float(env_cfg.get("goal_radius", 0.5))
        self.default_max_steps = int(env_cfg.get("max_steps", 200))
        self.max_steps = self.default_max_steps
        self.sensor_noise_std = float(env_cfg.get("sensor_noise_std", 0.02))
        # Distance at which the risk score reaches 0. Older configs used
        # safety.safe_distance for this, so that stays the fallback.
        self.risk_distance = float(env_cfg.get("risk_distance", safety_cfg.get("safe_distance", 0.8)))

        self.num_beams = int(sensor_cfg.get("num_beams", 0))
        self.max_range = float(sensor_cfg.get("max_range", 3.0))
        self.range_noise_std = float(sensor_cfg.get("range_noise_std", 0.0))
        self.beam_offsets = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False) if self.num_beams else np.zeros(0)

        self.reward_cfg = reward_cfg
        self.curriculum = curriculum
        self.task = dict(DEFAULT_TASK, **(task or {}))
        if task and "max_steps" in task:
            self.max_steps = int(task["max_steps"])

        obs_dim = BASE_OBS_DIM + self.num_beams
        low = np.concatenate([[0.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0], np.zeros(self.num_beams)])
        high = np.concatenate([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.ones(self.num_beams)])
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32),
                                            shape=(obs_dim,), dtype=np.float32)

        # Action: [v, ω]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.obstacle_map = ObstacleMap(workspace_size=self.workspace_size)
        self._rng = np.random.default_rng(seed=None)

        # State
        self.robot_pos = np.zeros(2)
        self.robot_theta = 0.0
        self.start_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self._step_count = 0
        self._prev_dist_to_goal = 0.0

        # Trajectory storage (for evaluation and visualization)
        self.trajectory: list = []
        self.headings: list = []

    # ── Task / difficulty ──────────────────────────────────────────────────

    def set_task(self, task: Dict[str, Any]) -> None:
        """Fix the difficulty (obstacle count, goal distance, max_steps) for future resets."""
        self.task = dict(DEFAULT_TASK, **task)
        self.max_steps = int(task.get("max_steps", self.default_max_steps))
        self.curriculum = None

    def _current_task(self) -> Dict[str, Any]:
        if self.curriculum is not None:
            params = self.curriculum.get_env_params()
            self.max_steps = int(params.get("max_steps", self.max_steps))
            return dict(DEFAULT_TASK, **params)
        return self.task

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        task = self._current_task()

        # Place robot randomly in the lower-left part of the workspace
        margin = 1.5
        self.robot_pos = self._rng.uniform(margin, self.workspace_size / 2, size=2)
        self.robot_theta = float(self._rng.uniform(-np.pi, np.pi))
        self.start_pos = self.robot_pos.copy()

        # Place goal at a random distance and direction from the robot
        goal_dist = self._rng.uniform(task["min_goal_dist"], task["max_goal_dist"])
        goal_angle = self._rng.uniform(-np.pi, np.pi)
        self.goal_pos = self.robot_pos + goal_dist * np.array([np.cos(goal_angle), np.sin(goal_angle)])
        # Clip goal inside workspace
        self.goal_pos = np.clip(self.goal_pos, margin, self.workspace_size - margin)

        # Generate obstacles
        self.obstacle_map.min_obstacles = int(task["min_obstacles"])
        self.obstacle_map.max_obstacles = int(task["max_obstacles"])
        self.obstacle_map.generate(self.robot_pos, self.goal_pos, self._rng)

        self._step_count = 0
        self._prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        self.trajectory = [self.robot_pos.copy()]
        self.headings = [self.robot_theta]

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

        # ── Unicycle kinematics (forward Euler) ────────────────────
        self.robot_pos[0] += v * np.cos(self.robot_theta) * self.dt
        self.robot_pos[1] += v * np.sin(self.robot_theta) * self.dt
        self.robot_theta += omega * self.dt
        # Normalize angle to [-π, π)
        self.robot_theta = float((self.robot_theta + np.pi) % (2 * np.pi) - np.pi)

        self._step_count += 1
        self.trajectory.append(self.robot_pos.copy())
        self.headings.append(self.robot_theta)

        # ── Termination checks ─────────────────────────────────────
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        obstacle_collision = self.obstacle_map.check_collision(*self.robot_pos, self.robot_radius)
        out_of_bounds = self.obstacle_map.is_out_of_bounds(*self.robot_pos, self.robot_radius)
        collision = obstacle_collision or out_of_bounds
        # Touching an obstacle while entering the goal region counts as a collision.
        goal_reached = (dist_to_goal < self.goal_radius) and not collision

        terminated = bool(goal_reached or collision)
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        # ── Reward ─────────────────────────────────────────────────
        reward = self._compute_reward(
            dist_to_goal=dist_to_goal,
            collision=collision,
            goal_reached=goal_reached,
        )
        self._prev_dist_to_goal = dist_to_goal

        obs = self._get_obs()
        info = self._get_info(
            goal_reached=goal_reached,
            collision=collision,
            obstacle_collision=obstacle_collision,
            out_of_bounds=out_of_bounds,
            dist_to_goal=dist_to_goal,
            executed_action=np.array([v, omega]),
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
        reward -= beta * self.risk_score()

        if goal_reached:
            reward += r.get("goal_reward", 10.0)
        elif collision:
            reward += r.get("collision_penalty", -5.0)

        return float(reward)

    # ── Observation helpers ─────────────────────────────────────────────────

    def risk_score(self) -> float:
        """Proximity risk in [0, 1] at the true robot position."""
        return self.obstacle_map.get_risk_score(*self.robot_pos, self.risk_distance)

    def range_readings(self, noisy: bool = False) -> np.ndarray:
        """Range sensor readings (metres) for the current pose, shape (num_beams,)."""
        if self.num_beams == 0:
            return np.zeros(0)
        ranges = self.obstacle_map.ray_cast(self.robot_pos, self.robot_theta + self.beam_offsets, self.max_range)
        if noisy and self.range_noise_std > 0:
            ranges = ranges + self._rng.normal(0.0, self.range_noise_std, size=ranges.shape)
        return np.clip(ranges, 0.0, self.max_range)

    def clearance(self) -> Tuple[float, float]:
        """(obstacle clearance, wall clearance) of the robot body in metres; negative = penetration."""
        obstacle = self.obstacle_map.min_surface_distance(*self.robot_pos) - self.robot_radius
        wall = self.obstacle_map.wall_distance(*self.robot_pos) - self.robot_radius
        return float(obstacle), float(wall)

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation vector. Position and range readings get Gaussian
        sensor noise (domain randomization); the risk score uses the true position.
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

        base = [x_norm, y_norm, sin_theta, cos_theta, d_goal_norm, angle_error_norm, self.risk_score()]
        ranges = self.range_readings(noisy=True) / self.max_range if self.num_beams else np.zeros(0)
        obs = np.concatenate([np.asarray(base, dtype=float), ranges]).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        obstacle_clearance, wall_clearance = self.clearance()
        info = {
            "step": self._step_count,
            "robot_pos": self.robot_pos.copy(),
            "robot_theta": self.robot_theta,
            "goal_pos": self.goal_pos.copy(),
            "obstacles": self.obstacle_map.obstacles,
            "risk_score": self.risk_score(),
            "obstacle_clearance": obstacle_clearance,
            "wall_clearance": wall_clearance,
            "goal_reached": False,
            "collision": False,
            "obstacle_collision": False,
            "out_of_bounds": False,
        }
        info.update(kwargs)
        return info

    def render(self):
        """Rendering is handled by safenav.viz (static plots and animations)."""
        return None

    def close(self):
        pass
