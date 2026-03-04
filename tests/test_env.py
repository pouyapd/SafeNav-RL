"""
test_env.py — Unit tests for the navigation environment.

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import yaml

from env.navigation_env import NavigationEnv
from env.obstacle_map import ObstacleMap
from env.curriculum import CurriculumScheduler


@pytest.fixture
def config():
    with open("configs/default_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(config):
    config["curriculum"]["enabled"] = False
    return NavigationEnv(config)


class TestNavigationEnv:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (7,), f"Expected obs shape (7,), got {obs.shape}"
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_step_returns_correct_types(self, env):
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self, env):
        """Actions outside the valid range should be clipped, not crash."""
        env.reset(seed=2)
        extreme_action = np.array([100.0, 100.0])  # Way out of range
        obs, reward, terminated, truncated, info = env.step(extreme_action)
        assert obs is not None

    def test_episode_terminates(self, env):
        """An episode must eventually terminate."""
        env.reset(seed=3)
        done = False
        steps = 0
        max_steps = 500
        while not done and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within max steps"

    def test_trajectory_recorded(self, env):
        """Trajectory should be recorded during episode."""
        env.reset(seed=4)
        for _ in range(10):
            env.step(np.array([0.5, 0.0]))
        assert len(env.trajectory) > 1


class TestObstacleMap:
    def test_obstacle_generation(self):
        rng = np.random.default_rng(42)
        omap = ObstacleMap(workspace_size=10.0, min_obstacles=3, max_obstacles=5)
        robot_pos = np.array([2.0, 2.0])
        goal_pos = np.array([8.0, 8.0])
        obstacles = omap.generate(robot_pos, goal_pos, rng)

        assert len(obstacles) >= 0
        # No obstacle should collide with robot start
        for obs in obstacles:
            dist = np.sqrt((obs.x - robot_pos[0])**2 + (obs.y - robot_pos[1])**2)
            assert dist > obs.radius, "Obstacle overlaps robot start position"

    def test_risk_score_range(self):
        rng = np.random.default_rng(0)
        omap = ObstacleMap()
        omap.generate(np.array([1.0, 1.0]), np.array([9.0, 9.0]), rng)
        risk = omap.get_risk_score(5.0, 5.0, safe_distance=1.0)
        assert 0.0 <= risk <= 1.0


class TestCurriculum:
    def test_stage_advancement(self, config):
        sched = CurriculumScheduler(config["curriculum"])
        assert sched.current_stage_idx == 0

        # Record enough successes to advance
        for _ in range(100):
            sched.record_outcome(success=True)

        assert sched.current_stage_idx > 0

    def test_no_advance_on_failures(self, config):
        sched = CurriculumScheduler(config["curriculum"])
        for _ in range(100):
            sched.record_outcome(success=False)
        assert sched.current_stage_idx == 0
