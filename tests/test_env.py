"""
test_env.py — environment dynamics, termination, reward, sensing and seeding.

Run with: python -m pytest
"""

import numpy as np
import pytest

from safenav.env.curriculum import CurriculumScheduler
from safenav.env.navigation_env import BASE_OBS_DIM, NavigationEnv
from safenav.env.obstacle_map import Obstacle, ObstacleMap


@pytest.fixture
def env(config):
    config["curriculum"]["enabled"] = False
    return NavigationEnv(config)


def _place(env, pos, theta, goal, obstacles=()):
    """Put the environment into a hand-made state."""
    env.reset(seed=0)
    env.robot_pos = np.array(pos, dtype=float)
    env.robot_theta = float(theta)
    env.goal_pos = np.array(goal, dtype=float)
    env.obstacle_map.set_obstacles(list(obstacles))
    env._prev_dist_to_goal = float(np.linalg.norm(env.goal_pos - env.robot_pos))
    env._step_count = 0


class TestNavigationEnv:
    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (BASE_OBS_DIM + env.num_beams,)
        assert obs.dtype == np.float32
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_step_returns_correct_types(self, env):
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert type(terminated) is bool
        assert type(truncated) is bool
        assert isinstance(info, dict)

    def test_unicycle_kinematics_exact(self, env):
        """One Euler step moves the robot by v*dt along the old heading and turns by omega*dt."""
        _place(env, (3.0, 3.0), 0.5, (9.0, 9.0))
        env.step(np.array([0.8, -0.4]))
        expected = np.array([3.0, 3.0]) + 0.8 * env.dt * np.array([np.cos(0.5), np.sin(0.5)])
        np.testing.assert_allclose(env.robot_pos, expected, atol=1e-12)
        assert env.robot_theta == pytest.approx(0.5 - 0.4 * env.dt)

    def test_heading_wraps_to_minus_pi_pi(self, env):
        _place(env, (5.0, 5.0), np.pi - 0.01, (9.0, 9.0))
        env.step(np.array([0.0, 1.0]))
        assert -np.pi <= env.robot_theta < np.pi
        assert env.robot_theta == pytest.approx(-np.pi + 0.09, abs=1e-9)

    def test_action_clipping(self, env):
        """Out-of-range actions are clipped to [0, 1] x [-1, 1]."""
        _place(env, (5.0, 5.0), 0.0, (9.0, 9.0))
        _, _, _, _, info = env.step(np.array([100.0, -100.0]))
        np.testing.assert_allclose(info["executed_action"], [1.0, -1.0])

    def test_goal_reached_terminates_with_bonus(self, env):
        _place(env, (5.0, 5.0), 0.0, (5.5, 5.0))
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
        assert terminated and not truncated
        assert info["goal_reached"] and not info["collision"]
        # step penalty + shaping (0.5 -> 0.4) + goal bonus
        assert reward == pytest.approx(-0.01 + 1.0 * 0.1 + 10.0)

    def test_obstacle_collision_terminates_with_penalty(self, env):
        _place(env, (5.0, 5.0), 0.0, (9.0, 9.0), [Obstacle(5.65, 5.0, 0.3)])
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
        assert terminated and info["obstacle_collision"] and info["collision"]
        assert not info["goal_reached"]
        assert reward < -4.0

    def test_out_of_bounds_is_a_collision(self, env):
        _place(env, (0.35, 5.0), np.pi, (9.0, 9.0))
        _, _, terminated, _, info = env.step(np.array([1.0, 0.0]))
        assert terminated and info["out_of_bounds"] and info["collision"]
        assert not info["obstacle_collision"]

    def test_truncation_at_max_steps(self, env):
        _place(env, (5.0, 5.0), 0.0, (9.0, 9.0))
        env.max_steps = 3
        flags = [env.step(np.array([0.0, 0.0]))[2:4] for _ in range(3)]
        assert flags[-1] == (False, True)
        assert all(f == (False, False) for f in flags[:-1])

    def test_reward_shaping_and_risk_penalty(self, env):
        """Reward = step penalty + alpha * progress - beta * risk (no terminal event)."""
        _place(env, (5.0, 5.0), 0.0, (9.0, 5.0), [Obstacle(5.1, 6.0, 0.4)])
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
        assert not terminated
        progress = 4.0 - 3.9
        risk = env.obstacle_map.get_risk_score(5.1, 5.0, env.risk_distance)
        assert risk > 0
        assert reward == pytest.approx(-0.01 + progress - 0.5 * risk)

    def test_same_seed_same_scenario(self, config):
        a, b = NavigationEnv(config), NavigationEnv(config)
        a.reset(seed=123)
        b.reset(seed=123)
        np.testing.assert_allclose(a.robot_pos, b.robot_pos)
        np.testing.assert_allclose(a.goal_pos, b.goal_pos)
        assert [(o.x, o.y, o.radius) for o in a.obstacle_map.obstacles] == \
               [(o.x, o.y, o.radius) for o in b.obstacle_map.obstacles]

    def test_start_and_goal_are_collision_free(self, config):
        env = NavigationEnv(config, task=config["curriculum"]["stages"][-1])
        for seed in range(50):
            env.reset(seed=seed)
            assert not env.obstacle_map.check_collision(*env.start_pos, env.robot_radius)
            assert not env.obstacle_map.check_collision(*env.goal_pos, env.robot_radius)

    def test_episode_terminates(self, env):
        env.reset(seed=3)
        done, steps = False, 0
        while not done and steps < 500:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert done

    def test_legacy_seven_dim_observation(self, config):
        config["sensor"]["num_beams"] = 0
        obs, _ = NavigationEnv(config).reset(seed=0)
        assert obs.shape == (7,)


class TestRangeSensor:
    def test_ray_hits_wall_at_known_distance(self):
        omap = ObstacleMap(workspace_size=10.0)
        omap.set_obstacles([])
        d = omap.ray_cast(np.array([2.0, 5.0]), np.array([0.0, np.pi, np.pi / 2]), max_range=20.0)
        np.testing.assert_allclose(d, [8.0, 2.0, 5.0])

    def test_ray_hits_circle_surface(self):
        omap = ObstacleMap(workspace_size=10.0)
        omap.set_obstacles([Obstacle(6.0, 5.0, 1.0)])
        d = omap.ray_cast(np.array([2.0, 5.0]), np.array([0.0, np.pi]), max_range=3.5)
        np.testing.assert_allclose(d, [3.0, 2.0])  # obstacle surface at x=5; wall at x=0

    def test_range_capped_and_inside_obstacle_reads_zero(self):
        omap = ObstacleMap(workspace_size=10.0)
        omap.set_obstacles([Obstacle(5.0, 5.0, 1.0)])
        assert omap.ray_cast(np.array([5.2, 5.0]), np.array([0.0]), 3.0)[0] == 0.0
        assert omap.ray_cast(np.array([1.0, 1.0]), np.array([np.pi / 4]), 1.0)[0] == pytest.approx(1.0)

    def test_observation_contains_normalised_ranges(self, env):
        _place(env, (2.0, 5.0), 0.0, (9.0, 9.0))
        obs = env._get_obs()
        assert len(obs) == BASE_OBS_DIM + env.num_beams
        assert np.all((obs[BASE_OBS_DIM:] >= 0) & (obs[BASE_OBS_DIM:] <= 1))


class TestObstacleMap:
    def test_obstacle_generation(self):
        rng = np.random.default_rng(42)
        omap = ObstacleMap(workspace_size=10.0, min_obstacles=3, max_obstacles=5)
        robot_pos = np.array([2.0, 2.0])
        goal_pos = np.array([8.0, 8.0])
        obstacles = omap.generate(robot_pos, goal_pos, rng)
        assert len(obstacles) <= 5
        for obs in obstacles:
            assert obs.distance_to_point(*robot_pos) >= omap.min_clearance + 0.5
            assert obs.distance_to_point(*goal_pos) >= omap.min_clearance + 0.5

    def test_risk_score_range(self):
        rng = np.random.default_rng(0)
        omap = ObstacleMap()
        omap.generate(np.array([1.0, 1.0]), np.array([9.0, 9.0]), rng)
        risk = omap.get_risk_score(5.0, 5.0, safe_distance=1.0)
        assert 0.0 <= risk <= 1.0

    def test_risk_score_values(self):
        omap = ObstacleMap()
        omap.set_obstacles([Obstacle(5.0, 5.0, 1.0)])
        assert omap.get_risk_score(6.0, 5.0, 0.8) == pytest.approx(1.0)   # on the surface
        assert omap.get_risk_score(6.4, 5.0, 0.8) == pytest.approx(0.5)   # halfway
        assert omap.get_risk_score(7.0, 5.0, 0.8) == 0.0                  # beyond risk distance


class TestCurriculum:
    def test_stage_advancement(self, config):
        sched = CurriculumScheduler(config["curriculum"])
        assert sched.current_stage_idx == 0
        for _ in range(100):
            sched.record_outcome(success=True)
        assert sched.current_stage_idx > 0

    def test_no_advance_on_failures(self, config):
        sched = CurriculumScheduler(config["curriculum"])
        for _ in range(100):
            sched.record_outcome(success=False)
        assert sched.current_stage_idx == 0

    def test_env_uses_curriculum_stage(self, config):
        sched = CurriculumScheduler(config["curriculum"])
        env = NavigationEnv(config, curriculum=sched)
        env.reset(seed=0)
        assert env.max_steps == config["curriculum"]["stages"][0]["max_steps"]
