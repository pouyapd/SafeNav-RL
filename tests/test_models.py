"""
test_models.py — Unit tests for neural network models.

Run with: python -m pytest tests/ -v
"""

import numpy as np
import torch
import pytest

from safenav.agents.actor_critic import ActorCritic
from safenav.safety.cbf_filter import SafetyLayer
from safenav.env.obstacle_map import Obstacle

OBS_DIM = 7


@pytest.fixture
def model(config):
    return ActorCritic(config, obs_dim=OBS_DIM)


class TestActorCritic:
    def test_forward_shape(self, model, config):
        obs_dim = OBS_DIM
        action_dim = config["network"]["action_dim"]
        obs = torch.randn(4, obs_dim)  # batch of 4

        action, log_prob, value = model(obs)

        assert action.shape == (4, action_dim)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)

    def test_single_obs_forward(self, model, config):
        obs = torch.randn(1, OBS_DIM)
        action, log_prob, value = model(obs)
        assert not torch.isnan(action).any()
        assert not torch.isnan(value).any()

    def test_evaluate_actions(self, model, config):
        obs_dim = OBS_DIM
        action_dim = config["network"]["action_dim"]
        batch = 8
        obs = torch.randn(batch, obs_dim)
        actions = torch.randn(batch, action_dim)

        log_probs, entropy, values = model.evaluate_actions(obs, actions)

        assert log_probs.shape == (batch,)
        assert entropy.shape == (batch,)
        assert values.shape == (batch,)
        assert (entropy >= 0).all(), "Entropy should be non-negative"

    def test_deterministic_action(self, model, config):
        obs = torch.randn(1, OBS_DIM)

        # Deterministic action should be the same every time
        a1 = model.get_deterministic_action(obs)
        a2 = model.get_deterministic_action(obs)
        assert torch.allclose(a1, a2)

    def test_parameter_count(self, model):
        n_params = model.num_parameters
        assert n_params > 0
        print(f"\n  Model parameters: {n_params:,}")


class TestSafetyLayer:
    def test_no_constraint_far_obstacle(self, config):
        """Safety layer should not modify action when obstacle is far."""
        sl = SafetyLayer(config)
        action = np.array([1.0, 0.0])
        robot_pos = np.array([5.0, 5.0])
        obstacles = [Obstacle(x=9.0, y=9.0, radius=0.3)]  # far away

        safe_action = sl.project_action(action, robot_pos, 0.0, obstacles)
        # Should be (approximately) unchanged
        assert abs(safe_action[0] - 1.0) < 0.1

    def test_constraint_applied_near_obstacle(self, config):
        """Safety layer should reduce velocity near obstacles."""
        sl = SafetyLayer(config)
        action = np.array([1.0, 0.0])
        robot_pos = np.array([5.0, 5.0])
        # Obstacle directly in front at unsafe distance
        obstacles = [Obstacle(x=5.5, y=5.0, radius=0.3)]

        safe_action = sl.project_action(action, robot_pos, 0.0, obstacles)
        # Velocity should be reduced
        assert safe_action[0] < action[0], "Safety layer should reduce velocity near obstacle"

    def test_output_in_valid_range(self, config):
        """Safety layer output should always be in [0,1] x [-1,1]."""
        sl = SafetyLayer(config)
        for _ in range(50):
            action = np.random.uniform([-0.5, -2.0], [1.5, 2.0])
            robot_pos = np.random.uniform(1.0, 9.0, size=2)
            obstacles = [Obstacle(x=float(robot_pos[0] + 0.5), y=float(robot_pos[1]), radius=0.3)]

            safe_action = sl.project_action(action, robot_pos, 0.0, obstacles)
            assert 0.0 <= safe_action[0] <= 1.0
            assert -1.0 <= safe_action[1] <= 1.0
            # Angular velocity is only clipped to the action bounds, never otherwise changed
            assert safe_action[1] == np.clip(action[1], -1.0, 1.0)
