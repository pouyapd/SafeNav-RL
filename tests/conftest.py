import copy

import pytest

from safenav.config import load_config

_DEFAULT = load_config()


@pytest.fixture
def config():
    """A fresh copy of the validated default configuration."""
    return copy.deepcopy(_DEFAULT)


@pytest.fixture
def small_config(config):
    """Default config shrunk for fast training tests."""
    config["ppo"].update(total_timesteps=512, rollout_steps=256, batch_size=64, n_epochs=2)
    config["network"]["hidden_dims"] = [32, 32]
    config["training"].update(eval_every=256, eval_episodes=3)
    return config
