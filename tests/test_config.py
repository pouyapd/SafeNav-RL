"""
test_config.py — configuration loading, overrides and validation.
"""

import re

import pytest
import yaml

from safenav.config import (ConfigError, apply_overrides, get_stage, load_config, parse_override,
                            validate_config)


def test_default_config_is_valid(config):
    validate_config(config)


def test_dotted_and_nested_overrides(config):
    out = apply_overrides(config, {"ppo.total_timesteps": 1000, "safety": {"enabled": False}})
    assert out["ppo"]["total_timesteps"] == 1000
    assert out["safety"]["enabled"] is False
    assert out["safety"]["safe_distance"] == config["safety"]["safe_distance"]
    assert config["ppo"]["total_timesteps"] != 1000            # original untouched


def test_parse_override_types():
    assert parse_override("safety.enabled=false") == ("safety.enabled", False)
    assert parse_override("ppo.learning_rate=1e-4") == ("ppo.learning_rate", 1e-4)
    assert parse_override("network.hidden_dims=[64, 64]") == ("network.hidden_dims", [64, 64])
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


@pytest.mark.parametrize("key,value,fragment", [
    ("env.dt", 0, "env.dt"),
    ("env.robot_radius", -1, "env.robot_radius"),
    ("safety.cbf_gamma", 50.0, "cbf_gamma * env.dt"),
    ("safety.qp_solver", "cvxpy", "safety.qp_solver"),
    ("ppo.batch_size", 999999, "batch_size must not exceed"),
    ("ppo.lr_schedule", "cosine", "ppo.lr_schedule"),
    ("network.hidden_dims", [], "network.hidden_dims"),
    ("training.seed", "abc", "training.seed"),
])
def test_invalid_values_are_reported(config, key, value, fragment):
    with pytest.raises(ConfigError, match=re.escape(fragment)):
        validate_config(apply_overrides(config, {key: value}))


def test_missing_section(config):
    del config["ppo"]
    with pytest.raises(ConfigError, match="missing section 'ppo'"):
        validate_config(config)


def test_invalid_stage(config):
    config["curriculum"]["stages"][0]["min_obstacles"] = 9
    with pytest.raises(ConfigError, match="min_obstacles <= max_obstacles"):
        validate_config(config)


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_get_stage(config):
    assert get_stage(config, "expert")["max_obstacles"] == 15
    assert get_stage(config, 0)["name"] == "beginner"
    with pytest.raises(ConfigError):
        get_stage(config, "impossible")
