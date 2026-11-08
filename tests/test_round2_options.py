"""
test_round2_options.py — opt-in options added for the round-2 experiments:
the Beta policy (E4), the rejection-sampled goal generator (E3) and the DWA-style planner.
The defaults must leave existing behaviour unchanged.
"""

import numpy as np
import pytest
import torch

from safenav.agents.actor_critic import ActorCritic
from safenav.agents.baselines import DynamicWindowPolicy
from safenav.agents.policy import PPOPolicy
from safenav.config import ConfigError, apply_overrides, get_stage, validate_config
from safenav.env.navigation_env import NavigationEnv
from safenav.factory import make_env
from safenav.training.ppo_trainer import PPOTrainer


# ── Beta policy ─────────────────────────────────────────────────────────────

def test_default_network_is_unchanged_gaussian(config):
    model = ActorCritic(config, obs_dim=23)
    keys = set(model.state_dict())
    assert model.distribution == "gaussian" and "log_std" in keys
    assert not any(k.startswith("action_") for k in keys)
    assert model.actor[-1].out_features == 2


def test_beta_samples_in_bounds_and_log_probs_consistent(config):
    cfg = apply_overrides(config, {"network.distribution": "beta"})
    torch.manual_seed(0)
    model = ActorCritic(cfg, obs_dim=23)
    assert "log_std" not in dict(model.named_parameters())
    obs = torch.randn(256, 23)
    with torch.no_grad():
        actions, logp, _ = model(obs)
        logp2, entropy, _ = model.evaluate_actions(obs, actions)
        mean = model.get_deterministic_action(obs)
    for a in (actions, mean):
        assert torch.all(a[:, 0] >= 0) and torch.all(a[:, 0] <= 1)
        assert torch.all(a[:, 1] >= -1) and torch.all(a[:, 1] <= 1)
    torch.testing.assert_close(logp, logp2)          # PPO ratio is exactly 1 before an update
    assert torch.all(torch.isfinite(entropy))


def test_beta_policy_trains_and_reloads(small_config, tmp_path):
    torch.set_num_threads(1)
    cfg = apply_overrides(small_config, {"network.distribution": "beta"})
    PPOTrainer(cfg, run_dir=str(tmp_path), verbose=False).train()
    policy = PPOPolicy.from_checkpoint(tmp_path / "checkpoints" / "final.pt")
    assert policy.model.distribution == "beta"
    a = policy.act(np.zeros(policy.model.obs_dim, dtype=np.float32))
    assert 0.0 < a[0] < 1.0 and -1.0 < a[1] < 1.0


def test_invalid_distribution_rejected(config):
    with pytest.raises(ConfigError):
        validate_config(apply_overrides(config, {"network.distribution": "cauchy"}))


# ── Goal generator ──────────────────────────────────────────────────────────

def _goals(config, stage, sampling, seeds):
    cfg = apply_overrides(config, {"env.goal_sampling": sampling}) if sampling else config
    env = NavigationEnv(cfg, task=get_stage(cfg, stage))
    out = []
    for s in seeds:
        env.reset(seed=s)
        out.append((env.robot_pos.copy(), env.goal_pos.copy(), env.goal_sampling_fallback,
                    [(o.x, o.y, o.radius) for o in env.obstacle_map.obstacles]))
    return out


def test_default_goal_sampling_is_the_original_clip(config):
    seeds = range(2_200_000, 2_200_030)
    default, explicit = _goals(config, "expert", None, seeds), _goals(config, "expert", "clip", seeds)
    for (p1, g1, _, o1), (p2, g2, _, o2) in zip(default, explicit):
        assert np.array_equal(p1, p2) and np.array_equal(g1, g2) and o1 == o2


@pytest.mark.parametrize("stage", ["beginner", "intermediate", "expert"])
def test_rejection_goals_respect_the_distance_range(config, stage):
    task = get_stage(config, stage)
    for start, goal, fallback, _ in _goals(config, stage, "rejection", range(7_000_000, 7_000_200)):
        d = np.linalg.norm(goal - start)
        assert not fallback
        assert task["min_goal_dist"] - 1e-9 <= d <= task["max_goal_dist"] + 1e-9
        assert np.all(goal >= 1.5) and np.all(goal <= 8.5)


def test_invalid_goal_sampling_rejected(config):
    with pytest.raises(ConfigError):
        validate_config(apply_overrides(config, {"env.goal_sampling": "uniform"}))


# ── DWA-style planner ───────────────────────────────────────────────────────

def test_dwa_heads_for_a_free_goal(config):
    policy = DynamicWindowPolicy.from_config(config)
    obs = np.zeros(23, dtype=np.float32)
    obs[4], obs[5], obs[7:] = 0.3, 0.25, 1.0        # goal 4.2 m away, 45 degrees left, nothing in range
    v, w = policy.act(obs)
    assert v > 0.5 and w > 0


def test_dwa_with_filter_is_collision_free(config):
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    policy = DynamicWindowPolicy.from_config(config)
    for seed in range(2_200_000, 2_200_005):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            action = policy.act(obs)
            assert 0.0 <= action[0] <= 1.0 and -1.0 <= action[1] <= 1.0
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        assert not info["collision"]
