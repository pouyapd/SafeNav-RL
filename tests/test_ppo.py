"""
test_ppo.py — rollout buffer (GAE), trainer bookkeeping and reproducibility.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from safenav.agents.policy import PPOPolicy
from safenav.training.ppo_trainer import PPOTrainer
from safenav.training.rollout_buffer import RolloutBuffer


def _buffer(rewards, values, dones, gamma=0.9, lam=0.8):
    buf = RolloutBuffer(buffer_size=len(rewards), obs_dim=1, action_dim=1, gamma=gamma, gae_lambda=lam)
    for r, v, d in zip(rewards, values, dones):
        buf.add(obs=np.zeros(1), action=np.zeros(1), reward=r, done=d, value=v, log_prob=0.0)
    return buf


def test_gae_hand_computed_with_episode_boundary():
    # dones[1] = 1: transition 1 ends an episode, so it must not bootstrap and
    # the recursion is cut between t=1 and t=2.
    buf = _buffer(rewards=[1.0, 1.0, 1.0], values=[0.5, 0.5, 0.5], dones=[0, 1, 0])
    buf.compute_advantages(last_value=2.0)
    a2 = 1 + 0.9 * 2.0 - 0.5              # 2.3
    a1 = 1 - 0.5                          # 0.5 (terminal)
    a0 = (1 + 0.9 * 0.5 - 0.5) + 0.9 * 0.8 * a1
    np.testing.assert_allclose(buf.advantages, [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(buf.returns, buf.advantages + 0.5, rtol=1e-6)


def test_gae_last_transition_terminal_ignores_last_value():
    buf = _buffer(rewards=[0.0, 1.0], values=[0.0, 0.0], dones=[0, 1])
    buf.compute_advantages(last_value=100.0)
    assert buf.advantages[1] == pytest.approx(1.0)


def test_gae_lambda_one_equals_discounted_return():
    rewards = [0.1, -0.2, 0.3, 0.4]
    buf = _buffer(rewards, values=[0.0] * 4, dones=[0, 0, 0, 0], gamma=0.99, lam=1.0)
    buf.compute_advantages(last_value=0.0)
    expected = [sum(0.99 ** k * r for k, r in enumerate(rewards[t:])) for t in range(4)]
    np.testing.assert_allclose(buf.advantages, expected, rtol=1e-5)


def test_rollout_stores_policy_samples_and_matching_log_probs(small_config, tmp_path):
    """
    The buffer must hold the policy's own sampled action (not the filtered or
    clipped one), so that the PPO ratio is exactly 1 before the first update.
    """
    torch.set_num_threads(1)
    trainer = PPOTrainer(small_config, run_dir=str(tmp_path), safety_filter=True, verbose=False)
    obs, _ = trainer.env.reset(seed=0)
    trainer._ep_return, trainer._ep_length, trainer._ep_interventions = 0.0, 0, 0
    trainer._collect_rollout(obs)
    buf = trainer.buffer
    with torch.no_grad():
        logp, _, _ = trainer.model.evaluate_actions(torch.as_tensor(buf.observations),
                                                    torch.as_tensor(buf.actions))
    np.testing.assert_allclose(logp.numpy(), buf.log_probs, atol=1e-5)
    # Raw Gaussian samples fall outside the executable range [0, 1] for v.
    assert (buf.actions[:, 0] < 0).any() or (buf.actions[:, 0] > 1).any()
    trainer.logger.close()


def test_training_writes_logs_and_loadable_checkpoints(small_config, tmp_path):
    torch.set_num_threads(1)
    PPOTrainer(small_config, run_dir=str(tmp_path), verbose=False).train()
    for name in ("episodes.csv", "progress.csv", "validation.csv", "run_info.json", "train_summary.json"):
        assert (tmp_path / name).is_file(), name
    progress = pd.read_csv(tmp_path / "progress.csv")
    assert len(progress) == 2 and {"approx_kl", "explained_variance", "success_rate_100"} <= set(progress.columns)
    for ckpt in ("best.pt", "final.pt"):
        policy = PPOPolicy.from_checkpoint(tmp_path / "checkpoints" / ckpt)
        action = policy.act(np.zeros(policy.model.obs_dim, dtype=np.float32))
        assert action.shape == (2,)


def test_interrupted_and_resumed_training_matches_uninterrupted(small_config, tmp_path):
    """Exact resumption: stop after 2 of 4 updates, resume in a fresh trainer, compare everything."""
    torch.set_num_threads(1)
    small_config["ppo"]["total_timesteps"] = 4 * 256
    small_config["training"].update(eval_every=512, eval_episodes=3, resume_every=512)

    full = PPOTrainer(small_config, run_dir=str(tmp_path / "full"), verbose=False)
    full.train()

    first = PPOTrainer(small_config, run_dir=str(tmp_path / "part"), verbose=False)
    assert first.train(stop_after_updates=2)["interrupted_after_update"] == 2
    assert (tmp_path / "part" / "checkpoints" / "resume.pt").is_file()
    resumed = PPOTrainer(small_config, run_dir=str(tmp_path / "part"), verbose=False, resume=True)
    resumed.train()

    for key, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    for name in ("episodes.csv", "validation.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "full" / name), pd.read_csv(tmp_path / "part" / name))
    timing = ["fps"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "full" / "progress.csv").drop(columns=timing),
                                  pd.read_csv(tmp_path / "part" / "progress.csv").drop(columns=timing))
    assert not (tmp_path / "part" / "checkpoints" / "resume.pt").exists()   # removed once finished


def test_training_is_reproducible(small_config, tmp_path):
    """Same seed, same config -> identical weights (CPU, single thread)."""
    torch.set_num_threads(1)
    states = []
    for run in ("a", "b"):
        trainer = PPOTrainer(small_config, run_dir=str(tmp_path / run), verbose=False)
        trainer.train()
        states.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


def test_correction_penalty_reaches_training_reward(small_config, tmp_path):
    """E1 control vs treatment: the --set override must change the rewards PPO trains on."""
    from safenav.config import apply_overrides
    torch.set_num_threads(1)
    runs = {}
    for arm, alpha in (("control", 0.0), ("penalty", 1.0)):
        cfg = apply_overrides(small_config, {"safety.correction_penalty": alpha})
        trainer = PPOTrainer(cfg, run_dir=str(tmp_path / arm), safety_filter=True, verbose=False)
        trainer.train()
        runs[arm] = (pd.read_csv(tmp_path / arm / "progress.csv"), trainer.model.state_dict())
    control, penalty = runs["control"][0], runs["penalty"][0]
    assert (control["correction_penalty_mean"] == 0).all()
    assert (penalty["correction_penalty_mean"] > 0).all()
    assert (control["filter_gap_sq_mean"] >= 0).all()
    # Same seed and config otherwise: the first rollout is identical, so any weight difference
    # comes from the penalised rewards.
    assert control["intervention_rate"].iloc[0] == penalty["intervention_rate"].iloc[0]
    assert any(not torch.equal(v, runs["penalty"][1][k]) for k, v in runs["control"][1].items())
