"""The optimised planner (safenav/recovery/fast.py, round 4) must be bit-identical to the reference."""

from pathlib import Path

import numpy as np
import pytest

from safenav.config import get_stage, load_config
from safenav.factory import make_env
from safenav.recovery import (EscapePlanner, EuclideanField, FilteredUnicycleModel, GeodesicField,
                              RecoveryConfig, RecoveryController)
from safenav.recovery.fast import FastEscapePlanner, rollout_end

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "checkpoints"


def _env(stage="expert"):
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides={"env.goal_sampling": "rejection"})
    return make_env(config, task=get_stage(config, stage), safety_filter=True)


def _safe_states(env, model, rng, k):
    out = []
    while len(out) < k:
        p = rng.uniform(0.3, 9.7, 2)
        th = rng.uniform(-np.pi, np.pi)
        h, _ = model.barriers(np.array([p[0]]), np.array([p[1]]), np.array([th]))
        if h.min() >= 0:
            out.append((p, th))
    # include states exactly on the boundary and inside the goal region
    g = model.goal
    out.append((g + np.array([0.1, 0.0]), 0.3))
    return out


@pytest.mark.parametrize("seed", range(8_900_900, 8_900_908))
def test_rollout_end_bit_identical(seed):
    env = _env()
    env.reset(seed=seed)
    model = FilteredUnicycleModel.from_env(env)
    ref = EscapePlanner()
    rng = np.random.default_rng(seed)
    for p, th in _safe_states(env, model, rng, 25):
        hold = np.repeat(rng.uniform(-0.2, 1.2, 2)[None, :], 40, axis=0)
        cands = np.concatenate([ref.library, hold[None]])
        ro = model.rollout(p, th, cands)
        end, reached = rollout_end(model, p, th, cands)
        assert np.array_equal(ro.pos[:, -1, :], end)
        assert np.array_equal(ro.reached_at, reached)


@pytest.mark.parametrize("cost", ["geodesic", "euclidean"])
def test_plans_bit_identical(cost):
    env = _env()
    ref, fast = EscapePlanner(), FastEscapePlanner()
    rng = np.random.default_rng(1)
    for seed in range(8_900_910, 8_900_916):
        env.reset(seed=seed)
        base = env.unwrapped
        model = FilteredUnicycleModel.from_env(env)
        field = GeodesicField(base.obstacle_map.obstacles, base.goal_pos) if cost == "geodesic" else EuclideanField(base.goal_pos)
        for p, th in _safe_states(env, model, rng, 20):
            hold = np.repeat(rng.uniform(-0.2, 1.2, 2)[None, :], 40, axis=0)
            a = ref.plan(model, field, p, th, hold[None])
            b = fast.plan(model, field, p, th, hold[None])
            assert (a.index, a.cost_now, a.cost_end, a.gain, a.reaches_goal) == (b.index, b.cost_now, b.cost_end, b.gain, b.reaches_goal)
            assert np.array_equal(a.controls, b.controls)
            assert ref.predicted_cost(model, field, p, th, hold) == b.extra_costs[0]
            assert fast.predicted_cost(model, field, p, th, hold) == b.extra_costs[0]


class _Stop:
    def reset(self, seed=None):
        pass

    def act(self, obs):
        return np.array([0.0, 0.0])


def _episode(env, policy, cfg, seed):
    ctl = RecoveryController(policy, cfg)
    obs, _ = env.reset(seed=seed)
    ctl.reset(seed=seed)
    ctl.bind(env)
    term = trunc = False
    acts = []
    while not (term or trunc):
        a = np.asarray(ctl.act(obs), dtype=float)
        acts.append(a)
        obs, _, term, trunc, info = env.step(a)
    ctl.notify_goal(int(info["step"]))
    return np.asarray(env.unwrapped.trajectory).copy(), np.asarray(acts), [(x.start_step, x.end_step, x.reason) for x in ctl.stats.activations]


@pytest.mark.parametrize("mode,cost", [("A", "geodesic"), ("C", "geodesic"), ("C", "euclidean")])
def test_episodes_identical_stop_policy(mode, cost):
    env = _env()
    for seed in (8_900_920, 8_900_921):
        r = _episode(env, _Stop(), RecoveryConfig(mode=mode, cost=cost, planner="reference"), seed)
        f = _episode(env, _Stop(), RecoveryConfig(mode=mode, cost=cost, planner="fast"), seed)
        assert np.array_equal(r[0], f[0]) and np.array_equal(r[1], f[1]) and r[2] == f[2]


@pytest.mark.skipif(not (CKPT / "unfiltered" / "seed_1.pt").is_file(), reason="committed checkpoints missing")
@pytest.mark.parametrize("mode", ["A", "C"])
def test_episodes_identical_committed_agent(mode):
    from safenav.agents.policy import PPOPolicy
    env = _env()
    for agent, seed in (("unfiltered", 8_900_930), ("filtered", 8_900_931)):
        pol = PPOPolicy.from_checkpoint(CKPT / agent / "seed_1.pt", deterministic=True)
        r = _episode(env, pol, RecoveryConfig(mode=mode, planner="reference"), seed)
        f = _episode(env, pol, RecoveryConfig(mode=mode, planner="fast"), seed)
        assert np.array_equal(r[0], f[0]) and np.array_equal(r[1], f[1]) and r[2] == f[2]


def test_default_planner_is_reference():
    assert RecoveryConfig().planner == "reference"
    assert type(RecoveryController(_Stop()).planner) is EscapePlanner
    with pytest.raises(ValueError):
        RecoveryConfig(planner="turbo")


def test_timing_lists_recorded():
    env = _env()
    ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C", planner="fast"))
    obs, _ = env.reset(seed=8_900_940)
    ctl.reset()
    ctl.bind(env)
    for _ in range(60):
        obs, *_ = env.step(ctl.act(obs))
    s = ctl.stats
    assert s.planner_calls == len(s.call_wall_ms) == len(s.call_cpu_ms) > 0
    assert abs(sum(s.call_wall_ms) / 1000 - s.planner_time_s) < 1e-9
