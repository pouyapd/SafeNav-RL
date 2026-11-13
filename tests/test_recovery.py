"""Tests for the opt-in deadlock-recovery package (safenav/recovery, round 3)."""

from pathlib import Path

import numpy as np
import pytest

from safenav.config import get_stage, load_config
from safenav.factory import make_env
from safenav.recovery import (EscapePlanner, EuclideanField, FilteredUnicycleModel, GeodesicField,
                              RecoveryConfig, RecoveryController, maneuver_library)
from safenav.safety.cbf_filter import CBFSafetyFilter

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "checkpoints"


def _env(stage="expert", rejection=True):
    overrides = {"env.goal_sampling": "rejection"} if rejection else {}
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides=overrides)
    return config, make_env(config, task=get_stage(config, stage), safety_filter=True)


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_model_matches_environment_step_by_step(seed):
    """The vectorised model reproduces SafetyFilterWrapper + NavigationEnv (positions, headings, speeds)."""
    config, env = _env()
    env.reset(seed=8_900_000 + seed)
    base = env.unwrapped
    rng = np.random.default_rng(seed)
    controls = np.column_stack([rng.uniform(-0.2, 1.2, 120), rng.uniform(-1.2, 1.2, 120)])
    model = FilteredUnicycleModel.from_env(env)
    ro = model.rollout(base.robot_pos.copy(), float(base.robot_theta), controls[None])
    for t, a in enumerate(controls):
        _, _, term, trunc, info = env.step(a)
        assert np.allclose(base.robot_pos, ro.pos[0, t + 1], atol=1e-9, rtol=0)
        assert abs(base.robot_theta - ro.theta[0, t + 1]) < 1e-9
        assert abs(info["executed_action"][0] - ro.v_exec[0, t]) < 1e-9
        assert bool(info["safety_intervened"]) == bool(ro.limited[0, t])
        if term or trunc:
            assert (ro.reached_at[0] == t + 1) == bool(info["goal_reached"])
            break
    assert ro.h_min[0] >= -1e-9


def test_model_speed_limit_matches_filter():
    config, env = _env()
    env.reset(seed=8_900_100)
    base = env.unwrapped
    flt = CBFSafetyFilter.from_config(config)
    model = FilteredUnicycleModel.from_env(env)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.5, 9.5, size=(200, 2))
    th = rng.uniform(-np.pi, np.pi, 200)
    lim = model.speed_limit(pts[:, 0], pts[:, 1], th)
    ref = [flt.speed_limit(p, t, base.obstacle_map.obstacles) for p, t in zip(pts, th)]
    assert np.allclose(lim, ref, atol=1e-12)


def test_geodesic_field_properties():
    config, env = _env()
    env.reset(seed=8_900_200)
    base = env.unwrapped
    geo = GeodesicField(base.obstacle_map.obstacles, base.goal_pos)
    euc = EuclideanField(base.goal_pos)
    assert geo(base.goal_pos) <= 0.05 + 1e-9
    free_i, free_j = np.nonzero(geo.free & np.isfinite(geo.field))
    xy = np.column_stack([geo.centres[free_j], geo.centres[free_i]])[::97]
    # a grid path is never shorter than the straight line (up to one cell and the goal radius)
    assert np.all(geo(xy) >= euc(xy) - 0.5 - 0.1)
    # and at most ~8 % longer in open space plus detours; never below 0
    assert np.all(geo(xy) >= 0)


def test_geodesic_field_unreachable_is_inf():
    from safenav.env.obstacle_map import Obstacle
    # a ring of obstacles around the goal blocks it
    ring = [Obstacle(5 + 1.2 * np.cos(a), 5 + 1.2 * np.sin(a), 0.5) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    geo = GeodesicField(ring, (5.0, 5.0))
    assert np.isinf(geo(np.array([1.0, 1.0])))


def test_planner_deterministic_and_prefers_goal():
    config, env = _env()
    env.reset(seed=8_900_300)
    base = env.unwrapped
    model = FilteredUnicycleModel.from_env(env)
    geo = GeodesicField(base.obstacle_map.obstacles, base.goal_pos)
    pl = EscapePlanner()
    assert pl.library.shape == (65, 40, 2)
    p1 = pl.plan(model, geo, base.robot_pos.copy(), base.robot_theta)
    p2 = pl.plan(model, geo, base.robot_pos.copy(), base.robot_theta)
    assert p1.index == p2.index and np.array_equal(p1.controls, p2.controls)
    # a goal 1 m straight ahead in free space is reached by driving straight
    from safenav.env.obstacle_map import Obstacle
    m = FilteredUnicycleModel([Obstacle(9.0, 9.0, 0.3)], (6.0, 5.0))
    g = GeodesicField([Obstacle(9.0, 9.0, 0.3)], (6.0, 5.0))
    p = pl.plan(m, g, np.array([5.0, 5.0]), 0.0)
    assert p.reaches_goal and p.index == 0


def test_library_layout():
    lib = maneuver_library()
    assert np.all(lib[:, :, 0] >= 0) and np.all(np.abs(lib[:, :, 1]) <= 1)
    # the first candidate drives straight
    assert np.all(lib[0, :, 0] == 1.0) and np.all(lib[0, :, 1] == 0.0)


class _Stop:
    """A policy that never moves (worst-case self-stop)."""
    def reset(self, seed=None):
        pass

    def act(self, obs):
        return np.array([0.0, 0.0])


class _Push:
    """A policy that always drives straight ahead (worst-case filter-blocked stall)."""
    def reset(self, seed=None):
        pass

    def act(self, obs):
        return np.array([1.0, 0.0])


def _run(env, ctl, seed, max_steps=None):
    obs, info = env.reset(seed=seed)
    ctl.reset(seed=seed)
    ctl.bind(env)
    base = env.unwrapped
    flt = env.safety_filter
    clear, steps = [info["obstacle_clearance"]], 0
    term = trunc = False
    while not (term or trunc):
        a = np.asarray(ctl.act(obs), dtype=float)
        expect, _ = flt.filter(a, base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
        obs, _, term, trunc, info = env.step(a)
        assert np.allclose(info["executed_action"], expect), "the filter must check every proposed action"
        clear.append(info["obstacle_clearance"])
        steps += 1
        if max_steps and steps >= max_steps:
            break
    ctl.notify_goal(steps)
    return info, float(np.min(clear)), base.trajectory


def test_detector_triggers_after_window_only():
    config, env = _env()
    ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C", window=30, min_disp=0.15))
    _run(env, ctl, 8_900_400, max_steps=29)
    assert ctl.stats.activations == []
    ctl2 = RecoveryController(_Stop(), RecoveryConfig(mode="C", window=30, min_disp=0.15))
    _run(env, ctl2, 8_900_400, max_steps=40)
    assert len(ctl2.stats.activations) >= 1 and ctl2.stats.activations[0].start_step == 30


@pytest.mark.parametrize("mode,cost", [("A", "geodesic"), ("B", "geodesic"), ("C", "geodesic"), ("C", "euclidean"), ("A", "euclidean")])
@pytest.mark.parametrize("policy", [_Stop, _Push])
def test_recovery_safe_and_terminates(mode, cost, policy):
    """Every mode: 0 collisions, clearance >= margin, activations end within max_steps."""
    config, env = _env()
    for seed in (8_900_500, 8_900_501, 8_900_502):
        cfg = RecoveryConfig(mode=mode, cost=cost)
        ctl = RecoveryController(policy(), cfg)
        info, min_clear, _ = _run(env, ctl, seed)
        assert not info["collision"]
        assert min_clear >= 0.3 - 1e-6
        for a in ctl.stats.activations:
            assert a.end_step >= a.start_step
            if mode != "A":
                assert a.end_step - a.start_step <= cfg.max_steps + 1


def test_recovery_helps_a_stopped_robot():
    """Sanity: with C, a policy that never moves still reaches the goal in some open scenarios."""
    config, env = _env(stage="beginner", rejection=False)
    reached = 0
    for seed in range(8_900_600, 8_900_610):
        ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C"))
        info, _, _ = _run(env, ctl, seed)
        reached += bool(info["goal_reached"])
    assert reached >= 5


@pytest.mark.skipif(not (CKPT / "filtered" / "seed_0.pt").is_file(), reason="committed checkpoints missing")
@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_deterministic_with_committed_agent(mode):
    from safenav.agents.policy import PPOPolicy
    config, env = _env()
    outs = []
    for _ in range(2):
        ctl = RecoveryController(PPOPolicy.from_checkpoint(CKPT / "filtered" / "seed_0.pt", deterministic=True),
                                 RecoveryConfig(mode=mode))
        info, _, traj = _run(env, ctl, 8_900_700)
        outs.append((np.asarray(traj).copy(), [(a.start_step, a.end_step, a.reason) for a in ctl.stats.activations]))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RecoveryConfig(mode="D")
    with pytest.raises(ValueError):
        RecoveryConfig(cost="manhattan")
