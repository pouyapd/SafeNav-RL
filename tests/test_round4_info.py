"""Round 4: imperfect-information recovery (perception.py) and map-based baselines (global_baselines.py)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safenav.config import get_stage, load_config
from safenav.factory import make_env
from safenav.recovery import (ExactInfo, MapError, NoisyLocalisation, RecoveryConfig, RecoveryController, SensedMap)
from safenav.recovery.global_baselines import PathFollower, PlannerOnly

ROOT = Path(__file__).resolve().parents[1]


def _env(stage="expert", rejection=True):
    over = {"env.goal_sampling": "rejection"} if rejection else {}
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides=over)
    return make_env(config, task=get_stage(config, stage), safety_filter=True)


class _Stop:
    def reset(self, seed=None):
        pass

    def act(self, obs):
        return np.array([0.0, 0.0])


def _run(env, ctl, seed, check_filter=True):
    obs, info = env.reset(seed=seed)
    ctl.reset(seed=seed)
    ctl.bind(env, seed)
    base, flt = env.unwrapped, env.safety_filter
    clear = [info["obstacle_clearance"]]
    term = trunc = False
    while not (term or trunc):
        a = np.asarray(ctl.act(obs), dtype=float)
        if check_filter:
            expect, _ = flt.filter(a, base.robot_pos, base.robot_theta, base.obstacle_map.obstacles)
        obs, _, term, trunc, info = env.step(a)
        if check_filter:
            assert np.allclose(info["executed_action"], expect)
        clear.append(info["obstacle_clearance"])
    if hasattr(ctl, "notify_goal"):
        ctl.notify_goal(int(info["step"]))
    return info, float(np.min(clear)), np.asarray(base.trajectory).copy()


def test_exact_info_is_the_default_behaviour():
    env = _env()
    a = _run(env, RecoveryController(_Stop(), RecoveryConfig(mode="C")), 8_901_000)
    b = _run(env, RecoveryController(_Stop(), RecoveryConfig(mode="C"), info=ExactInfo()), 8_901_000)
    assert np.array_equal(a[2], b[2])


@pytest.mark.parametrize("info_cls", [NoisyLocalisation, MapError, SensedMap])
def test_imperfect_information_safe_and_deterministic(info_cls):
    env = _env()
    trajs = []
    for _ in range(2):
        ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C", planner="fast"), info=info_cls())
        info, min_clear, traj = _run(env, ctl, 8_901_010)
        assert not info["collision"] and min_clear >= 0.3 - 1e-6
        trajs.append(traj)
    assert np.array_equal(trajs[0], trajs[1])


def test_noisy_localisation_noise_level():
    env = _env()
    env.reset(seed=8_901_020)
    loc = NoisyLocalisation(0.05, 0.05)
    loc.bind(env, 8_901_020)
    errs = np.array([loc.state()[0] - env.unwrapped.robot_pos for _ in range(4000)])
    assert abs(errs.std() - 0.05) < 0.005 and abs(errs.mean()) < 0.005


def test_map_error_changes_map_but_scoring_uses_truth():
    env = _env()
    ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C", planner="fast"), info=MapError())
    env.reset(seed=8_901_030)
    ctl.reset()
    ctl.bind(env, 8_901_030)
    true = env.unwrapped.obstacle_map.obstacles
    wrong = ctl.info.obstacles()
    assert len(wrong) <= len(true)
    assert ctl.plan_geo is not ctl.geo
    assert len(ctl.model.ox) == len(wrong)
    # the scoring field is the true one
    from safenav.recovery import GeodesicField
    g = GeodesicField(true, env.unwrapped.goal_pos)
    assert np.array_equal(g.field, ctl.geo.field)


def test_sensed_map_only_adds_obstacles_that_were_hit():
    env = _env()
    obs, _ = env.reset(seed=8_901_040)
    sm = SensedMap()
    sm.bind(env, 8_901_040)
    assert sm.obstacles() == []
    changed = sm.update(obs)
    base = env.unwrapped
    true = base.obstacle_map.obstacles
    # every known obstacle must be within sensor range (3 m + radius) of the robot
    for o in sm.obstacles():
        assert o in true
        assert np.hypot(o.x - base.robot_pos[0], o.y - base.robot_pos[1]) - o.radius <= base.max_range + 0.2
    assert changed == (len(sm.obstacles()) > 0)


def test_sensed_map_rebuilds_during_episode():
    env = _env()
    ctl = RecoveryController(_Stop(), RecoveryConfig(mode="C", planner="fast"), info=SensedMap())
    info, min_clear, _ = _run(env, ctl, 8_901_050)
    assert ctl.map_rebuilds >= 1 and ctl.info.known.sum() >= 1
    assert min_clear >= 0.3 - 1e-6


@pytest.mark.parametrize("cls", [PlannerOnly, PathFollower])
def test_map_baselines_safe_and_reach_easy_goals(cls):
    env = _env(stage="beginner", rejection=False)
    reached = 0
    for seed in range(8_901_100, 8_901_108):
        info, min_clear, _ = _run(env, cls(), seed)
        assert not info["collision"] and min_clear >= 0.3 - 1e-6
        reached += bool(info["goal_reached"])
    assert reached >= 6


def test_runner_smoke(tmp_path):
    cmd = [sys.executable, str(ROOT / "research" / "experiments" / "round4" / "run_round4.py"), "--tag", "smoke",
           "--out-root", str(tmp_path), "--methods", "nofilter", "baseline", "C", "C:reference", "C_loc", "C_map",
           "C_sensed", "planner_only", "path_follower", "--agent", "filtered=results/checkpoints/filtered",
           "--stages", "expert", "--seed-base", "8901200", "--n", "1", "--set", "env.goal_sampling=rejection",
           "--record-calls", "--heading-perturb"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    df = pd.read_csv(tmp_path / "smoke" / "episodes.csv")
    assert len(df) == 5 * 7 + 2
    assert set(df.method) == {"nofilter", "baseline", "C", "C_loc", "C_map", "C_sensed", "planner_only", "path_follower"}
    f = df[df.method != "nofilter"]
    assert (f.outcome != "collision").all() and f.min_obstacle_clearance.min() >= 0.3 - 1e-6
    assert df.heading_offset.abs().max() > 0
    # resumability: a second invocation adds nothing
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    assert len(pd.read_csv(tmp_path / "smoke" / "episodes.csv")) == len(df)
    calls = pd.read_csv(tmp_path / "smoke" / "calls.csv")
    assert set(calls.method) <= set(df.method)
