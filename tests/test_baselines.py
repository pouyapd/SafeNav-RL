"""
test_baselines.py — reference controllers and the seeded evaluation runner.
"""

import numpy as np
import pytest

from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy, RandomPolicy
from safenav.env.navigation_env import BASE_OBS_DIM
from safenav.evaluation.runner import episode_seeds, evaluate_policy


def _obs(angle_error, ranges=None, n_beams=16, max_range=3.0):
    obs = np.zeros(BASE_OBS_DIM + n_beams, dtype=np.float32)
    obs[5] = angle_error / np.pi
    obs[BASE_OBS_DIM:] = 1.0 if ranges is None else np.asarray(ranges) / max_range
    return obs


def test_go_to_goal_turns_toward_goal():
    pol = GoToGoalPolicy()
    v, w = pol.act(_obs(0.0))
    assert v == pytest.approx(1.0) and w == pytest.approx(0.0)
    v, w = pol.act(_obs(np.pi / 2))
    assert v == pytest.approx(0.0, abs=1e-6) and w == pytest.approx(1.0)
    assert pol.act(_obs(-0.2))[1] < 0


def test_potential_field_steers_away_from_obstacle_ahead(config):
    pol = PotentialFieldPolicy.from_config(config)
    ranges = np.full(16, 3.0)
    ahead = int(np.argmin(np.abs(pol.beam_offsets)))           # beam pointing forward
    ranges[ahead] = 0.6
    ranges[ahead + 1] = 0.7                                    # obstacle slightly to the left
    _, w = pol.act(_obs(0.0, ranges))
    assert w < 0, "should turn right, away from an obstacle ahead-left"
    v_free, _ = pol.act(_obs(0.0))
    v_blocked, _ = pol.act(_obs(0.0, ranges))
    assert v_blocked < v_free


def test_potential_field_requires_range_sensor(config):
    config["sensor"]["num_beams"] = 0
    with pytest.raises(ValueError):
        PotentialFieldPolicy.from_config(config)


def test_random_policy_is_seeded():
    np.testing.assert_allclose(RandomPolicy(seed=3).act(None), RandomPolicy(seed=3).act(None))


def test_episode_seeds_disjoint_between_stages():
    s0, s1 = episode_seeds(2_000_000, 0, 100), episode_seeds(2_000_000, 1, 100)
    assert len(set(s0) | set(s1)) == 200


def test_evaluation_is_deterministic_and_uses_identical_scenarios(config):
    seeds = episode_seeds(7_000_000, 0, 5)
    a = evaluate_policy(config, GoToGoalPolicy(), "beginner", seeds, safety_filter=True)
    b = evaluate_policy(config, GoToGoalPolicy(), "beginner", seeds, safety_filter=True)
    assert [r.to_row() for r in a] == [r.to_row() for r in b]
    c = evaluate_policy(config, GoToGoalPolicy(), "beginner", seeds, safety_filter=False)
    assert [r.straight_line for r in a] == [r.straight_line for r in c]   # same start and goal


def test_filter_prevents_collisions_of_go_to_goal(config):
    seeds = episode_seeds(7_000_000, 2, 30)
    unsafe = evaluate_policy(config, GoToGoalPolicy(), "expert", seeds, safety_filter=False)
    safe = evaluate_policy(config, GoToGoalPolicy(), "expert", seeds, safety_filter=True)
    assert any(r.collision for r in unsafe)
    assert not any(r.collision for r in safe)
    assert min(r.min_clearance for r in safe) >= config["safety"]["safe_distance"] - 1e-9
