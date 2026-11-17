"""
test_heading_escape.py — the opt-in test-time heading escape (safenav/safety/heading_escape.py).

Key properties: kappa = 0 reproduces the plain filter exactly; for kappa > 0 the
one-step barrier bound still holds under an adversarial policy (the guarantee does
not depend on the turn rate); the escape turns the heading away from the obstacle.
"""

import numpy as np
import pytest

from safenav.config import get_stage
from safenav.env.obstacle_map import Obstacle
from safenav.factory import make_env
from safenav.safety.cbf_filter import CBFSafetyFilter
from safenav.safety.heading_escape import escape_turn_sign, make_escape_env


def _rollout(env, seed, actions):
    env.reset(seed=seed)
    out = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((obs.tobytes(), r, tuple(info["executed_action"])))
        if term or trunc:
            break
    return out


def test_kappa_zero_matches_plain_filter(config):
    rng = np.random.default_rng(5)
    actions = [rng.normal([0.7, 0.0], 0.7) for _ in range(250)]
    task = get_stage(config, "expert")
    assert _rollout(make_escape_env(config, task, 0.0), 11, actions) == \
        _rollout(make_env(config, task=task, safety_filter=True), 11, actions)


def test_escape_turns_away_from_the_obstacle():
    n = np.array([1.0, 0.0])                       # obstacle straight ahead along +x
    left = np.array([np.cos(0.5), np.sin(0.5)])    # heading 0.5 rad to the left of n
    right = np.array([np.cos(-0.5), np.sin(-0.5)])
    assert escape_turn_sign(n, left, np.array([1.0, 0.0])) == 1.0     # keep turning left
    assert escape_turn_sign(n, right, np.array([1.0, 0.0])) == -1.0   # keep turning right
    assert escape_turn_sign(n, n, np.array([1.0, -1.0])) == -1.0      # head-on: towards the goal side


def test_escape_only_when_filter_intervenes(config):
    env = make_escape_env(config, get_stage(config, "beginner"), 1.0)
    env.reset(seed=0)
    base = env.unwrapped
    base.robot_pos, base.robot_theta = np.array([5.0, 5.0]), 0.3
    base.obstacle_map.set_obstacles([Obstacle(6.4, 5.0, 0.3)])      # h = 1.4 - 0.9 = 0.5 m
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))                # filter limits v -> escape
    assert info["safety_intervened"] and info["heading_escape"]
    assert info["executed_action"][1] == pytest.approx(1.0)         # turned left at full rate
    base.robot_pos, base.robot_theta = np.array([2.0, 5.0]), np.pi
    base.obstacle_map.set_obstacles([Obstacle(8.0, 5.0, 0.3)])
    _, _, _, _, info = env.step(np.array([0.2, 0.1]))                # nothing ahead -> unchanged
    assert not info["heading_escape"]
    assert info["executed_action"][1] == pytest.approx(0.1)


@pytest.mark.parametrize("kappa", [0.5, 1.0])
def test_forward_invariance_with_escape(config, kappa):
    """Same adversarial test as test_safety_filter.py, with the escape on."""
    flt = CBFSafetyFilter.from_config(config)
    gamma_dt = flt.gamma * flt.dt
    rng = np.random.default_rng(1)
    env = make_escape_env(config, get_stage(config, "expert"), kappa)
    base = env.unwrapped
    for seed in range(30):
        env.reset(seed=seed)
        obstacles = base.obstacle_map.obstacles
        h_prev, _ = flt.barriers(base.robot_pos, base.robot_theta, obstacles)
        for _ in range(250):
            near = base.obstacle_map.nearest_obstacle(*base.robot_pos)
            target = np.arctan2(near.y - base.robot_pos[1], near.x - base.robot_pos[0])
            omega = np.clip(2.0 * ((target - base.robot_theta + np.pi) % (2 * np.pi) - np.pi), -1, 1)
            omega = omega if rng.random() < 0.8 else rng.uniform(-1, 1)
            _, _, term, trunc, info = env.step(np.array([1.0, omega]))
            h, _ = flt.barriers(base.robot_pos, base.robot_theta, obstacles)
            assert np.all(h >= (1 - gamma_dt) * h_prev - 1e-9)
            assert not info["collision"]
            h_prev = h
            if term or trunc:
                break
