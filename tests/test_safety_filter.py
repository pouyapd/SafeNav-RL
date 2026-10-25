"""
test_safety_filter.py — the discrete-time CBF speed filter.

The key property: with the filter in the loop, no barrier ever becomes negative,
whatever the policy does (forward invariance under the assumptions listed in
safenav/safety/cbf_filter.py).
"""

import numpy as np
import pytest

from safenav.config import get_stage
from safenav.env.navigation_env import NavigationEnv
from safenav.env.obstacle_map import Obstacle
from safenav.factory import make_env
from safenav.safety.cbf_filter import CBFSafetyFilter, SafetyFilterWrapper, SafetyLayer


@pytest.fixture
def flt():
    return CBFSafetyFilter(dt=0.1, robot_radius=0.3, safe_distance=0.3, gamma=0.5,
                           workspace_size=10.0, wall_margin=0.1)


def test_no_change_when_far_from_everything(flt):
    action, info = flt.filter([1.0, 0.3], (5.0, 5.0), 0.0, [Obstacle(9.0, 9.0, 0.3)])
    np.testing.assert_allclose(action, [1.0, 0.3])
    assert not info["intervened"]


def test_speed_limit_matches_formula_head_on(flt):
    # h = 2.0 - 0.5 - 0.3 - 0.3 = 0.9, closing rate a = 1 -> v <= gamma * h = 0.45
    action, info = flt.filter([1.0, 0.0], (5.0, 5.0), 0.0, [Obstacle(7.0, 5.0, 0.5)])
    assert action[0] == pytest.approx(0.45)
    assert info["intervened"]


def test_speed_limit_matches_formula_at_an_angle(flt):
    # heading 60 degrees off the obstacle direction: a = cos(60°) = 0.5 -> v <= 0.45 / 0.5 = 0.9
    action, _ = flt.filter([1.0, 0.0], (5.0, 5.0), np.deg2rad(60), [Obstacle(7.0, 5.0, 0.5)])
    assert action[0] == pytest.approx(0.9)


def test_moving_away_is_not_limited(flt):
    action, _ = flt.filter([1.0, 0.0], (5.0, 5.0), np.pi, [Obstacle(6.2, 5.0, 0.5)])
    assert action[0] == pytest.approx(1.0)


def test_omega_is_never_changed(flt):
    rng = np.random.default_rng(0)
    for _ in range(200):
        omega = rng.uniform(-1, 1)
        action, _ = flt.filter([rng.uniform(0, 1), omega], rng.uniform(1, 9, 2), rng.uniform(-np.pi, np.pi),
                               [Obstacle(*rng.uniform(1, 9, 2), 0.5)])
        assert action[1] == omega


def test_inside_margin_and_approaching_stops(flt):
    # h < 0 (an assumption was violated): approaching -> v = 0, moving away -> allowed
    obs = [Obstacle(6.0, 5.0, 0.5)]           # h = 1.0 - 0.5 - 0.6 = -0.1
    assert flt.filter([1.0, 0.0], (5.0, 5.0), 0.0, obs)[0][0] == 0.0
    assert flt.filter([1.0, 0.0], (5.0, 5.0), np.pi, obs)[0][0] == pytest.approx(1.0)


def test_wall_constraint(flt):
    # 0.6 m from the left wall, driving straight at it: h = 0.6 - 0.4 = 0.2 -> v <= 0.1
    action, _ = flt.filter([1.0, 0.0], (0.6, 5.0), np.pi, [])
    assert action[0] == pytest.approx(0.1)


def test_rejects_gamma_dt_above_one():
    with pytest.raises(ValueError):
        CBFSafetyFilter(dt=0.1, gamma=20.0)


def test_forward_invariance_under_adversarial_policy(config):
    """Drive at full speed toward the nearest obstacle with random turning; barriers stay >= 0."""
    flt = CBFSafetyFilter.from_config(config)
    gamma_dt = flt.gamma * flt.dt
    rng = np.random.default_rng(1)
    env = NavigationEnv(config, task=get_stage(config, "expert"))
    worst = np.inf
    for seed in range(40):
        env.reset(seed=seed)
        obstacles = env.obstacle_map.obstacles
        h_prev, _ = flt.barriers(env.robot_pos, env.robot_theta, obstacles)
        assert h_prev.min() >= 0, "scenario generator must start in the safe set"
        for _ in range(250):
            near = env.obstacle_map.nearest_obstacle(*env.robot_pos)
            target = np.arctan2(near.y - env.robot_pos[1], near.x - env.robot_pos[0])
            omega = np.clip(2.0 * ((target - env.robot_theta + np.pi) % (2 * np.pi) - np.pi), -1, 1)
            omega = omega if rng.random() < 0.8 else rng.uniform(-1, 1)
            safe, _ = flt.filter([1.0, omega], env.robot_pos, env.robot_theta, obstacles)
            _, _, terminated, truncated, info = env.step(safe)
            h, _ = flt.barriers(env.robot_pos, env.robot_theta, obstacles)
            # discrete-time CBF bound h_{t+1} >= (1 - gamma*dt) h_t for every barrier
            assert np.all(h >= (1 - gamma_dt) * h_prev - 1e-9)
            assert not info["collision"]
            worst = min(worst, h.min())
            h_prev = h
            if terminated or truncated:
                break
    assert worst >= -1e-9


def test_gap_between_samples_is_bounded(config):
    """
    The guarantee is for the sampled states. Between two samples the simulated robot
    moves on a straight segment of length L <= v_max * dt; if both ends keep distance
    rho = R + r + d_safe from an obstacle centre, every point of the segment keeps at
    least sqrt(rho^2 - (L/2)^2) (worst case: the segment is a chord). Check this along
    every step of the adversarial policy, and that the gap stays below 1.4 mm.
    """
    flt = CBFSafetyFilter.from_config(config)
    rng = np.random.default_rng(2)
    env = NavigationEnv(config, task=get_stage(config, "expert"))
    worst_gap = 0.0
    for seed in range(20):
        env.reset(seed=seed)
        obstacles = env.obstacle_map.obstacles
        for _ in range(250):
            near = env.obstacle_map.nearest_obstacle(*env.robot_pos)
            target = np.arctan2(near.y - env.robot_pos[1], near.x - env.robot_pos[0])
            omega = np.clip(2.0 * ((target - env.robot_theta + np.pi) % (2 * np.pi) - np.pi), -1, 1)
            omega = omega if rng.random() < 0.8 else rng.uniform(-1, 1)
            p0 = env.robot_pos.copy()
            safe, _ = flt.filter([1.0, omega], env.robot_pos, env.robot_theta, obstacles)
            _, _, terminated, truncated, _ = env.step(safe)
            seg = env.robot_pos - p0
            length = float(np.linalg.norm(seg))
            assert length <= flt.v_max * flt.dt + 1e-12
            for o in obstacles:
                c = np.array([o.x, o.y])
                s = 0.0 if length == 0 else float(np.clip(np.dot(c - p0, seg) / length**2, 0.0, 1.0))
                closest = float(np.linalg.norm(p0 + s * seg - c))
                rho = o.radius + flt.robot_radius + flt.safe_distance
                assert closest >= np.sqrt(rho**2 - (length / 2) ** 2) - 1e-9
                worst_gap = max(worst_gap, rho - closest)
            if terminated or truncated:
                break
    rho_min = 0.3 + flt.robot_radius + flt.safe_distance         # smallest generated obstacle radius is 0.3 m
    bound = rho_min - np.sqrt(rho_min**2 - (flt.v_max * flt.dt / 2) ** 2)
    assert bound < 1.4e-3
    assert worst_gap <= bound + 1e-9


def test_default_margin_matches_config(config):
    assert CBFSafetyFilter().safe_distance == config["safety"]["safe_distance"]
    assert CBFSafetyFilter.from_config({}).safe_distance == config["safety"]["safe_distance"]


def test_wrapper_filters_and_reports(config):
    env = make_env(config, task=get_stage(config, "beginner"), safety_filter=True)
    assert isinstance(env, SafetyFilterWrapper)
    env.reset(seed=0)
    base = env.unwrapped
    base.robot_pos = np.array([5.0, 5.0])
    base.robot_theta = 0.0
    base.obstacle_map.set_obstacles([Obstacle(7.0, 5.0, 0.5)])
    _, _, _, _, info = env.step(np.array([1.0, 0.25]))
    assert info["safety_intervened"]
    assert info["executed_action"][0] == pytest.approx(0.45)
    assert info["executed_action"][1] == pytest.approx(0.25)
    np.testing.assert_allclose(info["nominal_action"], [1.0, 0.25])


def test_legacy_safety_layer_interface(config):
    sl = SafetyLayer(config)
    action = np.array([1.0, 0.0])
    safe = sl.project_action(action, np.array([5.0, 5.0]), 0.0, [Obstacle(5.5, 5.0, 0.3)])
    assert safe[0] < action[0]
    assert 0.0 <= safe[0] <= 1.0


def _rollout_rewards(env, seed, actions):
    env.reset(seed=seed)
    out = []
    for a in actions:
        _, r, term, trunc, info = env.step(a)
        out.append((r, info.get("correction_penalty", 0.0)))
        if term or trunc:
            break
    return out


def test_correction_penalty_off_by_default_leaves_rewards_unchanged(config):
    rng = np.random.default_rng(3)
    actions = [rng.normal([0.6, 0.0], 0.8) for _ in range(150)]
    task = get_stage(config, "expert")
    trained = _rollout_rewards(make_env(config, task=task, safety_filter=True, training=True), 7, actions)
    evaluated = _rollout_rewards(make_env(config, task=task, safety_filter=True), 7, actions)
    assert trained == evaluated
    assert all(p == 0.0 for _, p in trained)


def test_correction_penalty_value_and_training_only(config):
    from safenav.config import apply_overrides
    cfg = apply_overrides(config, {"safety.correction_penalty": 2.0})
    task = get_stage(cfg, "beginner")
    train_env = make_env(cfg, task=task, safety_filter=True, training=True)
    eval_env = make_env(cfg, task=task, safety_filter=True)
    for env in (train_env, eval_env):
        env.reset(seed=0)
        base = env.unwrapped
        base.robot_pos = np.array([5.0, 5.0])
        base.robot_theta = 0.0
        base.obstacle_map.set_obstacles([Obstacle(7.0, 5.0, 0.5)])
    raw = np.array([1.3, 0.25])            # v above the bound and above the filter limit (0.45)
    _, r_train, _, _, info_train = train_env.step(raw)
    _, r_eval, _, _, info_eval = eval_env.step(raw)
    expected = 2.0 * ((1.3 - 0.45) ** 2 + 0.0 ** 2)
    assert info_train["correction_penalty"] == pytest.approx(expected)
    assert r_eval - r_train == pytest.approx(expected)
    assert info_eval["correction_penalty"] == 0.0


def test_negative_correction_penalty_rejected(config):
    from safenav.config import ConfigError, apply_overrides, validate_config
    with pytest.raises(ConfigError):
        validate_config(apply_overrides(config, {"safety.correction_penalty": -1.0}))
    with pytest.raises(ValueError):
        SafetyFilterWrapper(NavigationEnv(config), CBFSafetyFilter(), correction_penalty=-0.5)
