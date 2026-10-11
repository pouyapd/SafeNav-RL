"""
The version 0.1 safety layer and the current CBF filter under the same adversarial test.

The test is the one in tests/test_safety_filter.py: 40 expert scenarios, up to 250
steps each, full speed toward the nearest obstacle with random turns. The v0.1
projection rule is copied unchanged below from models/safety_layer.py at commit
7e42c9d (SafetyLayer.project_action); both rules run in the current simulator.

    python scripts/v01_safety_layer_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from safenav.config import get_stage, load_config  # noqa: E402
from safenav.env.navigation_env import NavigationEnv  # noqa: E402
from safenav.safety.cbf_filter import CBFSafetyFilter  # noqa: E402


def v01_project_action(action, robot_pos, robot_theta, obstacles, safe_distance, cbf_gamma, robot_radius):
    """SafetyLayer.project_action of version 0.1 (unchanged apart from the arguments)."""
    if not obstacles:
        return action
    v, omega = float(action[0]), float(action[1])
    safe_v = v
    for obs in obstacles:
        dist_centers = np.sqrt((robot_pos[0] - obs.x) ** 2 + (robot_pos[1] - obs.y) ** 2)
        dist_surface = dist_centers - obs.radius - robot_radius
        h = dist_surface - safe_distance
        if h >= 0:
            continue
        dx = obs.x - robot_pos[0]
        dy = obs.y - robot_pos[1]
        direction = np.array([dx, dy])
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
        vel_direction = np.array([np.cos(robot_theta), np.sin(robot_theta)])
        approaching_component = np.dot(vel_direction, direction)
        if approaching_component <= 0:
            continue
        max_safe_v = cbf_gamma * max(h, 0.0) / (approaching_component + 1e-6)
        safe_v = min(safe_v, max_safe_v)
    safe_v = float(np.clip(safe_v, 0.0, 1.0))
    return np.array([safe_v, omega], dtype=np.float32)


def run(rule: str, margin: float, scenarios: int = 40, steps: int = 250) -> dict:
    cfg = load_config(overrides={"safety.safe_distance": margin})
    flt = CBFSafetyFilter.from_config(cfg)
    env = NavigationEnv(cfg, task=get_stage(cfg, "expert"))
    rng = np.random.default_rng(1)
    worst, obstacle_hits, left_workspace = np.inf, 0, 0
    for seed in range(scenarios):
        env.reset(seed=seed)
        obstacles = env.obstacle_map.obstacles
        for _ in range(steps):
            near = env.obstacle_map.nearest_obstacle(*env.robot_pos)
            target = np.arctan2(near.y - env.robot_pos[1], near.x - env.robot_pos[0])
            omega = np.clip(2.0 * ((target - env.robot_theta + np.pi) % (2 * np.pi) - np.pi), -1, 1)
            omega = omega if rng.random() < 0.8 else rng.uniform(-1, 1)
            if rule == "v0.1":
                action = v01_project_action(np.array([1.0, omega]), env.robot_pos, env.robot_theta, obstacles,
                                            margin, flt.gamma, flt.robot_radius)
            else:
                action, _ = flt.filter([1.0, omega], env.robot_pos, env.robot_theta, obstacles)
            _, _, terminated, truncated, info = env.step(action)
            h, _ = flt.barriers(env.robot_pos, env.robot_theta, obstacles)
            worst = min(worst, float(h[: len(obstacles)].min()))       # obstacle barriers only
            if info["collision"]:
                obstacle_hits += bool(info["obstacle_collision"])
                left_workspace += not info["obstacle_collision"]
            if terminated or truncated:
                break
    return {"rule": rule, "margin": margin, "deepest_entry_m": max(0.0, -worst),
            "obstacle_collisions": obstacle_hits, "left_workspace": left_workspace, "episodes": scenarios}


def main() -> int:
    print(f"{'rule':<6} {'margin':>6} {'deepest entry into margin':>26} {'obstacle hits':>14} {'left workspace':>15}")
    for margin in (0.8, 0.3):
        for rule in ("v0.1", "cbf"):
            r = run(rule, margin)
            print(f"{r['rule']:<6} {r['margin']:>6.1f} {r['deepest_entry_m']:>24.3f} m "
                  f"{r['obstacle_collisions']:>11}/{r['episodes']} {r['left_workspace']:>12}/{r['episodes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
