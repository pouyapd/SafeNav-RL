r"""
perception.py — what the recovery planner knows (round 4; opt-in).

The CBF filter always uses the true state and the true map (its guarantee needs them). These
classes change only what the *recovery controller* knows, so that the robustness of recovery to
imperfect information can be measured without touching safety:

  ExactInfo            true state and true obstacles (the round-3 setting; the default)
  NoisyLocalisation    state estimate = true state + Gaussian noise, fresh every step
  MapError             a fixed, wrong map per episode: centres shifted, radii scaled, obstacles dropped
  SensedMap            "known when seen": an obstacle enters the map once a range beam of the
                       robot's own (noisy) sensor hits it; its circle is then known exactly.
                       This idealises data association and shape estimation; it is not a SLAM system.

Each object is bound to an episode with bind(env, seed); obstacles() returns the planner's map and
update(obs) returns True when that map changed (the controller then rebuilds its model and fields).
All randomness comes from a generator seeded per episode, so every method sees the same errors.
"""

import numpy as np

from safenav.env.obstacle_map import Obstacle


class ExactInfo:
    name = "exact"

    def bind(self, env, seed=None):
        self.base = env.unwrapped

    def state(self):
        return self.base.robot_pos.copy(), float(self.base.robot_theta)

    def obstacles(self):
        return self.base.obstacle_map.obstacles

    def update(self, obs) -> bool:
        return False

    @property
    def exact_map(self):
        return True


class NoisyLocalisation(ExactInfo):
    name = "localisation_error"

    def __init__(self, sigma_pos=0.05, sigma_theta=0.05):
        self.sigma_pos, self.sigma_theta = float(sigma_pos), float(sigma_theta)

    def bind(self, env, seed=None):
        super().bind(env, seed)
        self.rng = np.random.default_rng([int(seed or 0), 41])

    def state(self):
        p = self.base.robot_pos + self.rng.normal(0.0, self.sigma_pos, 2)
        th = float(self.base.robot_theta + self.rng.normal(0.0, self.sigma_theta))
        return p, float((th + np.pi) % (2 * np.pi) - np.pi)


class MapError(ExactInfo):
    name = "map_error"

    def __init__(self, sigma_centre=0.10, radius_scale=(0.85, 1.15), p_missing=0.10):
        self.sigma_centre, self.radius_scale, self.p_missing = float(sigma_centre), radius_scale, float(p_missing)

    def bind(self, env, seed=None):
        super().bind(env, seed)
        rng = np.random.default_rng([int(seed or 0), 42])
        out = []
        for o in self.base.obstacle_map.obstacles:
            dx, dy = rng.normal(0.0, self.sigma_centre, 2)
            s = rng.uniform(*self.radius_scale)
            keep = rng.uniform() >= self.p_missing
            if keep:
                out.append(Obstacle(o.x + dx, o.y + dy, o.radius * s))
        self._obs = out

    def obstacles(self):
        return self._obs

    @property
    def exact_map(self):
        return False


class SensedMap(ExactInfo):
    """Obstacles become known when a beam of the noisy range sensor hits them (within `assoc_m`)."""

    name = "sensed_map"

    def __init__(self, assoc_m=0.15):
        self.assoc_m = float(assoc_m)

    def bind(self, env, seed=None):
        super().bind(env, seed)
        b = self.base
        self.known = np.zeros(len(b.obstacle_map.obstacles), dtype=bool)
        self.offsets = np.asarray(b.beam_offsets, dtype=float)
        self.max_range = float(b.max_range)
        self._obs = []

    def update(self, obs) -> bool:
        b = self.base
        if len(self.offsets) == 0 or len(self.known) == 0:
            return False
        ranges = np.asarray(obs[7:7 + len(self.offsets)], dtype=float) * self.max_range   # the robot's own noisy readings
        hit = ranges < 0.999 * self.max_range
        if not np.any(hit):
            return False
        ang = b.robot_theta + self.offsets[hit]
        pts = b.robot_pos[None, :] + ranges[hit, None] * np.column_stack([np.cos(ang), np.sin(ang)])
        ob = b.obstacle_map.obstacles
        c = np.array([[o.x, o.y] for o in ob])
        rad = np.array([o.radius for o in ob])
        surf = np.abs(np.hypot(pts[:, None, 0] - c[None, :, 0], pts[:, None, 1] - c[None, :, 1]) - rad[None, :])
        k = surf.argmin(axis=1)
        new = np.zeros_like(self.known)
        new[k[surf[np.arange(len(k)), k] <= self.assoc_m]] = True
        new &= ~self.known
        if not np.any(new):
            return False
        self.known |= new
        self._obs = [o for o, kn in zip(ob, self.known) if kn]
        return True

    def obstacles(self):
        return self._obs

    @property
    def exact_map(self):
        return False


INFO = {"exact": ExactInfo, "localisation_error": NoisyLocalisation, "map_error": MapError, "sensed_map": SensedMap}
