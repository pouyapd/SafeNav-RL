"""
obstacle_map.py

Randomized circular-obstacle layouts for the navigation environment, plus the
geometric queries the environment, the safety filter and the range sensor need
(clearance, collision, risk score and ray casting).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Obstacle:
    """A circular obstacle in the 2D workspace."""
    x: float
    y: float
    radius: float

    def distance_to_point(self, px: float, py: float) -> float:
        """Returns the distance from the obstacle surface to the point."""
        center_dist = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return float(center_dist - self.radius)

    def is_colliding(self, px: float, py: float, robot_radius: float) -> bool:
        return self.distance_to_point(px, py) < robot_radius


class ObstacleMap:
    """
    Manages obstacle placement with domain randomization.

    Obstacles keep a clearance from each other, from the robot start and from
    the goal, so every generated start and goal is collision-free. (A free path
    between them is likely but not guaranteed.)
    """

    def __init__(
        self,
        workspace_size: float = 10.0,
        min_obstacles: int = 2,
        max_obstacles: int = 6,
        obstacle_radius_range: Tuple[float, float] = (0.3, 0.8),
        min_clearance: float = 1.0,
    ):
        self.workspace_size = workspace_size
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.min_clearance = min_clearance

        self.obstacles: List[Obstacle] = []
        self._centers = np.zeros((0, 2))
        self._radii = np.zeros(0)

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        rng: np.random.Generator,
    ) -> List[Obstacle]:
        """
        Generate a new random obstacle layout.

        Args:
            robot_pos: Robot starting position [x, y]
            goal_pos: Goal position [x, y]
            rng: NumPy random generator (for reproducibility)

        Returns:
            List of placed Obstacle objects
        """
        n_obstacles = rng.integers(self.min_obstacles, self.max_obstacles + 1)
        self.obstacles = []

        max_attempts = 200
        for _ in range(n_obstacles):
            for _ in range(max_attempts):
                radius = rng.uniform(*self.obstacle_radius_range)
                margin = radius + self.min_clearance
                x = rng.uniform(margin, self.workspace_size - margin)
                y = rng.uniform(margin, self.workspace_size - margin)

                candidate = Obstacle(float(x), float(y), float(radius))

                # Check clearance from robot start
                if candidate.distance_to_point(*robot_pos) < self.min_clearance + 0.5:
                    continue

                # Check clearance from goal
                if candidate.distance_to_point(*goal_pos) < self.min_clearance + 0.5:
                    continue

                # Check clearance from other obstacles
                if any(
                    np.sqrt((o.x - x) ** 2 + (o.y - y) ** 2) < (o.radius + radius + self.min_clearance)
                    for o in self.obstacles
                ):
                    continue

                self.obstacles.append(candidate)
                break
            # If an obstacle cannot be placed after max_attempts it is skipped.

        self.set_obstacles(self.obstacles)
        return self.obstacles

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Replace the obstacle list (used by generate() and by tests)."""
        self.obstacles = list(obstacles)
        if self.obstacles:
            self._centers = np.array([[o.x, o.y] for o in self.obstacles], dtype=float)
            self._radii = np.array([o.radius for o in self.obstacles], dtype=float)
        else:
            self._centers = np.zeros((0, 2))
            self._radii = np.zeros(0)

    # ── Geometric queries ───────────────────────────────────────────────────

    def surface_distances(self, px: float, py: float) -> np.ndarray:
        """Distance from (px, py) to every obstacle surface (negative inside)."""
        if len(self._radii) == 0:
            return np.zeros(0)
        d = np.hypot(self._centers[:, 0] - px, self._centers[:, 1] - py)
        return d - self._radii

    def min_surface_distance(self, px: float, py: float) -> float:
        """Distance from (px, py) to the nearest obstacle surface (inf if none)."""
        d = self.surface_distances(px, py)
        return float(d.min()) if d.size else float("inf")

    def wall_distance(self, px: float, py: float) -> float:
        """Distance from (px, py) to the nearest workspace boundary."""
        w = self.workspace_size
        return float(min(px, py, w - px, w - py))

    def get_risk_score(self, px: float, py: float, safe_distance: float) -> float:
        """
        Compute a scalar collision risk score at position (px, py).

        Risk is 1.0 when touching an obstacle surface and decays linearly to 0
        at safe_distance. Used in reward shaping and as an observation feature.

        Args:
            px, py: Query position
            safe_distance: Distance at which risk becomes 0

        Returns:
            Risk score in [0, 1]
        """
        if not self.obstacles or safe_distance <= 0:
            return 0.0
        min_dist = self.min_surface_distance(px, py)
        risk = max(0.0, 1.0 - min_dist / safe_distance)
        return float(np.clip(risk, 0.0, 1.0))

    def check_collision(self, px: float, py: float, robot_radius: float) -> bool:
        """Returns True if a robot of radius robot_radius at (px, py) touches an obstacle."""
        d = self.surface_distances(px, py)
        return bool(d.size and np.any(d < robot_radius))

    def is_out_of_bounds(self, px: float, py: float, robot_radius: float) -> bool:
        """Returns True if the robot body leaves the workspace."""
        margin = robot_radius
        return bool(
            px < margin or px > self.workspace_size - margin
            or py < margin or py > self.workspace_size - margin
        )

    def ray_cast(
        self,
        origin: np.ndarray,
        angles: np.ndarray,
        max_range: float,
        include_walls: bool = True,
    ) -> np.ndarray:
        """
        Simulated range sensor: distance along each ray to the first obstacle
        or workspace wall, capped at max_range.

        Args:
            origin: Ray origin [x, y] (robot centre).
            angles: World-frame ray directions in radians, shape (K,).
            max_range: Maximum sensing distance (metres).
            include_walls: Also detect the workspace boundary.

        Returns:
            Array of shape (K,) with distances in [0, max_range].
        """
        angles = np.asarray(angles, dtype=float)
        ux, uy = np.cos(angles), np.sin(angles)                         # (K,)
        dist = np.full(angles.shape, float(max_range))

        if len(self._radii):
            # Solve |o + t u - c|^2 = R^2 for t >= 0 (smallest root).
            ocx = origin[0] - self._centers[:, 0]                        # (M,)
            ocy = origin[1] - self._centers[:, 1]
            b = ux[:, None] * ocx[None, :] + uy[:, None] * ocy[None, :]  # (K, M)
            c = ocx ** 2 + ocy ** 2 - self._radii ** 2                   # (M,)
            disc = b ** 2 - c[None, :]
            hit = disc >= 0
            sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
            t_near = -b - sqrt_disc
            # Origin inside an obstacle (c < 0): the reading is 0.
            t = np.where(c[None, :] < 0, 0.0, t_near)
            t = np.where(hit & (t >= 0), t, np.inf)
            dist = np.minimum(dist, t.min(axis=1))

        if include_walls:
            w = self.workspace_size
            with np.errstate(divide="ignore", invalid="ignore"):
                tx = np.where(ux > 0, (w - origin[0]) / ux, np.where(ux < 0, -origin[0] / ux, np.inf))
                ty = np.where(uy > 0, (w - origin[1]) / uy, np.where(uy < 0, -origin[1] / uy, np.inf))
            t_wall = np.maximum(np.minimum(tx, ty), 0.0)
            dist = np.minimum(dist, t_wall)

        return np.clip(dist, 0.0, max_range)

    def nearest_obstacle(self, px: float, py: float) -> Optional[Obstacle]:
        """The obstacle whose surface is closest to (px, py), or None."""
        d = self.surface_distances(px, py)
        return self.obstacles[int(np.argmin(d))] if d.size else None
