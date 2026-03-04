"""
obstacle_map.py

Generates randomized obstacle layouts for the navigation environment.
Supports domain randomization for sim-to-real transfer robustness.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Obstacle:
    """A circular obstacle in the 2D workspace."""
    x: float
    y: float
    radius: float

    def distance_to_point(self, px: float, py: float) -> float:
        """Returns the distance from the obstacle surface to the point."""
        center_dist = np.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return center_dist - self.radius

    def is_colliding(self, px: float, py: float, robot_radius: float) -> bool:
        return self.distance_to_point(px, py) < robot_radius


class ObstacleMap:
    """
    Manages obstacle placement with domain randomization.
    
    Ensures obstacles do not overlap each other, the robot start,
    or the goal position — critical for generating solvable episodes.
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
            placed = False
            for _ in range(max_attempts):
                radius = rng.uniform(*self.obstacle_radius_range)
                margin = radius + self.min_clearance
                x = rng.uniform(margin, self.workspace_size - margin)
                y = rng.uniform(margin, self.workspace_size - margin)

                candidate = Obstacle(x, y, radius)

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
                placed = True
                break

            # If we can't place an obstacle after max_attempts, skip it
            if not placed:
                continue

        return self.obstacles

    def get_risk_score(self, px: float, py: float, safe_distance: float) -> float:
        """
        Compute a scalar collision risk score at position (px, py).
        
        Risk is 1.0 when touching an obstacle surface and decays to 0
        at safe_distance. This is used in reward shaping and as an
        observation feature.
        
        Args:
            px, py: Query position
            safe_distance: Distance at which risk becomes 0
        
        Returns:
            Risk score in [0, 1]
        """
        if not self.obstacles:
            return 0.0

        min_dist = min(
            obs.distance_to_point(px, py) for obs in self.obstacles
        )
        # Clamp and normalize
        risk = max(0.0, 1.0 - min_dist / safe_distance)
        return float(np.clip(risk, 0.0, 1.0))

    def check_collision(self, px: float, py: float, robot_radius: float) -> bool:
        """Returns True if position (px, py) collides with any obstacle."""
        return any(obs.is_colliding(px, py, robot_radius) for obs in self.obstacles)

    def is_out_of_bounds(self, px: float, py: float, robot_radius: float) -> bool:
        """Returns True if position is outside the workspace boundary."""
        margin = robot_radius
        return (
            px < margin or px > self.workspace_size - margin
            or py < margin or py > self.workspace_size - margin
        )
