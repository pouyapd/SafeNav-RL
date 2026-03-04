"""
curriculum.py

Progressive difficulty scheduler for curriculum learning.

The agent starts in easy environments (few obstacles, nearby goals)
and advances to harder stages as its success rate improves.
This is a well-established technique for accelerating RL training
on tasks that are too hard to learn from scratch.
"""

from collections import deque
from typing import Dict, Any, List


class CurriculumScheduler:
    """
    Manages stage transitions based on rolling success rate.
    
    Each stage defines the environment difficulty parameters
    (obstacle count, goal distance, episode length).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: The 'curriculum' section of default_config.yaml
        """
        self.enabled = config.get("enabled", True)
        self.advance_threshold = config.get("advance_threshold", 0.70)
        self.window_size = config.get("window_size", 100)
        self.stages: List[Dict] = config.get("stages", [])

        if not self.stages:
            raise ValueError("Curriculum config must define at least one stage.")

        self.current_stage_idx = 0
        # Rolling window of episode outcomes (True=success, False=fail/collision)
        self._outcome_window: deque = deque(maxlen=self.window_size)

    @property
    def current_stage(self) -> Dict[str, Any]:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.get("name", f"stage_{self.current_stage_idx}")

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def record_outcome(self, success: bool) -> bool:
        """
        Record an episode outcome and check if we should advance.
        
        Args:
            success: Whether the episode ended with goal reached
        
        Returns:
            True if the stage just advanced
        """
        if not self.enabled:
            return False

        self._outcome_window.append(float(success))

        if (
            len(self._outcome_window) >= self.window_size
            and not self.is_final_stage
            and self.success_rate >= self.advance_threshold
        ):
            self.current_stage_idx += 1
            self._outcome_window.clear()
            return True

        return False

    @property
    def success_rate(self) -> float:
        if not self._outcome_window:
            return 0.0
        return sum(self._outcome_window) / len(self._outcome_window)

    def get_env_params(self) -> Dict[str, Any]:
        """
        Returns the environment parameters for the current stage.
        These are passed to ObstacleMap and the navigation env.
        """
        stage = self.current_stage
        return {
            "min_obstacles": stage["min_obstacles"],
            "max_obstacles": stage["max_obstacles"],
            "min_goal_dist": stage["min_goal_dist"],
            "max_goal_dist": stage["max_goal_dist"],
            "max_steps": stage["max_steps"],
        }

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(stage={self.stage_name} [{self.current_stage_idx+1}/"
            f"{len(self.stages)}], success_rate={self.success_rate:.2f})"
        )
