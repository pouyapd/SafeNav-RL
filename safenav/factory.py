"""
factory.py

One place that builds environments the same way for training, evaluation,
benchmarks and tests.
"""

from typing import Any, Dict, Optional

import gymnasium as gym

from safenav.env.curriculum import CurriculumScheduler
from safenav.env.navigation_env import NavigationEnv
from safenav.safety.cbf_filter import CBFSafetyFilter, SafetyFilterWrapper


def make_env(
    config: Dict[str, Any],
    task: Optional[Dict[str, Any]] = None,
    curriculum: Optional[CurriculumScheduler] = None,
    safety_filter: Optional[bool] = None,
    training: bool = False,
) -> gym.Env:
    """
    Build a NavigationEnv, optionally wrapped with the CBF safety filter.

    Args:
        config: Validated configuration dictionary.
        task: Fixed difficulty (a curriculum stage dict); ignored if `curriculum` is given.
        curriculum: Curriculum scheduler used during training.
        safety_filter: Apply the CBF filter. Default: config["safety"]["enabled"].
        training: Training environment. Only then is safety.correction_penalty
            (default 0) applied, so evaluation rewards are never penalised.
    """
    env = NavigationEnv(config, curriculum=curriculum, task=task)
    use_filter = config.get("safety", {}).get("enabled", True) if safety_filter is None else safety_filter
    if use_filter:
        flt = CBFSafetyFilter.from_config(config)
        flt.enabled = True
        penalty = float(config.get("safety", {}).get("correction_penalty", 0.0)) if training else 0.0
        env = SafetyFilterWrapper(env, flt, correction_penalty=penalty)
    return env
