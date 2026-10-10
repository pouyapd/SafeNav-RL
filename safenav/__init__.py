"""
SafeNav-RL: safety-filtered reinforcement learning for 2D robot navigation.

Subpackages:
    safenav.env         navigation environment, obstacle maps, curriculum
    safenav.safety      discrete-time control-barrier-function (CBF) speed filter
    safenav.agents      PPO actor-critic and non-learned baseline controllers
    safenav.training    PPO trainer, rollout buffer, logging and checkpoints
    safenav.evaluation  episode metrics, seeded evaluation and benchmarks
    safenav.viz         figures (trajectories, comparisons, training curves)
"""

__version__ = "0.2.0"
