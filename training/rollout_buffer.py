"""
rollout_buffer.py

Experience buffer for collecting and processing PPO rollouts.

Stores transitions (obs, action, reward, done, value, log_prob)
and computes Generalized Advantage Estimation (GAE) for the PPO update.

GAE Reference:
  Schulman et al. (2015), "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation"
"""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """
    Fixed-size circular buffer for PPO rollout data.
    
    After collection, call compute_advantages() before iterating
    to get training batches.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate numpy arrays for speed
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)

        # Computed after collection
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add a single transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_advantages(self, last_value: float, last_done: bool):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE(λ) trades off bias vs variance:
          - λ=0: low variance, high bias (TD error)
          - λ=1: high variance, low bias (Monte Carlo)
          - λ=0.95: standard choice for PPO
        
        Also computes returns = advantages + values (target for critic).
        
        Args:
            last_value: V(s_T) — value of the final state in the rollout
            last_done: Whether the final step was terminal
        """
        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            # TD error: δ = r + γ·V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )

            # GAE: A_t = δ_t + (γλ)·A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Generator:
        """
        Iterate over the buffer in random mini-batches.
        
        Yields:
            Tuple of tensors: (obs, actions, log_probs, advantages, returns)
        """
        assert self.full or self.pos > 0, "Buffer is empty."
        size = self.buffer_size if self.full else self.pos

        # Normalize advantages (reduces variance, stabilizes training)
        adv = self.advantages[:size]
        adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
        normalized_adv = (adv - adv_mean) / adv_std

        indices = np.random.permutation(size)
        start = 0
        while start < size:
            batch_idx = indices[start: start + batch_size]
            start += batch_size

            yield (
                torch.tensor(self.observations[batch_idx], device=self.device),
                torch.tensor(self.actions[batch_idx], device=self.device),
                torch.tensor(self.log_probs[batch_idx], device=self.device),
                torch.tensor(normalized_adv[batch_idx], device=self.device),
                torch.tensor(self.returns[batch_idx], device=self.device),
            )

    def reset(self):
        """Reset the buffer for the next rollout."""
        self.pos = 0
        self.full = False
