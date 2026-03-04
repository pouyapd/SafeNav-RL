"""
actor_critic.py

PPO Actor-Critic neural network implementation.

Architecture:
  - Shared feature extractor (MLP)
  - Actor head: outputs mean of Gaussian action distribution
  - Critic head: outputs scalar state value V(s)
  - Log std: learned as a standalone parameter (not state-dependent)

The Gaussian policy allows continuous action spaces and provides
natural entropy bonus via the log-probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer sizes
        output_dim: Output dimension
        activation: Hidden layer activation ('tanh' or 'relu')
        output_activation: Optional activation on output layer
    
    Returns:
        nn.Sequential module
    """
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

    layers = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))

    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation == "softmax":
        layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    The actor and critic share no weights — they are separate MLPs.
    This is the standard approach for PPO, providing stable training.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        net_cfg = config.get("network", {})
        obs_dim = net_cfg.get("obs_dim", 7)
        action_dim = net_cfg.get("action_dim", 2)
        hidden_dims = net_cfg.get("hidden_dims", [256, 256])
        activation = net_cfg.get("activation", "tanh")
        log_std_init = net_cfg.get("log_std_init", -0.5)
        self.log_std_min = net_cfg.get("log_std_min", -3.0)
        self.log_std_max = net_cfg.get("log_std_max", 1.0)

        # Actor: maps observation → action mean
        self.actor = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=activation,
        )

        # Critic: maps observation → scalar value
        self.critic = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

        # Log std as a learned parameter (not state-dependent)
        # Shape: (action_dim,)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, dtype=torch.float32)
        )

        # Initialize weights with orthogonal initialization (common for PPO)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization for stable training."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        # Smaller gain for actor output layer
        if isinstance(self.actor[-1], nn.Linear):
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """
        Compute the action distribution given an observation.
        
        Returns a Normal distribution with state-dependent mean
        and state-independent (learned) std.
        """
        mean = self.actor(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the critic's value estimate V(s)."""
        return self.critic(obs).squeeze(-1)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: Critic value estimate
        """
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        value = self.get_value(obs)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy for a batch of (obs, action) pairs.
        Used during PPO update.
        
        Returns:
            log_probs: Log probability of each action
            entropy: Distribution entropy (for entropy bonus)
            values: Critic estimates
        """
        dist = self.get_distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.get_value(obs)
        return log_probs, entropy, values

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean action (no sampling) for deterministic evaluation.
        """
        return self.actor(obs)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
