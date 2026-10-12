"""
policy.py

Common interface for everything that can drive the robot (trained PPO agents
and hand-designed baselines), plus checkpoint loading.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import torch

from safenav.agents.actor_critic import ActorCritic


class Policy(Protocol):
    """Anything with a name, reset() and act(obs) -> [v, omega]."""

    name: str

    def reset(self, seed: Optional[int] = None) -> None:
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a SafeNav-RL checkpoint (.pt) saved by the trainer."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on file contents
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{path} is not a SafeNav-RL checkpoint (no 'model_state' entry)")
    return ckpt


class PPOPolicy:
    """Wraps a trained ActorCritic as a Policy."""

    def __init__(self, model: ActorCritic, deterministic: bool = True, name: str = "PPO"):
        self.model = model.eval()
        self.deterministic = deterministic
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], deterministic: bool = True,
                        name: Optional[str] = None) -> "PPOPolicy":
        ckpt = load_checkpoint(path)
        config = ckpt.get("config", {})
        obs_dim = ckpt.get("obs_dim")
        if obs_dim is None:
            # Checkpoints from version 0.1 had no obs_dim entry; infer it from the weights.
            obs_dim = ckpt["model_state"]["actor.0.weight"].shape[1]
        model = ActorCritic(config, obs_dim=int(obs_dim))
        model.load_state_dict(ckpt["model_state"])
        policy = cls(model, deterministic=deterministic, name=name or "PPO")
        policy.checkpoint = ckpt
        return policy

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None and not self.deterministic:
            torch.manual_seed(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.deterministic:
                action = self.model.get_deterministic_action(obs_t)
            else:
                action, _, _ = self.model(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float64)
