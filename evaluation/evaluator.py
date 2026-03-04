"""
evaluator.py

Evaluation pipeline for a trained SafeNav-RL policy.

Runs the policy in deterministic mode (no sampling, uses mean action)
and computes research-grade metrics:
  - Success rate
  - Collision rate
  - Average path length (normalized)
  - Average risk score along trajectory
  - Goal reach time

These metrics directly connect to the safety-critical evaluation
methodology used in the SafeTraj-Prototype project.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any

from env.navigation_env import NavigationEnv
from models.actor_critic import ActorCritic
from models.safety_layer import SafetyLayer


@dataclass
class EpisodeResult:
    """Stores the result of a single evaluation episode."""
    success: bool
    collision: bool
    timeout: bool
    total_reward: float
    steps: int
    trajectory: List[np.ndarray]
    risk_scores: List[float]
    goal_pos: np.ndarray
    obstacles: list

    @property
    def path_length(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        return sum(
            np.linalg.norm(self.trajectory[i + 1] - self.trajectory[i])
            for i in range(len(self.trajectory) - 1)
        )

    @property
    def straight_line_distance(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        return float(np.linalg.norm(self.goal_pos - self.trajectory[0]))

    @property
    def path_efficiency(self) -> float:
        """Ratio of straight-line distance to actual path length. 1.0 = optimal."""
        pl = self.path_length
        sld = self.straight_line_distance
        if pl < 1e-6:
            return 0.0
        return float(sld / pl)

    @property
    def mean_risk(self) -> float:
        return float(np.mean(self.risk_scores)) if self.risk_scores else 0.0

    @property
    def max_risk(self) -> float:
        return float(np.max(self.risk_scores)) if self.risk_scores else 0.0


@dataclass
class EvaluationSummary:
    """Aggregated metrics over all evaluation episodes."""
    n_episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_reward: float
    std_reward: float
    mean_path_length: float
    mean_path_efficiency: float
    mean_risk_score: float
    mean_goal_reach_steps: float  # Only for successful episodes
    results: List[EpisodeResult] = field(default_factory=list)

    def print(self):
        print("\n" + "=" * 55)
        print("  Evaluation Summary")
        print("=" * 55)
        print(f"  Episodes:          {self.n_episodes}")
        print(f"  Success Rate:      {self.success_rate:.1%}")
        print(f"  Collision Rate:    {self.collision_rate:.1%}")
        print(f"  Timeout Rate:      {self.timeout_rate:.1%}")
        print(f"  Mean Reward:       {self.mean_reward:.3f} ± {self.std_reward:.3f}")
        print(f"  Mean Path Length:  {self.mean_path_length:.2f} m")
        print(f"  Path Efficiency:   {self.mean_path_efficiency:.2%}")
        print(f"  Mean Risk Score:   {self.mean_risk_score:.4f}")
        if self.mean_goal_reach_steps > 0:
            print(f"  Avg Steps (succ):  {self.mean_goal_reach_steps:.1f}")
        print("=" * 55 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "timeout_rate": self.timeout_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_path_length": self.mean_path_length,
            "mean_path_efficiency": self.mean_path_efficiency,
            "mean_risk_score": self.mean_risk_score,
            "mean_goal_reach_steps": self.mean_goal_reach_steps,
        }


class PolicyEvaluator:
    """
    Runs a trained policy for N evaluation episodes and computes metrics.
    """

    def __init__(
        self,
        env: NavigationEnv,
        model: ActorCritic,
        safety_layer: SafetyLayer,
        device: torch.device,
    ):
        self.env = env
        self.model = model
        self.safety_layer = safety_layer
        self.device = device

    def evaluate(self, n_episodes: int = 50, deterministic: bool = True) -> EvaluationSummary:
        """
        Run evaluation for n_episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: If True, use mean action (no sampling)

        Returns:
            EvaluationSummary with all metrics
        """
        self.model.eval()
        results: List[EpisodeResult] = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)  # Fixed seeds for reproducibility
            done = False
            ep_reward = 0.0
            risk_scores = []
            last_info = {}

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    if deterministic:
                        action = self.model.get_deterministic_action(obs_tensor).squeeze(0)
                    else:
                        action, _, _ = self.model(obs_tensor)
                        action = action.squeeze(0)

                action_np = action.cpu().numpy()
                safe_action = self.safety_layer.project_action(
                    action_np,
                    self.env.robot_pos,
                    self.env.robot_theta,
                    self.env.obstacle_map.obstacles,
                )

                obs, reward, terminated, truncated, info = self.env.step(safe_action)
                done = terminated or truncated
                ep_reward += reward
                risk_scores.append(info.get("risk_score", 0.0))
                last_info = info

            result = EpisodeResult(
                success=last_info.get("goal_reached", False),
                collision=last_info.get("collision", False),
                timeout=not last_info.get("goal_reached", False) and not last_info.get("collision", False),
                total_reward=ep_reward,
                steps=last_info.get("step", 0),
                trajectory=[t.copy() for t in self.env.trajectory],
                risk_scores=risk_scores,
                goal_pos=self.env.goal_pos.copy(),
                obstacles=list(self.env.obstacle_map.obstacles),
            )
            results.append(result)

        return self._aggregate(results)

    def _aggregate(self, results: List[EpisodeResult]) -> EvaluationSummary:
        rewards = [r.total_reward for r in results]
        successful = [r for r in results if r.success]

        return EvaluationSummary(
            n_episodes=len(results),
            success_rate=np.mean([r.success for r in results]),
            collision_rate=np.mean([r.collision for r in results]),
            timeout_rate=np.mean([r.timeout for r in results]),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_path_length=float(np.mean([r.path_length for r in results])),
            mean_path_efficiency=float(np.mean([r.path_efficiency for r in results])),
            mean_risk_score=float(np.mean([r.mean_risk for r in results])),
            mean_goal_reach_steps=float(np.mean([r.steps for r in successful])) if successful else 0.0,
            results=results,
        )
