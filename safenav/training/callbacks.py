"""
callbacks.py

Training logs and checkpoints.

A run directory contains:
    episodes.csv      one row per finished training episode
    progress.csv      one row per PPO update (losses, KL, rolling success rate, ...)
    validation.csv    one row per periodic evaluation on fixed validation seeds
    checkpoints/      best.pt (highest validation success rate) and final.pt
    run_info.json     configuration, seed, software versions and git commit
"""

import csv
import json
import os
import platform
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

EPISODE_FIELDS = [
    "episode", "timestep", "stage", "return", "length",
    "goal_reached", "collision", "obstacle_collision", "out_of_bounds", "timeout",
    "interventions",
]


def _json_convert(obj):
    """Convert numpy and bool types for JSON serialization."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    """Current git commit of the repository (None if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir or Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> Dict[str, Any]:
    """Software versions recorded with every run and benchmark."""
    import gymnasium
    import scipy  # noqa: F401  (version recorded if present)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "git_commit": git_commit(),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_convert)


class _CsvLog:
    """
    Append-only CSV file with a fixed header.

    With `resume_after` set, an existing file is kept up to (and including)
    the rows whose timestep is <= resume_after, and new rows are appended.
    Rows written after the last resume checkpoint of an interrupted run are
    dropped, so a resumed log has no duplicates.
    """

    def __init__(self, path: Path, fields: Iterable[str], resume_after: Optional[int] = None):
        self.path = path
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        kept = []
        if resume_after is not None and self.path.is_file():
            with open(self.path, newline="", encoding="utf-8") as f:
                kept = [row for row in csv.DictReader(f) if int(float(row["timestep"])) <= resume_after]
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields, extrasaction="ignore")
        self._writer.writeheader()
        for row in kept:
            self._writer.writerow(row)
        self._file.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({k: _fmt(row.get(k)) for k in self.fields})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _fmt(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return value


class TrainingLogger:
    """Writes episode, update and validation logs and prints a console summary."""

    PROGRESS_FIELDS = [
        "update", "timestep", "episodes", "stage", "lr", "fps",
        "mean_return_100", "success_rate_100", "collision_rate_100", "timeout_rate_100",
        "intervention_rate", "correction_penalty_mean", "filter_gap_sq_mean",
        "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_fraction", "explained_variance",
    ]
    VALIDATION_FIELDS = ["timestep", "episodes", "success_rate", "collision_rate", "timeout_rate", "mean_return"]

    def __init__(self, log_dir: str, verbose: bool = True, resume_state: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self._recent = deque(maxlen=100)
        self.total_episodes = 0
        self.total_timesteps = 0
        resume_after = None
        if resume_state is not None:
            self.load_state(resume_state)
            resume_after = self.total_timesteps

        self._episodes = _CsvLog(self.log_dir / "episodes.csv", EPISODE_FIELDS, resume_after)
        self._progress = _CsvLog(self.log_dir / "progress.csv", self.PROGRESS_FIELDS, resume_after)
        self._validation = _CsvLog(self.log_dir / "validation.csv", self.VALIDATION_FIELDS, resume_after)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_timesteps": self.total_timesteps,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "recent": list(self._recent),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.total_episodes = int(state["total_episodes"])
        self.total_timesteps = int(state["total_timesteps"])
        self.episode_rewards.extend(state["episode_rewards"])
        self.episode_lengths.extend(state["episode_lengths"])
        self._recent.extend(state["recent"])

    def log_episode(self, reward: float, length: int, info: Dict[str, Any],
                    stage: str = "", interventions: int = 0) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes += 1
        entry = {
            "episode": self.total_episodes,
            "timestep": self.total_timesteps,
            "stage": stage,
            "return": float(reward),
            "length": int(length),
            "goal_reached": bool(info.get("goal_reached", False)),
            "collision": bool(info.get("collision", False)),
            "obstacle_collision": bool(info.get("obstacle_collision", False)),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "timeout": not info.get("goal_reached", False) and not info.get("collision", False),
            "interventions": int(interventions),
        }
        self._recent.append(entry)
        self._episodes.write(entry)

    def rolling_rates(self) -> Dict[str, float]:
        if not self._recent:
            return {"success_rate_100": 0.0, "collision_rate_100": 0.0, "timeout_rate_100": 0.0,
                    "mean_return_100": 0.0}
        recent = list(self._recent)
        return {
            "success_rate_100": float(np.mean([e["goal_reached"] for e in recent])),
            "collision_rate_100": float(np.mean([e["collision"] for e in recent])),
            "timeout_rate_100": float(np.mean([e["timeout"] for e in recent])),
            "mean_return_100": float(np.mean([e["return"] for e in recent])),
        }

    def log_update(self, row: Dict[str, Any]) -> None:
        row = dict(row, episodes=self.total_episodes, timestep=self.total_timesteps, **self.rolling_rates())
        self._progress.write(row)
        if self.verbose and row.get("update", 0) % 10 == 0:
            print(
                f"[Step {self.total_timesteps:>9,}] episodes {self.total_episodes:>6} | "
                f"return(100) {row['mean_return_100']:>7.2f} | success {row['success_rate_100']:.0%} | "
                f"collision {row['collision_rate_100']:.0%} | stage {row.get('stage', '')} | "
                f"{row.get('fps', 0):.0f} steps/s"
            )

    def log_validation(self, row: Dict[str, Any]) -> None:
        self._validation.write(dict(row, timestep=self.total_timesteps))
        if self.verbose:
            print(
                f"[Validation @ {self.total_timesteps:,}] success {row['success_rate']:.0%} | "
                f"collision {row['collision_rate']:.0%} | timeout {row['timeout_rate']:.0%}"
            )

    def close(self) -> None:
        for log in (self._episodes, self._progress, self._validation):
            log.close()


class CheckpointCallback:
    """Saves the final model and the best model by validation success rate."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_score: Optional[tuple] = None

    def save(self, path: Path, model: torch.nn.Module, config: Dict, timestep: int,
             extra: Optional[Dict[str, Any]] = None,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Path:
        checkpoint = {
            "format_version": 2,
            "timestep": int(timestep),
            "model_state": model.state_dict(),
            "obs_dim": int(getattr(model, "obs_dim", 7)),
            "config": config,
        }
        if extra:
            checkpoint.update(extra)
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        return path

    def maybe_save_best(self, score: tuple, **kwargs) -> bool:
        """Save best.pt if `score` (compared lexicographically) improves."""
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save(self.checkpoint_dir / "best.pt", **kwargs)
            return True
        return False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
