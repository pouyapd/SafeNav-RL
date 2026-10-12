"""
metrics.py

Per-episode navigation metrics and their aggregation.

Definitions (all distances in metres, times in seconds):

  success            goal reached (distance < goal_radius) without a collision
  obstacle_collision robot body touched an obstacle
  out_of_bounds      robot body left the workspace
  collision          obstacle_collision or out_of_bounds
  timeout            neither success nor collision within the step limit
  path_length        length of the executed path
  straight_line      start-to-goal distance minus goal_radius (lower bound on
                     any path that reaches the goal region)
  path_efficiency    straight_line / path_length (successful episodes; 1 = straight)
  spl                success weighted by path length, S * l / max(p, l)
                     (Anderson et al., 2018) with l = straight_line
  min_clearance      smallest distance between robot body and an obstacle
                     surface during the episode (negative = penetration)
  interventions      steps in which the safety filter reduced the speed
  mean_abs_domega    mean |ω_t - ω_{t-1}| of executed commands (rad/s per step);
                     lower = smoother steering
  mean_abs_dv        mean |v_t - v_{t-1}| of executed commands (m/s per step)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class EpisodeResult:
    """Outcome and metrics of one evaluation episode."""

    method: str
    stage: str
    episode_seed: int
    safety_filter: bool
    success: bool
    obstacle_collision: bool
    out_of_bounds: bool
    timeout: bool
    steps: int
    episode_return: float
    straight_line: float
    path_length: float
    min_clearance: float
    min_wall_clearance: float
    mean_risk: float
    max_risk: float
    interventions: int
    mean_abs_dv: float
    mean_abs_domega: float
    mean_speed: float
    dt: float = 0.1
    train_seed: Optional[int] = None
    # Kept only when requested (for plots); not written to CSV.
    trajectory: Optional[np.ndarray] = field(default=None, repr=False)
    headings: Optional[np.ndarray] = field(default=None, repr=False)
    goal: Optional[np.ndarray] = field(default=None, repr=False)
    obstacles: Optional[List[tuple]] = field(default=None, repr=False)
    interventions_mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def collision(self) -> bool:
        return bool(self.obstacle_collision or self.out_of_bounds)

    @property
    def path_efficiency(self) -> float:
        if self.path_length < 1e-9:
            return 0.0
        return float(min(1.0, self.straight_line / self.path_length))

    @property
    def spl(self) -> float:
        if not self.success:
            return 0.0
        return float(self.straight_line / max(self.path_length, self.straight_line, 1e-9))

    @property
    def intervention_rate(self) -> float:
        return float(self.interventions / self.steps) if self.steps else 0.0

    @property
    def time_to_goal(self) -> float:
        return float(self.steps * self.dt) if self.success else float("nan")

    @property
    def outcome(self) -> str:
        if self.success:
            return "success"
        if self.obstacle_collision:
            return "collision"
        if self.out_of_bounds:
            return "out_of_bounds"
        return "timeout"

    def to_row(self) -> Dict[str, Any]:
        """Flat dictionary for CSV export (no arrays)."""
        row = {k: v for k, v in asdict(self).items()
               if k not in ("trajectory", "headings", "goal", "obstacles", "interventions_mask")}
        row.update(
            outcome=self.outcome,
            collision=self.collision,
            path_efficiency=self.path_efficiency,
            spl=self.spl,
            intervention_rate=self.intervention_rate,
            time_to_goal=self.time_to_goal,
        )
        return row


def path_length(positions: np.ndarray) -> float:
    """Sum of segment lengths of a (T, 2) polyline."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def mean_abs_change(values: Sequence[float]) -> float:
    """Mean absolute difference between consecutive values (0 for < 2 values)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    low = 0.0 if successes == 0 else float(max(0.0, centre - half))
    high = 1.0 if successes == n else float(min(1.0, centre + half))
    return (low, high)


def t_interval(values: Sequence[float], confidence: float = 0.95,
               bounds: Optional[Tuple[float, float]] = None) -> tuple:
    """
    Mean and two-sided t confidence interval across independent runs (e.g. training seeds).

    The interval is symmetric around the mean, so for a rate near 0 or 1 it can
    extend past the possible range. `bounds=(0, 1)` intersects it with that
    range; the true value lies inside the range, so this does not lower the
    coverage. The interval reflects variation between runs only: when every run
    gives the same value it has zero width.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    if n == 1:
        return (mean, float("nan"), float("nan"))
    sem = values.std(ddof=1) / np.sqrt(n)
    half = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1) * sem)
    low, high = mean - half, mean + half
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return (mean, low, high)


def summarize(results: Iterable[EpisodeResult]) -> Dict[str, Any]:
    """
    Aggregate a list of episodes (pooled over seeds).

    Rates come with 95% Wilson intervals. Path efficiency and time-to-goal are
    averaged over successful episodes only; everything else over all episodes.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        return {"episodes": 0}

    def rate(flags: List[bool]) -> Dict[str, float]:
        k = int(np.sum(flags))
        low, high = wilson_interval(k, n)
        return {"rate": k / n, "ci_low": low, "ci_high": high, "count": k}

    succ = [r for r in results if r.success]
    summary = {
        "episodes": n,
        "success": rate([r.success for r in results]),
        "collision": rate([r.collision for r in results]),
        "obstacle_collision": rate([r.obstacle_collision for r in results]),
        "out_of_bounds": rate([r.out_of_bounds for r in results]),
        "timeout": rate([r.timeout for r in results]),
        "spl": float(np.mean([r.spl for r in results])),
        "path_efficiency_success": float(np.mean([r.path_efficiency for r in succ])) if succ else float("nan"),
        "time_to_goal_success": float(np.mean([r.time_to_goal for r in succ])) if succ else float("nan"),
        "mean_return": float(np.mean([r.episode_return for r in results])),
        "min_clearance_median": float(np.median([r.min_clearance for r in results])),
        "min_clearance_p05": float(np.percentile([r.min_clearance for r in results], 5)),
        "intervention_rate": float(np.mean([r.intervention_rate for r in results])),
        "mean_abs_domega": float(np.mean([r.mean_abs_domega for r in results])),
        "mean_speed": float(np.mean([r.mean_speed for r in results])),
    }
    return summary
