r"""
planner.py — multi-step escape planner (round 3; opt-in).

Candidate manoeuvres ("turn, then go"): turn on the spot at omega = +-1 rad/s for k steps
(k in TURN_STEPS; k = 0 means no turn), then drive at v = 1 m/s with a constant turn rate
from GO_OMEGAS for the rest of the horizon. With the defaults this gives 13 x 5 = 65
sequences of H = 40 steps (4 s). Every sequence is rolled out through the exact filter
model (model.py), so the prediction is what the environment would execute: the filter
can only slow a manoeuvre down, never let it leave the safe set.

Score: the cost-to-go of the end point (geodesic or Euclidean, costmap.py). A sequence
that reaches the goal scores below every other one, earlier is better. Ties go to the
lower candidate index (fewer turning steps first), so the planner is deterministic.
"""

from dataclasses import dataclass

import numpy as np

TURN_STEPS = (0, 4, 8, 12, 16, 24, 31)   # 31 steps at 1 rad/s turn the robot by about pi
GO_OMEGAS = (0.0, -0.5, 0.5, -1.0, 1.0)


def maneuver_library(horizon=40, turn_steps=TURN_STEPS, go_omegas=GO_OMEGAS):
    seqs = []
    for k in turn_steps:
        for sign in ((1.0,) if k == 0 else (1.0, -1.0)):
            for w in go_omegas:
                c = np.zeros((horizon, 2))
                c[:k, 1] = sign
                c[k:, 0] = 1.0
                c[k:, 1] = w
                seqs.append(c)
    return np.stack(seqs)


@dataclass
class Plan:
    controls: np.ndarray     # (H, 2) best sequence
    index: int
    cost_now: float
    cost_end: float          # predicted cost-to-go at the end of the best sequence
    gain: float              # cost_now - cost_end (inf-safe)
    reaches_goal: bool


class EscapePlanner:
    def __init__(self, horizon=40, turn_steps=TURN_STEPS, go_omegas=GO_OMEGAS):
        self.library = maneuver_library(horizon, turn_steps, go_omegas)
        self.horizon = horizon

    def score(self, rollout, field):
        end = rollout.pos[:, -1, :]
        cost = field(end).astype(float)
        reached = rollout.reached_at >= 0
        # goal-reaching sequences: below any distance, earlier is better
        cost = np.where(reached, -1.0 - (self.horizon - rollout.reached_at) / self.horizon, cost)
        return cost

    def plan(self, model, field, pos, theta, extra=None) -> Plan:
        """Best sequence from the library (plus optional extra candidates appended after it)."""
        cands = self.library if extra is None else np.concatenate([self.library, extra])
        ro = model.rollout(pos, theta, cands)
        cost = self.score(ro, field)
        k = int(np.argmin(cost))           # first minimum: deterministic tie-break
        now = float(field(np.asarray(pos, dtype=float)))
        end = float(cost[k])
        reach = bool(ro.reached_at[k] >= 0)
        gain = (now - end) if np.isfinite(now) and np.isfinite(end) else 0.0
        return Plan(cands[k].copy(), k, now, end, float(gain if not reach else max(gain, now)), reach)

    def predicted_cost(self, model, field, pos, theta, controls):
        """Cost-to-go at the end of one given sequence (used to compare with holding the current command)."""
        ro = model.rollout(pos, theta, controls[None])
        return float(self.score(ro, field)[0])
