r"""
controller.py — deadlock recovery around a base policy (round 3; opt-in, unused by default).

A RecoveryController sits between the policy and the environment. It returns a *nominal*
action; the environment's CBF filter (SafetyFilterWrapper) still checks every action, so
no mode can bypass the filter or weaken its guarantee. The controller reads the true robot
state and the obstacle map from the environment, i.e. the same information the filter uses.

Modes (parameters: RecoveryConfig; defaults are the values fixed in
research/experiments/round3/PROTOCOL.md):
  "A"  multi-step escape planning, triggered at once by a stall signature (the filter cuts
       the commanded speed below v_cut, or the policy commands at most v_stop). The planner
       is used only if its best manoeuvre beats holding the current command by plan_margin
       (predicted cost-to-go); it then commits to a_commit steps and re-checks. Back-to-back
       commitments form one activation, which ends when control returns to the policy.
  "B"  deadlock-triggered fixed behaviour: after `window` steps with less than `min_disp`
       net displacement, turn on the spot towards the tangent of the nearest barrier (the
       side closer to the goal direction), then follow that tangent at full speed for
       `go_steps` steps.
  "C"  deadlock-triggered multi-step planning: the trigger of B, then receding-horizon
       planning (replan every `replan_every` steps) until the planner's own cost-to-go has dropped by
       `exit_gain`, the planner finds no improving manoeuvre (`min_gain`), or `max_steps` pass.
cost: "geodesic" (global map, costmap.GeodesicField) or "euclidean" (straight line), for A and C.

Bookkeeping per episode (self.stats): activations (start/end step, start/end geodesic cost,
exit reason, success), planner calls and time, controller time, steps under recovery.
An activation succeeds if the goal is reached during it or the geodesic cost-to-go drops by
at least `exit_gain` metres between its start and its end.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from safenav.recovery.costmap import EuclideanField, GeodesicField
from safenav.recovery.model import FilteredUnicycleModel
from safenav.recovery.planner import EscapePlanner

_WALL_DIRS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
MODES = ("A", "B", "C")


@dataclass
class RecoveryConfig:
    mode: str = "C"
    cost: str = "geodesic"
    horizon: int = 40
    window: int = 30             # B, C: steps of the no-progress test
    min_disp: float = 0.15       # B, C: metres
    exit_gain: float = 1.0       # C exit and success criterion (metres of geodesic cost-to-go)
    min_gain: float = 0.1        # C: give up if the best manoeuvre gains less than this
    max_steps: int = 100         # B, C: longest activation
    replan_every: int = 5        # C
    v_cut: float = 0.1           # A: stall signature, filter limit below this
    v_stop: float = 0.05         # A: stall signature, commanded speed at most this
    plan_margin: float = 0.5     # A: planner must beat holding the command by this (metres)
    a_commit: int = 10           # A: committed steps per decision
    go_steps: int = 30           # B
    turn_tol: float = 0.1        # B: radians
    k_turn: float = 2.0          # B: turn-rate gain

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.cost not in ("geodesic", "euclidean"):
            raise ValueError("cost must be geodesic or euclidean")


@dataclass
class Activation:
    start_step: int
    start_pos: tuple
    start_cost: float
    end_step: int = -1
    end_cost: float = float("nan")
    reason: str = ""
    success: bool = False
    start_plan_cost: float = float("nan")   # in the planner's own field (decisions use only this)


@dataclass
class EpisodeStats:
    activations: list = field(default_factory=list)
    planner_calls: int = 0
    planner_time_s: float = 0.0
    controller_time_s: float = 0.0
    recovery_steps: int = 0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RecoveryController:
    def __init__(self, policy, cfg: Optional[RecoveryConfig] = None):
        self.policy = policy
        self.cfg = cfg or RecoveryConfig()
        self.planner = EscapePlanner(self.cfg.horizon)
        self.env = None

    # ── episode set-up ──────────────────────────────────────────────────────
    def bind(self, env):
        """Call after env.reset(): builds the model and cost fields for this episode's map."""
        self.env = env
        base = env.unwrapped
        self.base = base
        self.model = FilteredUnicycleModel.from_env(env)
        flt = getattr(env, "safety_filter", None)
        kw = dict(workspace_size=self.model.W, robot_radius=self.model.r, safe_distance=self.model.d,
                  wall_margin=self.model.wall_margin, goal_radius=self.model.goal_radius)
        t0 = time.perf_counter()
        self.geo = GeodesicField(base.obstacle_map.obstacles, base.goal_pos, **kw)
        self.field_setup_s = time.perf_counter() - t0
        self.plan_field = self.geo if self.cfg.cost == "geodesic" else EuclideanField(base.goal_pos)
        self.hist = deque(maxlen=self.cfg.window + 1)
        self.active: Optional[Activation] = None
        self.queue = []            # committed controls (A, C)
        self.since_plan = 0
        self.b_side, self.b_phase, self.b_go = 0.0, "", 0
        self.stats = EpisodeStats()
        self.in_recovery = False
        self._has_filter = flt is not None

    def reset(self, seed=None):
        self.policy.reset(seed=seed)

    # ── helpers ────────────────────────────────────────────────────────────
    def _state(self):
        return self.base.robot_pos.copy(), float(self.base.robot_theta)

    def _cost(self, pos):
        return float(self.geo(pos))

    def _plan(self, pos, theta, extra=None):
        t0 = time.perf_counter()
        p = self.planner.plan(self.model, self.plan_field, pos, theta, extra)
        self.stats.planner_calls += 1
        self.stats.planner_time_s += time.perf_counter() - t0
        return p

    def _start(self, step, pos):
        self.active = Activation(step, (float(pos[0]), float(pos[1])), self._cost(pos),
                                 start_plan_cost=float(self.plan_field(np.asarray(pos, dtype=float))))
        self.stats.activations.append(self.active)

    def _end(self, step, pos, reason):
        a = self.active
        a.end_step, a.end_cost, a.reason = step, self._cost(pos), reason
        a.success = bool(np.isfinite(a.start_cost) and a.end_cost <= a.start_cost - self.cfg.exit_gain)
        self.active, self.queue, self.since_plan = None, [], 0
        self.hist.clear()

    def _stalled(self, pos):
        self.hist.append(pos)
        return len(self.hist) == self.hist.maxlen and np.hypot(*(self.hist[-1] - self.hist[0])) < self.cfg.min_disp

    def _nearest_barrier_dir(self, pos, theta):
        h, _ = self.model.barriers(np.array([pos[0]]), np.array([pos[1]]), np.array([theta]))
        k = int(np.argmin(h[0]))
        m = len(self.model.ox)
        if k < m:
            d = np.array([self.model.ox[k] - pos[0], self.model.oy[k] - pos[1]])
            return d / max(np.hypot(*d), 1e-9)
        return _WALL_DIRS[k - m]

    def notify_goal(self, step):
        """Called by the runner when the episode ends; closes an open activation."""
        if self.active is not None:
            pos, _ = self._state()
            self._end(step, pos, "episode end")

    # ── main entry ─────────────────────────────────────────────────────────
    def act(self, obs):
        t0 = time.perf_counter()
        step = int(self.base._step_count)
        pos, theta = self._state()
        mode = self.cfg.mode
        out = None
        if mode == "A":
            out = self._act_a(obs, step, pos, theta)
        elif mode == "B":
            out = self._act_b(obs, step, pos, theta)
        else:
            out = self._act_c(obs, step, pos, theta)
        self.in_recovery = self.active is not None
        if self.in_recovery:
            self.stats.recovery_steps += 1
        self.stats.controller_time_s += time.perf_counter() - t0
        return out

    # A: immediate stall signature + planning
    def _act_a(self, obs, step, pos, theta):
        c = self.cfg
        if self.queue:
            return self.queue.pop(0)
        a = np.asarray(self.policy.act(obs), dtype=float)
        v_cmd = float(np.clip(a[0], 0.0, 1.0))
        v_lim = float(self.model.speed_limit(np.array([pos[0]]), np.array([pos[1]]), np.array([theta]))[0])
        signature = (v_cmd > v_lim + 1e-9 and v_lim < c.v_cut) or v_cmd <= c.v_stop
        if signature:
            hold = np.repeat(a[None, :], c.horizon, axis=0)
            p = self._plan(pos, theta, extra=hold[None])
            hold_cost = self.planner.predicted_cost(self.model, self.plan_field, pos, theta, hold)
            if p.index < len(self.planner.library) and p.cost_end <= hold_cost - c.plan_margin:
                if self.active is None:            # back-to-back commitments form one activation
                    self._start(step, pos)
                self.queue = [p.controls[i] for i in range(1, c.a_commit)]
                return p.controls[0]
        if self.active is not None:                # control returns to the policy
            self._end(step, pos, "returned to policy")
        return a

    # B: no-progress trigger + fixed tangent behaviour
    def _act_b(self, obs, step, pos, theta):
        c = self.cfg
        if self.active is None:
            if not self._stalled(pos):
                return np.asarray(self.policy.act(obs), dtype=float)
            self._start(step, pos)
            n = self._nearest_barrier_dir(pos, theta)
            g = self.base.goal_pos - pos
            t_plus, t_minus = np.array([-n[1], n[0]]), np.array([n[1], -n[0]])
            self.b_side = 1.0 if t_plus @ g >= t_minus @ g else -1.0
            self.b_phase, self.b_go = "turn", 0
        if step - self.active.start_step >= c.max_steps:
            self._end(step, pos, "max steps")
            return np.asarray(self.policy.act(obs), dtype=float)
        n = self._nearest_barrier_dir(pos, theta)
        t = np.array([-n[1], n[0]]) * self.b_side
        err = _wrap(np.arctan2(t[1], t[0]) - theta)
        w = float(np.clip(c.k_turn * err, -1.0, 1.0))
        if self.b_phase == "turn":
            if abs(err) > c.turn_tol:
                return np.array([0.0, w])
            self.b_phase = "go"
        if self.b_go >= c.go_steps:
            self._end(step, pos, "behaviour done")
            return np.asarray(self.policy.act(obs), dtype=float)
        self.b_go += 1
        return np.array([1.0, w])

    # C: no-progress trigger + receding-horizon planning
    def _act_c(self, obs, step, pos, theta):
        c = self.cfg
        if self.active is None:
            if not self._stalled(pos):
                return np.asarray(self.policy.act(obs), dtype=float)
            self._start(step, pos)
            self.queue = []
        a = self.active
        if float(self.plan_field(pos)) <= a.start_plan_cost - c.exit_gain:
            self._end(step, pos, "progress")
            return np.asarray(self.policy.act(obs), dtype=float)
        if step - a.start_step >= c.max_steps:
            self._end(step, pos, "max steps")
            return np.asarray(self.policy.act(obs), dtype=float)
        if not self.queue or self.since_plan >= c.replan_every:
            p = self._plan(pos, theta)
            if p.gain < c.min_gain and not p.reaches_goal:
                self._end(step, pos, "no improving manoeuvre")
                return np.asarray(self.policy.act(obs), dtype=float)
            self.queue = [p.controls[i] for i in range(c.horizon)]
            self.since_plan = 0
        self.since_plan += 1
        return self.queue.pop(0)
