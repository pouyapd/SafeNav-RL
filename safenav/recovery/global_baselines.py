r"""
global_baselines.py — classical full-map controllers for the round-4 fairness comparison (opt-in).

Both controllers use the exact map and state (the information the recovery planner uses) and no
learned policy. Like every other method, they only propose actions; the environment's CBF filter
checks every action.

  PlannerOnly    the round-3 escape planner as the whole controller: plan with the geodesic
                 cost-to-go and replan every `replan_every` steps, from the first step to the end.
  PathFollower   a global shortest path through the filter's safe set (the geodesic field, Dijkstra
                 on a 5 cm grid) tracked by pure pursuit: aim at the point `lookahead` metres along
                 the steepest-descent path; turn rate k * heading error (saturated at 1 rad/s);
                 speed v_max * max(0, cos(heading error)).
"""

import time

import numpy as np

from safenav.recovery.costmap import GeodesicField
from safenav.recovery.model import FilteredUnicycleModel
from safenav.recovery.planner import EscapePlanner
from safenav.recovery.timing import thread_cpu_s


class _MapController:
    def reset(self, seed=None):
        pass

    def bind(self, env, seed=None):
        self.env, self.base = env, env.unwrapped
        self.model = FilteredUnicycleModel.from_env(env)
        t0 = time.perf_counter()
        self.geo = GeodesicField(self.base.obstacle_map.obstacles, self.base.goal_pos, workspace_size=self.model.W,
                                 robot_radius=self.model.r, safe_distance=self.model.d,
                                 wall_margin=self.model.wall_margin, goal_radius=self.model.goal_radius)
        self.field_setup_s = time.perf_counter() - t0
        self.call_wall_ms, self.call_cpu_ms = [], []
        self.in_recovery = False

    def notify_goal(self, step):
        pass


class PlannerOnly(_MapController):
    name = "planner_only"

    def __init__(self, replan_every=5, horizon=40, planner="fast"):
        if planner == "fast":
            from safenav.recovery.fast import FastEscapePlanner
            self.planner = FastEscapePlanner(horizon)
        else:
            self.planner = EscapePlanner(horizon)
        self.replan_every = int(replan_every)

    def bind(self, env, seed=None):
        super().bind(env, seed)
        self.queue, self.since = [], 0

    def act(self, obs):
        if not self.queue or self.since >= self.replan_every:
            pos, th = self.base.robot_pos.copy(), float(self.base.robot_theta)
            w0, c0 = time.perf_counter(), thread_cpu_s()
            p = self.planner.plan(self.model, self.geo, pos, th)
            self.call_wall_ms.append(1000 * (time.perf_counter() - w0))
            self.call_cpu_ms.append(1000 * (thread_cpu_s() - c0))
            self.queue, self.since = [p.controls[i] for i in range(len(p.controls))], 0
        self.since += 1
        return self.queue.pop(0)


class PathFollower(_MapController):
    name = "path_follower"

    def __init__(self, lookahead=0.5, k_turn=2.0, v_max=1.0):
        self.lookahead, self.k_turn, self.v_max = float(lookahead), float(k_turn), float(v_max)

    def act(self, obs):
        b = self.base
        pos, th = b.robot_pos.copy(), float(b.robot_theta)
        w0, c0 = time.perf_counter(), thread_cpu_s()
        d = self.geo.path_direction(pos, self.lookahead)
        if d is None:                               # inside the goal cell or no descent: turn to the goal
            g = b.goal_pos - pos
            d = g / max(float(np.hypot(*g)), 1e-9)
        err = float((np.arctan2(d[1], d[0]) - th + np.pi) % (2 * np.pi) - np.pi)
        self.call_wall_ms.append(1000 * (time.perf_counter() - w0))
        self.call_cpu_ms.append(1000 * (thread_cpu_s() - c0))
        return np.array([self.v_max * max(0.0, np.cos(err)), float(np.clip(self.k_turn * err, -1.0, 1.0))])
