r"""
fast.py — optimised escape planner (round 4; opt-in via RecoveryConfig(planner="fast")).

It returns the same plans as the reference planner (planner.py + model.py), which stays the
default and is not modified. Nothing safety-related is removed: every candidate is still rolled
out through the exact filter model, and the environment's filter still checks every executed
action. The savings come only from work whose result is known in advance or never used:

  1. Rows whose commanded speed is 0 (turning on the spot) skip the barrier computation. The
     filter can only lower a speed, so the executed speed is exactly 0 and the position stays.
  2. Rows that have reached the goal are frozen and skipped.
  3. The reference rollout also records the smallest barrier value, all positions, speeds and
     intervention flags for diagnostics. Planning needs only the end point and the goal step,
     so these are not computed; the second barrier evaluation per step disappears.
  4. cos and sin of the heading are computed once per step (the reference computes them twice).
  5. The goal test is applied only to rows that moved (and to all rows on the first step).
  6. Mode A compared its best plan with "hold the current command" by a second rollout; the
     hold candidate is already in the batch, so its score is reused (controller.py).

Exactness. Every remaining floating-point operation is the reference operation on the same
operands, in the same order: cos/sin are evaluated on the full heading array exactly as in
model.py (NumPy's vectorised kernels are then fed identical arrays), and the barrier terms are
formed as dist - R - r - d. End points, goal steps, scores and chosen plans are therefore
bit-identical to the reference (tests/test_recovery_fast.py checks this on thousands of states
and on whole episodes).
"""

import numpy as np

from safenav.recovery.planner import EscapePlanner, Plan

_EPS = 1e-9


def rollout_end(model, pos, theta, controls):
    """End positions (N, 2) and first goal step (N,; -1 if never) for N control sequences."""
    controls = np.asarray(controls, dtype=float)
    n, horizon = controls.shape[:2]
    px = np.full(n, float(pos[0]))
    py = np.full(n, float(pos[1]))
    th = np.full(n, float(theta))
    reached = np.full(n, -1)
    v_nom_all = np.clip(controls[:, :, 0], 0.0, model.v_max)
    om_all = np.clip(controls[:, :, 1], -model.omega_max, model.omega_max)
    ox, oy, orad = model.ox, model.oy, model.orad
    r, d, gamma, dt, vmax, W = model.r, model.d, model.gamma, model.dt, model.v_max, model.W
    inset = model.r + model.wall_margin
    gx, gy, gr = model.goal[0], model.goal[1], model.goal_radius
    for t in range(horizon):
        active = reached < 0
        ux_all, uy_all = np.cos(th), np.sin(th)
        v_nom = v_nom_all[:, t]
        move = np.flatnonzero(active & (v_nom > 0.0))
        if move.size:
            x, y = px[move], py[move]
            ux, uy = ux_all[move], uy_all[move]
            dx = ox[None, :] - x[:, None]
            dy = oy[None, :] - y[:, None]
            dist = np.hypot(dx, dy)
            h_obs = dist - orad[None, :] - r - d
            safe_dist = np.where(dist > _EPS, dist, 1.0)
            a_obs = np.where(dist > _EPS, (ux[:, None] * dx + uy[:, None] * dy) / safe_dist, 1.0)
            h_w = np.stack([x - inset, W - x - inset, y - inset, W - y - inset], axis=1)
            a_w = np.stack([-ux, ux, -uy, uy], axis=1)
            h = np.concatenate([h_obs, h_w], axis=1)
            a = np.concatenate([a_obs, a_w], axis=1)
            closing = a > _EPS
            lim = np.where(closing, gamma * h / np.where(closing, a, 1.0), np.inf).min(axis=1)
            lim = np.where(np.isfinite(lim), np.clip(lim, 0.0, vmax), vmax)
            v = np.clip(np.minimum(v_nom[move], lim), 0.0, 1.0)
            px[move] = x + v * ux * dt
            py[move] = y + v * uy * dt
        idx = np.flatnonzero(active)
        th_new = th[idx] + np.clip(om_all[idx, t], -1.0, 1.0) * dt
        th[idx] = (th_new + np.pi) % (2 * np.pi) - np.pi
        chk = idx if t == 0 else move
        if chk.size:
            dg = np.hypot(gx - px[chk], gy - py[chk])
            reached[chk[dg < gr]] = t + 1
    return np.column_stack([px, py]), reached


class _End:
    """Minimal stand-in for model.Rollout: what EscapePlanner.score reads."""

    def __init__(self, end, reached_at):
        self.pos = end[:, None, :]
        self.reached_at = reached_at


class FastEscapePlanner(EscapePlanner):
    """Same library, score and tie-break as EscapePlanner; rollouts through rollout_end()."""

    def plan(self, model, field, pos, theta, extra=None) -> Plan:
        cands = self.library if extra is None else np.concatenate([self.library, extra])
        end, reached = rollout_end(model, pos, theta, cands)
        cost = self.score(_End(end, reached), field)
        k = int(np.argmin(cost))
        now = float(field(np.asarray(pos, dtype=float)))
        e = float(cost[k])
        reach = bool(reached[k] >= 0)
        gain = (now - e) if np.isfinite(now) and np.isfinite(e) else 0.0
        p = Plan(cands[k].copy(), k, now, e, float(gain if not reach else max(gain, now)), reach)
        p.extra_costs = cost[len(self.library):].copy() if extra is not None else None
        return p

    def predicted_cost(self, model, field, pos, theta, controls):
        end, reached = rollout_end(model, pos, theta, controls[None])
        return float(self.score(_End(end, reached), field)[0])
