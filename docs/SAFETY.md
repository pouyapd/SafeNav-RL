# The safety filter: what it enforces and what it does not

Implementation: [`safenav/safety/cbf_filter.py`](../safenav/safety/cbf_filter.py) · tests: [`tests/test_safety_filter.py`](../tests/test_safety_filter.py)

## Setting

The simulated robot is a unicycle with position `p = (x, y)`, heading `θ`, speed `v ∈ [0, 1] m/s` and turn rate `ω ∈ [-1, 1] rad/s`. The environment integrates it with forward Euler, `dt = 0.1 s`:

```
p_{t+1} = p_t + v_t · dt · (cos θ_t, sin θ_t)
θ_{t+1} = θ_t + ω_t · dt
```

Obstacles are static circles (centre `c_i`, radius `R_i`); the workspace is the square `[0, W]²` with `W = 10 m`; the robot body has radius `r = 0.3 m`.

## Barrier functions

One barrier per obstacle and one per wall:

```
h_i(p) = ||p − c_i|| − (R_i + r + d_safe)        obstacle i       (d_safe = 0.3 m)
h_W(p) = x − (r + m),  W − x − (r + m),           walls            (m = 0.1 m)
         y − (r + m),  W − y − (r + m)
```

`h ≥ 0` for every barrier means the robot body keeps at least `d_safe` from every obstacle and `m` from every wall.

## The filter

For each barrier, `a_i` is the rate at which it shrinks per unit of speed when the robot drives along its heading `u = (cos θ, sin θ)`: `a_i = u · n_i` for an obstacle, where `n_i` is the unit vector from the robot to the obstacle centre, and `−u_x, u_x, −u_y, u_y` for the four walls. The filter keeps the policy's turn rate and reduces its speed to

```
v_safe = clip( min(v_nominal, min over barriers with a_i > 0 of  γ · h_i / a_i ),  0, v_max )
```

with `γ = 0.5 s⁻¹`. All constraints are upper bounds on the single variable `v`, so this closed form is the exact solution of the quadratic program `min (v − v_nominal)²` subject to `a_i · v ≤ γ · h_i` and `0 ≤ v ≤ v_max`. (The README of version 0.1 described a "differentiable QP" with an optional `cvxpy` solver; neither existed.)

## Guarantee

**Claim.** If every barrier is non-negative at time `t` and `γ · dt ≤ 1`, then after one step with the filtered action every barrier satisfies `h_i(p_{t+1}) ≥ (1 − γ dt) · h_i(p_t) ≥ 0`. By induction, a robot that starts in the safe set never leaves it.

**Proof.** The Euclidean norm is convex, so for an obstacle barrier
`||p_{t+1} − c|| ≥ ||p_t − c|| + ((p_t − c)/||p_t − c||) · (p_{t+1} − p_t) = ||p_t − c|| − dt · v · a_i`.
Hence `h_i(p_{t+1}) ≥ h_i(p_t) − dt · v · a_i`. If `a_i > 0` the filter enforces `v · a_i ≤ γ h_i(p_t)`, giving `h_i(p_{t+1}) ≥ (1 − γ dt) h_i(p_t)`; if `a_i ≤ 0` the barrier cannot decrease. Wall barriers are linear in `p`, so the same bound holds with equality. ∎

**Between samples.** The guarantee is about the sampled states `p_t`, and the benchmark measures clearance at those states. Between two samples the simulated robot moves along a straight segment of length `L = v · dt ≤ 0.1 m`. If both ends keep at least `ρ = R_i + r + d_safe` from an obstacle centre, every point of the segment keeps at least `√(ρ² − (L/2)²)` (the worst case is a chord of the circle of radius `ρ`). The smallest generated obstacle has `R = 0.3 m`, so `ρ ≥ 0.9 m` and the robot can come at most `0.9 − √(0.81 − 0.0025) ≈ 1.4 mm` closer than the margin between samples, far from contact. Wall barriers are linear, so for them there is no gap. A unit test checks this bound along every step of the adversarial test. This covers only the simulator's straight-line steps; the continuous-time motion of a real robot (curved paths, actuator dynamics) is not covered by the discrete-time guarantee.

**Assumptions.** The guarantee holds only inside the simulator, because it uses all of the following:

1. the robot moves exactly as the Euler unicycle model above (no slip, delay or actuation error, `v ≥ 0`);
2. obstacles are static circles whose positions and radii the filter knows exactly; the filter uses the true robot state (the observation noise of the policy does not reach the filter);
3. the episode starts in the safe set (the scenario generator ensures this: start and goal are at least 1.5 m from obstacle surfaces);
4. the filter is applied at every step.

**Not guaranteed.**

- *Reaching the goal.* The filter can stop the robot in front of an obstacle. If the robot is on the keep-out boundary (`h = 0`) and its heading is less than 90° from the obstacle centre (`a > 0`), the speed limit `γ h / a` is zero: the robot can only turn, and can only drive off again once its heading points along or away from the obstacle (`a ≤ 0`). If the policy keeps commanding speed without turning far enough, the robot waits there until the episode times out (a deadlock). This is the main failure of the PPO agent trained *without* the filter when the filter is added at test time (141 of its 175 expert timeouts). The agent trained *with* the filter mostly fails differently: in 122 of its 195 expert timeouts the policy itself commands zero speed, a median 0.41 m before the boundary, so these stops are not caused by the filter (README, *Where the policies struggle*; [`research/REPOSITORY_AUDIT.md`](../research/REPOSITORY_AUDIT.md) §5.1).
- *Moving obstacles, sensing errors, a physical robot.* None of these are modelled. The ROS 2 node in `ros2_integration/` uses laser-scan points as obstacles, which breaks assumption 2; it is an untested scaffold.
- *Infeasible states.* If a barrier were already negative (only possible when an assumption is broken), no speed `v ≥ 0` satisfies the approaching constraint; the filter then returns `v = 0` for that direction (stop, but still allow turning away).

## Evidence in this repository

| Check | Result |
|---|---|
| Unit test: adversarial policy driving at full speed toward the nearest obstacle with random turns, 40 expert scenarios, up to 250 steps each | the per-step bound `h_{t+1} ≥ (1 − γ dt) h_t` holds for every barrier at every step; no collision |
| Unit tests: closed-form speed limit (head-on, at an angle, moving away, wall), ω unchanged | pass |
| Unit test: the same adversarial policy, closest point of every straight step to every obstacle | never more than the bound above (≈ 1.4 mm) inside the margin |
| Benchmark: every condition with the filter on (3,600 test episodes over all controllers and stages) | 0 collisions; minimum clearance over all episodes 0.300 m (exactly the margin) |

## How the version 0.1 layer behaved

The previous `SafetyLayer.project_action` skipped every obstacle with `h ≥ 0` and only acted once the robot was already inside the margin (setting `v = 0` when approaching); it had no wall constraint. [`scripts/v01_safety_layer_check.py`](../scripts/v01_safety_layer_check.py) runs that rule (copied unchanged from version 0.1, commit `7e42c9d`) and the new filter through the same adversarial test (40 expert scenarios) in the current simulator:

| Rule | Margin | Deepest entry into the margin | Obstacle collisions | Left the workspace |
|---|---:|---:|---:|---:|
| version 0.1 | 0.8 m | 0.095 m | 0 / 40 | 2 / 40 |
| CBF filter | 0.8 m | 0 | 0 / 40 | 0 / 40 |
| version 0.1 | 0.3 m | 0.089 m | 0 / 40 | 2 / 40 |
| CBF filter | 0.3 m | 0 | 0 / 40 | 0 / 40 |

## Choice of the margin `d_safe`

Version 0.1 used `safe_distance = 0.8 m` (also as the risk-score distance). With an exactly enforced 0.8 m margin the robot needs a gap of 2 × (0.3 + 0.8) = 2.2 m between obstacle surfaces, but the scenario generator only guarantees 1.0 m. A geometric oracle (grid search at 5 cm in the free space inflated by `r + d_safe`; `safenav/evaluation/feasibility.py`) on the 100 expert test scenarios finds a path that keeps the margin in 54 % of them at 0.8 m, 82 % at 0.5 m and 100 % at 0.3 m or less. The default is therefore 0.3 m. The risk penalty in the reward still pushes the agent away from obstacles out to 0.8 m (`env.risk_distance`).

![Oracle feasibility and potential-field success versus margin](figures/safety_margin_sweep.png)

![Speed limit imposed by the filter for two headings](figures/cbf_speed_limit.png)

*The largest speed the filter allows at each position for a robot heading east (left) and north (right), computed with the implemented filter. White: outside the safe set.*
