# Planner profiling (round 4, Phase 2)

The script is [`../profile_planner.py`](../profile_planner.py), and its outputs are [`summary.json`](summary.json), [`calls.csv`](calls.csv), [`steps.csv`](steps.csv) and the `cprofile_*.txt` files.

## Configuration

| Item | Value |
|---|---|
| Machine | laptop, Intel Core i7-1255U (hybrid cores), Windows 11 |
| Software | Python 3.12.7, NumPy 2.4.1 |
| Run conditions | mains power, process priority normal, one thread |
| Background load | system CPU load 29–53 % from the user's other applications |
| Scenarios | dedicated profiling set: expert, corrected generator, seeds 4,100,000–4,100,019 (used nowhere else) |

## What was measured

1. **Episodes:** 10 agents × 20 scenarios × modes A and C, with the reference planner and with the fast one. All 400 episode pairs are identical, and 7,682 planner calls were recorded.
2. **Isolated calls:** a seeded sample of 4,000 recorded calls, re-timed with both planners in alternating order, 3 repetitions each. The chosen plan is identical in all 4,000.
3. **Counts, memory, cProfile and field construction.**

**CPU clock.** CPU time is read from the thread cycle counter, calibrated against the wall clock at about 2.61 GHz. `time.thread_time` has a 15.6 ms tick on Windows, too coarse for calls of a few milliseconds.

## Results

| Per call | Reference | Fast |
|---|---:|---:|
| Candidates | 65 (mode C) or 66 (mode A: + "hold the command") | same |
| Rollout row-steps | 2,600–2,640 | same |
| Barrier evaluations (mean) | **5,325** (two passes over all rows every step) | **1,674** (one pass over the moving rows) |
| Wall time p50 / p99 / max (ms; 12,000 timings) | 6.5 / 18.9 / 100.6 | 3.7 / 12.0 / 74.9 |
| CPU time p50 / p99 / max (ms) | 6.4 / 16.6 / 35.1 | 3.6 / 10.3 / 29.2 |
| CPU time, best of 3, p50 / p99 (ms) | 5.7 / 12.9 | 3.3 / 7.1 |
| Fast / reference CPU, per call (median, p99) | – | 0.57, 0.80 |
| Peak memory allocated per call (tracemalloc) | 209 KiB | 168 KiB |

| Per episode / per step | Reference | Fast |
|---|---:|---:|
| Controller step latency p99, mode A (ms) | 21.3 | 11.3 |
| Controller step latency p99, mode C (ms) | 10.7 | 5.1 |
| Cost-to-go field construction (once per episode, outside the control loop): median / p99 (ms) | 11.0 / 31.2 | same (unchanged) |

## Bottleneck (cProfile, 300 calls)

- **Reference planner.**
  - `model.barriers` takes 4.2 s of 9.3 s (45 %) on its own, and 60 % including its NumPy calls; it is called twice per rollout step.
  - Next come `rollout` itself, `np.stack`, `speed_limit` and `np.clip`.
  - The arrays are small (66 rows × about 13 barriers). The time is therefore dominated by the per-call overhead of many small NumPy operations, not by arithmetic.
- **Fast planner.** It removes the second barrier pass, the barrier work for rows turning on the spot (speed 0 cannot be lowered), the work for rows already at the goal, the duplicated trigonometry and mode A's extra rollout.

## Optimisations considered and not used

| Optimisation | Why not used |
|---|---|
| Re-using the previous plan, or pruning candidates by heuristics | Would change which manoeuvre is chosen, so behaviour would need a new non-inferiority test. The exact optimisation already meets the target |
| Culling obstacles that cannot bind within the horizon (barrier > 6.1 m) | Exact, but there are only 9–12 obstacles per map. With call overhead dominating, the saving is negligible |
| numba / compiled code | Not installed. Different floating-point maths would also break bit-identity |
| Caching the cost-to-go field | It is already built once per episode and only looked up per call. The goal-rooted field does not depend on the robot state, so no incremental update is needed |
