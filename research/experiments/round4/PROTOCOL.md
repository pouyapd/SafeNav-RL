# Round 4: planner efficiency, robustness and independent validation (protocol)

This protocol was committed **before** the round-4 pilots and confirmatory runs. Results are added below *Results*. Changes after a pilot are allowed only as dated **amendments** in *Amendments*, committed before the confirmatory run, and only for technical defects; a change motivated by outcome rates is not permitted.

Parts:
- **P** (latency, confirmatory);
- **R** (robustness, exploratory pilots then confirmatory);
- **F** (baseline fairness, confirmatory secondary);
- **T** (timeout analysis, exploratory);
- **Phase 1** (the round-3 audit) and **Phase 2** (profiling) are reported in [`audit/`](audit/) and [`profile/`](profile/).

## Part P: planner latency

**What changed.** [`safenav/recovery/fast.py`](../../../safenav/recovery/fast.py) (`RecoveryConfig(planner="fast")`, opt-in; the default stays the round-3 reference planner). It computes the **same plans**: end points, goal steps, scores and chosen manoeuvres are bit-identical, as tested in `tests/test_recovery_fast.py`. It only skips work whose result is known in advance or unused. No safety check is removed: every candidate is still rolled out through the exact filter model, and the environment's filter still checks every executed action.

**Hardware and load configuration (stated, not controlled beyond this).**

| Item | Value |
|---|---|
| Machine | laptop, Intel Core i7-1255U (hybrid: 2 performance + 8 efficiency cores, 12 threads), Windows 11 |
| Software | Python 3.12.7, NumPy 2.4.1, PyTorch 2.10 (CPU) |
| Power | mains power (checked; the run record stores the power state) |
| Process | one evaluation process, priority **normal**, 1 PyTorch thread, `OMP/MKL/OPENBLAS_NUM_THREADS=1`, no CPU pinning |
| Other load | no other experiment process. The user's normal applications (editor, browser) keep running; system CPU load is recorded per invocation |

This is a general-purpose OS on a laptop. **No result of Part P is a real-time guarantee.**

**Design.**
- **Scenarios:** expert stage, corrected generator, seeds 4,200,000 + i, i < 100 (unused before).
- **Agents and methods:** the 10 committed agents; methods `A:reference`, `A:fast`, `C:reference`, `C:fast`. These are 4,000 episodes, with the methods run in turn on each scenario so that the two planners alternate under the same load.
- **Measurement:** every planner call is timed with the wall clock (`time.perf_counter`), together with the thread CPU time (cycle counter, [`timing.py`](../../../safenav/recovery/timing.py)). Every controller step (`act`, including the policy's forward pass) is timed with the wall clock.

**Acceptance criteria (fixed now).**

| | Criterion |
|---|---|
| **P1 (primary)** | P99 of the per-call wall-clock latency of the fast planner, over all its calls in modes A and C, **< 100 ms** |
| **P2** | P99 of the per-step controller latency with the fast planner (modes A and C) < 100 ms, i.e. fewer than 1 % of control steps take ≥ 100 ms |
| **P3 (non-inferiority)** | For each mode and agent type, fast − reference must satisfy all three: success, the lower 95 % bound > −1.0 point; total deadlock, the upper 95 % bound < +1.0 point; 0 collisions with minimum clearance ≥ 0.30 m in both. The episode-level identity of all non-timing columns is also reported |
| Reported | P50/P90/P95/P99/max of call and step latency for both planners (wall and CPU); paired fast/reference ratio of per-call CPU time; for calls ≥ 100 ms, the share whose CPU time is below half the wall time (pre-emption, not computation) |

If P3 fails, the fast planner is not adopted, regardless of P1 and P2.

## Part R: robustness

**Conditions.** The filter always uses the true state and the true map. Only the recovery controller's information or the scenario changes, so exact-map and imperfect-information results are kept apart.

| Condition | Change | Information of the recovery controller |
|---|---|---|
| intermediate | intermediate stage (corrected generator) | exact |
| dense | expert stage with 15–25 requested obstacles | exact |
| heading | the initial heading is replaced by a uniformly random one (seeded per scenario; the same for every method) | exact |
| localisation error | expert | position + N(0, 0.05 m) per axis and heading + N(0, 0.05 rad), fresh each step |
| map error | expert | each obstacle centre shifted by N(0, 0.10 m) per axis, radius × U(0.85, 1.15), each obstacle missing with probability 0.10; fixed per episode |
| sensed map | expert | "known when seen": an obstacle enters the planner's map once a beam of the robot's own noisy range sensor (16 beams, 3 m) hits it, and is then known exactly. This is an idealisation, not SLAM |

Implementation: [`perception.py`](../../../safenav/recovery/perception.py). Activations are always *scored* with the true map at the true position.

**Scenario sets.** All are new; the pilots and the confirmatory runs use disjoint seeds.

| Set | Conditions | Pilot (exploratory) seeds, n | Confirmatory seeds, n | Methods |
|---|---|---|---|---|
| R-expert | localisation, map error, sensed map; also Part F | 4,300,000+, 30 | 4,400,000+, 200 | `nofilter`, `baseline`, `A`, `C`, `C_loc`, `C_map`, `C_sensed`, `planner_only`, `path_follower` |
| R-intermediate | intermediate | 4,310,000+, 30 | 4,410,000+, 200 | `baseline`, `C` |
| R-dense | dense | 4,320,000+, 30 | 4,420,000+, 200 | `baseline`, `C` |
| R-heading | heading | 4,330,000+, 30 | 4,430,000+, 200 | `baseline`, `C` |

- All recovery methods use the fast planner (Part P) with the round-3 parameters, unchanged and untuned.
- Agents: the 10 committed agents; policy mean.

**Primary family (Holm over 24 tests).** For each condition k (6) and agent type T (2), C_k is compared with the baseline on the same scenarios:
- **R1_k,T (deadlock):** total deadlock(C_k) − total deadlock(baseline) ≤ −10 points, with the 95 % interval below 0 and Holm p < 0.05;
- **R2_k,T (success):** success(C_k) − success(baseline) > 0, with the interval above 0 and Holm p < 0.05.

For the three information conditions, C_k is `C_loc`, `C_map` or `C_sensed`, and the baseline is from R-expert. For the three scenario conditions, C_k is `C` on that set.

**Claim per (k, T):** R1 and R2 hold, and there are 0 collisions with a minimum clearance ≥ 0.30 m in every method that uses the filter.

**Secondary (unadjusted).**
- Degradation from imperfect information: C_k − C (exact) on R-expert, for k ∈ {loc, map, sensed}, on deadlock, success and moving timeouts.
- Recovery success rate, loops, map rebuilds and obstacles known at the end (sensed map).
- Moving timeouts per condition.

## Part F: baseline fairness (R-expert set, confirmatory secondary)

| Method | Learned policy | Filter | Information used for decisions | Computation per step |
|---|---|---|---|---|
| `nofilter` | yes | **no** | noisy observation | one network forward pass |
| `baseline` | yes | yes | noisy observation (+ the filter's exact map) | forward pass + filter |
| `A`, `C` | yes | yes | observation + exact map and state at recovery | + planner calls when triggered |
| `planner_only` | no | yes | exact map and state, always | a planner call every 5 steps |
| `path_follower` | no | yes | exact map and state (global shortest path) | a steepest-descent lookup every step |

- **Reported per method:**
  - success, collisions, total deadlock, moving timeouts, minimum clearance;
  - path length and steps to goal (successes);
  - planner calls and latency.
- **Pre-registered secondary comparisons:** success of C − planner_only and C − path_follower, per agent type, with 95 % intervals. The bootstrap pairs scenarios and resamples the seeds of the learned agents. These are not part of the primary family.
- `nofilter` is a reference only: it has no safety layer, so it does not meet the safety requirement.

## Part T: timeout analysis (exploratory; round-3 main data and replays)

**Subject:** the 700 moving timeouts of C in round 3 (seeds 8,500,000+; already observed data, so everything in Part T is exploratory).

1. **Decomposition from replays** (reference planner, identical to round 3). For each step it records:
   - who proposed the action (policy or recovery);
   - the commanded and executed speed (filter slow-down);
   - the true geodesic cost-to-go.

   From these it measures:
   - steps before the first trigger;
   - the trigger delays (30-step windows);
   - recovery steps (turning on the spot vs driving);
   - policy steps after an activation in which the cost-to-go grew (the policy undoing progress);
   - the remaining distance at the time limit;
   - the ratio of the oracle path length to the 400-step budget at full speed.
2. **Counterfactual replays.** A seeded random sample of 200 of the 700 episodes; one factor changed at a time; diagnostics only, not proposed changes:
   - (a) time limit 800 steps instead of 400 (evaluation limit);
   - (b) trigger window 15 steps instead of 30 (late triggering);
   - (c) recovery never hands control back once triggered (policy steering back);
   - (d) planner-only from the first step (route and planner quality);
   - (e) filter decay rate γ = 1.0 instead of 0.5. This is the same safe set and still a valid discrete-time CBF (γ·Δt ≤ 1), but a less conservative speed cap, and it tests conservative speed limits. It is a diagnostic only; the filter of every reported method stays unchanged.
3. **Computation overhead.** In simulation the environment advances by steps, not by wall time, so planner computation time cannot cause a timeout. This is checked by design, not by experiment.

## Statistics

As in round 3:
- **Paired hierarchical bootstrap** with 10,000 replicates (scenarios paired; seeds paired when the same agents run under both methods), two-sided bootstrap p, Holm adjustment within the stated family.
- **Seed level:** per-seed rates, a paired t-test over seeds, d_z and Hedges' g_av.
- **Latency:** empirical percentiles over all calls or steps. Fast/reference ratios use a bootstrap over episodes.

Analysis code: [`analyze_round4.py`](analyze_round4.py), committed with this protocol.

## Amendments

**Amendment 1 (2026-11-17, after the pilots, before every confirmatory run).**

*Pilot assessment (exploratory; the outcome numbers are in `results/pilot_summary.json` and decide nothing).*
- **Safety:** 0 collisions and minimum clearance ≥ 0.30 m for every method with the filter; `nofilter` collided, as expected.
- **Mechanics:** triggers fired in all six conditions; map rebuilds occurred for the sensed map (about 6 per episode, about 7 obstacles known at the end).
- **Latency:** the latency tails in the pilots (below-normal priority, 26–30 % system load) were far larger than in profiling. This is why Part P uses its own controlled run.

*Two findings that are not defects of the pre-registered methods, and how they are handled.*
1. **The pre-registered `path_follower` reached the goal in 0 of 30 pilot scenarios.** All 30 were filter-blocked stalls.
   - **Cause:** the shortest path through the safe set runs along its boundary. Pure pursuit then steers slightly into an obstacle at barrier ≈ 0, and the filter holds the robot (replays: barrier 0.001–0.003 m, executed speed 0). This is the deadlock mechanism this study is about, not a code error.
   - **Handling:** the method stays as specified. In addition, an **exploratory, separately labelled** variant `path_follower_inflated` is added to the confirmatory R-expert set. It plans the path with 0.2 m more clearance than the filter keeps, as a classical planner normally inflates its margin. Without it, the comparison in Part F would rest only on a strawman.
   - The variant is not part of any pre-registered test. Its comparison with C is reported as exploratory.
2. **`C_loc` recovery success rate (secondary metric) was 0.29, against 0.91 for C,** while its episode success was similar.
   - **Cause:** under localisation noise, the exit test fires the first time the *estimated* cost-to-go has dropped by 1.0 m, so the true drop at that step is often just below 1.0 m, the threshold of the success definition.
   - **Handling:** the pre-registered metric is unchanged. A **supplementary** count with a 0.8 m threshold (`activations_success_08`) is recorded and reported separately as such.

No method parameter, endpoint, scenario set or hypothesis was changed.

---

## Results

Added on 2026-11-17, after all confirmatory runs. The numbers come from [`results/summary.json`](results/summary.json), produced by `python research/experiments/round4/analyze_round4.py`, and from [`timeouts/summary.json`](timeouts/summary.json). Pilot numbers (exploratory, seeds 4,3x0,000+) are in [`results/pilot_summary.json`](results/pilot_summary.json); they decided nothing and are not repeated here.

**Data integrity.**

| Set | Rows | Expected |
|---|---:|---:|
| R-expert | 14,600 | 10 agents × 200 × 7 learned methods + 3 map methods × 200 |
| R-intermediate | 4,000 | 4,000 |
| R-dense | 4,000 | 4,000 |
| R-heading | 4,000 | 4,000 |
| latency (Part P) | 4,000 | 4,000 |

- No duplicate keys, no NUL bytes, and only the pre-registered seeds were used.
- **Safety:** 0 collisions and minimum clearance ≥ 0.300 m in every method that uses the filter, in every set. `nofilter` collided, as expected; it has no safety layer.

### Part P: latency (confirmatory)

**Run configuration:**
- commit `f0194a1`;
- 4,000 episodes;
- mains power, priority normal;
- system load from the user's applications about 33 %.

| Per planner call (38,041 paired calls) | Reference | Fast |
|---|---:|---:|
| Wall p50 / p90 / p95 / p99 / max (ms) | 8.1 / 18.4 / 26.4 / 45.7 / 480.6 | 4.5 / 10.2 / 15.0 / **25.0** / 224.9 |
| CPU p50 / p99 / max (ms) | 8.0 / 31.2 / 54.4 | 4.4 / 18.7 / 40.6 |
| Calls ≥ 100 ms (all pre-empted: CPU < ½ wall) | 17 | 9 |
| Control steps ≥ 100 ms (of 419,777) | 57 (0.014 %) | 12 (0.003 %) |
| Per-episode longest step, p99 (ms) | 123.0 | 79.7 |

The paired fast/reference CPU ratio is 0.58, 95 % CI [0.573, 0.589]; the median per call is 0.56.

| Criterion | Result |
|---|---|
| P1: fast call p99 < 100 ms | **met** (25.0 ms) |
| P2: fewer than 1 % of control steps ≥ 100 ms | **met** (0.003 %) |
| P3: non-inferiority | **met**. All 2,000 matched episode pairs are identical in every non-timing column, so every success and deadlock difference is exactly 0 (CI [0, 0]); 0 collisions; clearance ≥ 0.300 m |

**Interpretation.**
- The fast planner is adopted for Parts R and F. It changes speed only, not behaviour.
- The reference planner also met the 100 ms target in this configuration. The round-3 figure of 189 ms was the p99 over episodes of each episode's longest call, measured partly under concurrent load (see [`audit/`](audit/)). It did not mean that the reference planner was too slow.
- Every call above 100 ms was operating-system pre-emption, not computation. The longest CPU time of any fast call was 40.6 ms.
- **This is one laptop with a general-purpose OS. It is not a real-time guarantee.**

**Latency outside Part P.** The R and F runs used below-normal priority and were not latency-controlled. Their tails are larger: in R-expert, 321 of 258,177 calls took ≥ 100 ms of wall time. Every one of them was pre-empted, and the longest CPU time of any call was 64.9 ms. These runs say nothing about latency beyond this.

### Part R: robustness (confirmatory, primary family of 24 tests)

C_k − baseline on the same scenarios, in percentage points, with 95 % paired hierarchical bootstrap intervals.
- Every bootstrap p is < 0.0001 (0 of 10,000 replicates), and so is every Holm-adjusted p.
- Every seed-level paired t-test gives p ≤ 0.0011, and every seed has the same sign.

| Condition (k) | Agent type | Deadlock: baseline → C_k | Δ deadlock [95 % CI] | Success: baseline → C_k | Δ success [95 % CI] | Claim |
|---|---|---|---|---|---|---|
| localisation error | with filter | 54.9 → 0.1 | −54.8 [−62.0, −47.6] | 43.0 → 84.5 | +41.5 [+34.5, +48.4] | yes |
| localisation error | without filter | 50.8 → 0.2 | −50.6 [−57.0, −44.0] | 49.1 → 84.7 | +35.6 [+29.4, +41.7] | yes |
| map error | with filter | 54.9 → 18.4 | −36.5 [−43.3, −30.0] | 43.0 → 71.5 | +28.5 [+21.2, +36.1] | yes |
| map error | without filter | 50.8 → 21.3 | −29.5 [−35.0, −24.0] | 49.1 → 71.6 | +22.5 [+17.4, +27.7] | yes |
| sensed map | with filter | 54.9 → 0.1 | −54.8 [−62.2, −47.3] | 43.0 → 86.6 | +43.6 [+36.4, +50.9] | yes |
| sensed map | without filter | 50.8 → 0.5 | −50.3 [−56.8, −43.7] | 49.1 → 85.2 | +36.1 [+30.0, +42.1] | yes |
| intermediate | with filter | 35.3 → 0.1 | −35.2 [−41.7, −28.7] | 61.3 → 91.5 | +30.2 [+24.0, +36.5] | yes |
| intermediate | without filter | 29.2 → 0.3 | −28.9 [−35.3, −22.8] | 70.5 → 92.9 | +22.4 [+17.2, +28.0] | yes |
| dense | with filter | 64.2 → 0.7 | −63.5 [−70.2, −56.9] | 33.2 → 83.3 | +50.1 [+43.8, +56.6] | yes |
| dense | without filter | 65.6 → 0.7 | −64.9 [−71.5, −58.4] | 34.0 → 80.2 | +46.2 [+39.6, +52.9] | yes |
| random heading | with filter | 60.5 → 0.4 | −60.1 [−66.6, −53.7] | 37.4 → 83.3 | +45.9 [+39.2, +52.8] | yes |
| random heading | without filter | 55.8 → 0.1 | −55.7 [−62.8, −48.6] | 43.9 → 81.0 | +37.1 [+30.4, +44.0] | yes |

**All 12 claims hold** (R1, R2, and the safety condition). On exact R-expert, C itself reaches deadlock 0.2 % / 0.3 % and success 86.3 % / 85.2 % (with / without filter during training).

**Imperfect information versus exact (secondary, unadjusted; C_k − C on R-expert).**

| Condition | Δ deadlock | Δ success | Δ moving timeouts |
|---|---|---|---|
| localisation error, with filter | −0.1 [−0.6, +0.4] | −1.8 [−4.1, +0.2] | +1.9 [−0.1, +4.2] |
| localisation error, without filter | −0.1 [−0.9, +0.7] | −0.5 [−2.1, +0.8] | +0.6 [−1.1, +2.6] |
| **map error, with filter** | **+18.2 [+13.7, +22.8]** | **−14.8 [−18.9, −10.8]** | −3.4 [−7.4, +0.2] |
| **map error, without filter** | **+21.0 [+15.8, +26.5]** | **−13.6 [−18.2, −9.2]** | −7.4 [−12.4, −2.4] |
| sensed map, with filter | −0.1 [−0.6, +0.0] | +0.3 [−0.9, +1.5] | −0.2 [−1.3, +0.9] |
| sensed map, without filter | +0.2 [+0.0, +0.8] | +0.0 [−1.0, +1.0] | −0.2 [−1.4, +0.9] |

**Reading.**
- **Localisation noise (σ = 5 cm / 0.05 rad) and the "known when seen" map cost almost nothing.**
- **A wrong map is the real weakness.**
  - With map errors, deadlock comes back to 18–21 %, almost all filter-blocked.
  - The planner plans through gaps that the true map does not have, or misses obstacles. The filter, which uses the truth, then blocks the manoeuvre.
  - C still more than halves the baseline deadlock here, but it is clearly worse than with the exact map.
- **Recovery success rate** (secondary):
  - C 0.87 / 0.86;
  - C_loc 0.34 / 0.36, or 0.81 / 0.78 on the supplementary 0.8 m threshold of Amendment 1;
  - C_map 0.31 / 0.21;
  - C_sensed 0.86 / 0.84, with about 5.9 map rebuilds and about 7.2 known obstacles per episode.
- **Scope.** The sensed-map condition is an idealisation: perfect obstacle shape once a beam hits it, with no SLAM. The map-error model is one synthetic error model. Neither result says how C would behave with a real mapping stack.

### Part F: baseline fairness (R-expert, 200 scenarios)

Rates in %. Learned methods are pooled over 10 agents (with / without filter during training); the map-based methods have no learned agent. Every method uses the same scenarios, the same filter (except `nofilter`), the same 400-step limit and the same success definition.

| Method | Success | Collision | Deadlock (filter-blocked) | Moving timeout | Min clearance (m) | Planner calls / episode |
|---|---|---|---|---|---|---|
| policy, no filter (`nofilter`) | 54.0 / 94.1 | **9.1 / 1.0** | 35.7 / 4.9 (0 / 0) | 1.2 / 0.0 | **−0.080 / −0.029** | 0 |
| policy + filter (`baseline`) | 43.0 / 49.1 | 0 | 54.9 / 50.8 (19.7 / 45.2) | 2.1 / 0.1 | 0.300 | 0 |
| A | 90.0 / 94.9 | 0 | 5.4 / 1.6 | 4.6 / 3.5 | 0.300 | 22.9 / 12.3 |
| C | 86.3 / 85.2 | 0 | 0.2 / 0.3 | 13.5 / 14.5 | 0.300 | 11.1 / 9.8 |
| planner only | 95.0 | 0 | 3.0 (2.5) | 2.0 | 0.300 | 35.1 |
| shortest-path follower (pre-registered) | 11.0 | 0 | 89.0 (89.0) | 0.0 | 0.300 | 368 lookups |
| path follower, +0.2 m margin (**exploratory**, Amendment 1) | 64.5 | 0 | 35.5 (35.5) | 0.0 | 0.300 | 220 lookups |

**Pre-registered secondary comparisons (unadjusted, success, C − X).**

| Comparison | With filter | Without filter |
|---|---|---|
| C − planner only | **−8.7 [−13.2, −4.4]** | **−9.8 [−14.9, −5.0]** |
| C − shortest-path follower | +75.3 [+69.5, +81.0] | +74.2 [+68.2, +80.1] |
| C − path follower +0.2 m (exploratory) | +21.8 [+15.4, +28.2] | +20.7 [+14.1, +27.2] |

**Reading.**
- **The planner alone is better than C.**
  - It has higher success (95 % against 86 % / 85 %), with the same filter and 0 collisions. Its median steps to the goal on successes (147.5) fall between C with filter-trained agents (173) and without (135).
  - C has less deadlock (0.2–0.3 % against 3.0 %), but many more moving timeouts (13.5–14.5 % against 2.0 %).
  - The study therefore does **not** show that the learned policy adds value over a model-based planner with an exact map. The C result is a recovery result for a learned policy, not a claim that the hybrid is the best navigator here.
- **Information access differs.**
  - Planner only and the path followers use the exact map and state at every step.
  - C uses the policy's noisy observation and needs the map only while recovering; with the sensed map it performs as with the exact map.
  - The planner-only and sensed-map combination was not run.
- **Computation differs.** Planner only makes about 35 planner calls per episode, against about 10–11 for C.
- **The classical shortest-path follower (pre-registered) fails** (89 % filter-blocked). This is the boundary-grazing deadlock explained in Amendment 1.
  - With a 0.2 m inflated margin (exploratory) it reaches 64.5 %.
  - All of its remaining failures are filter-blocked as well.
  - No further tuning of the classical baseline was done, so its numbers are a lower bound for a well-tuned classical stack.
- **Mode A has higher success than C here** (90.0 / 94.9 % against 86.3 / 85.2 %), but more deadlock (5.4 / 1.6 % against 0.2 / 0.3 %).

### Part T: moving timeouts of C (exploratory)

**Decomposition (round-3 C, 700 moving timeouts, all reproduced exactly; medians).**

| Quantity | Median |
|---|---|
| First trigger | step 108 |
| Steps spent waiting inside the 30-step trigger windows | 90 |
| Activations | 3 |
| Recovery steps | 138 (70 turning on the spot, 66 driving) |
| Policy steps | 262 |
| Steps where the filter lowered the commanded speed | 66 % |
| Executed / commanded speed | 0.32 |
| Mean executed speed | 0.22 m/s |
| Start → goal geodesic | 8.6 m |
| Progress | 4.3 m |
| Remaining at the time limit | 4.5 m |

- In 86 % of these episodes, more than 2 m remained at the limit.
- In 19 %, the policy undid more than 1 m of the recovery's progress.

**Counterfactual replays (200 sampled moving timeouts; one factor changed; 0 collisions in every variant).**

| Variant | Success | Moving timeout | Deadlock |
|---|---:|---:|---:|
| unchanged (control) | 0.0 | 100.0 | 0.0 |
| time limit 800 steps | 75.0 | 22.5 | 2.5 |
| trigger window 15 steps | 42.5 | 57.0 | 0.5 |
| recovery keeps control after triggering | 40.5 | 55.5 | 4.0 |
| planner only from step 0 | 82.0 | 14.5 | 3.5 |
| filter γ = 1.0 (diagnostic only) | 49.5 | 49.5 | 1.0 |

**Reading.** The moving timeouts are mainly **slow progress**, not being stuck.
- With the filter active and much turning on the spot, the robot moves at about a fifth of its top speed (0.22 m/s against 1.0 m/s).
- At that speed, the 400-step limit covers about 8.6 m, which is about the median path length.
- Each of the following recovers part of the lost time:
  - a longer limit;
  - earlier triggering;
  - not handing back to the policy;
  - a planner from the start;
  - a less conservative filter rate.

  None removes all timeouts.
- **Computation time cannot cause a timeout here.** The simulation advances by steps, not by wall time.

**Limits of this analysis.**
- The sample was selected on the outcome. The rates above say how many *of these failures* a change rescues, not what the change would do to all episodes: a change could also turn current successes into failures, which was not measured.
- γ = 1.0 changes the safety filter's speed cap. It is reported as a diagnostic of conservativeness only, and is not proposed.
- Nothing from Part T was used to change a method.

### What round 4 does and does not show

**Shows:**
- The fast planner gives identical behaviour at about 58 % of the CPU time per call.
- Under the stated laptop configuration, both planners meet the 100 ms p99 target.
- The C recovery result holds under the following, with 0 collisions throughout:
  - intermediate scenarios, dense scenarios and random initial headings;
  - small localisation noise and a "known when seen" map.
- It degrades clearly, though it still beats the baseline, when the planner's map is wrong.

**Does not show:**
- any real-time guarantee;
- behaviour with a real perception or mapping stack, or on a real robot;
- that a learned policy with recovery beats a model-based planner with the same exact map. **It does not: planner only had higher success.**
- that the classical baseline was tuned as well as C.
