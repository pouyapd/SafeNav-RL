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

*(added after the runs)*
