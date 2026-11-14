# Round 3: deadlock recovery (protocol)

This protocol was committed **before** the pilot and before the main experiment. Results are added later, below *Results*. Changes after the pilot are allowed only as dated **amendments** in the section *Amendments*, committed before the main run, and only for technical defects (see *Pilot*).

## Question

Round 2 found that every change that removed policy-side self-stops produced filter-blocked stalls instead, so total deadlock stayed at 34–40 %. Round 3 asks:

**Can planning-level recovery, triggered when the robot makes no progress, reduce total deadlock without weakening safety?**

The mechanism analysis ([`mechanism.py`](mechanism.py), hypotheses M1–M4, exploratory) tests separately whether stalls occur where progress needs a manoeuvre away from the goal direction.

## Methods compared (same agents, scenarios and filter)

| Method | What it does | Information |
|---|---|---|
| **baseline** | the trained policy (mean action) and the CBF filter, as in round 2 | – |
| **A**: multi-step escape planning | at every step with a *stall signature*, the planner compares its best manoeuvre with holding the policy's command and takes over for 10 steps if it is predicted to be ≥ 0.5 m better | exact map (geodesic cost-to-go) |
| **B**: deadlock-triggered recovery behaviour | after 30 steps with < 0.15 m net displacement: turn on the spot to the tangent of the nearest barrier (the side closer to the goal direction), then follow that tangent at full speed for 30 steps | nearest barrier only (local) |
| **C**: combined | the trigger of B, then receding-horizon planning (replan every 5 steps) until the cost-to-go has dropped by 1.0 m, no manoeuvre gains ≥ 0.1 m, or 100 steps pass | exact map (geodesic cost-to-go) |
| **C_euclid** (ablation) | C with the straight-line distance as cost-to-go instead of the geodesic one | local only |

- **Code:** [`safenav/recovery/`](../../../safenav/recovery/), opt-in and unused by default. The parameter values are the defaults of `RecoveryConfig`.
- **Safety.** Every method only proposes a nominal action. The environment's CBF filter (`SafetyFilterWrapper`) checks every action, as in all earlier rounds. No method bypasses or relaxes the filter, so the one-step guarantee is unchanged. The tests check that every executed action equals the filter applied to the proposed one (`tests/test_recovery.py`).
- **Information.** The planner and B read the true state and the obstacle map, which is the same information the filter already uses. The geodesic cost-to-go (shortest path through the filter's safe set on a 5 cm grid, `costmap.py`) is *global* information. `C_euclid` and B do not use it, so they measure what local information achieves.

## Definitions (fixed)

| Item | Definition |
|---|---|
| **Stall signature** (A) | the filter cuts the commanded speed to below 0.1 m/s, or the policy commands ≤ 0.05 m/s |
| **Deadlock detection / recovery trigger** (B, C) | net displacement over the last 30 steps (3 s) < 0.15 m, outside an activation. The history is cleared when an activation ends, so a new trigger needs 30 further steps |
| **Planning horizon and budget** (A, C) | 40 steps (4 s); 65 candidate manoeuvres per call, each rolled out through an exact vectorised model of the filter and the unicycle (`model.py`; equal to the environment to 1e-9 in tests); one call per decision. A: one call per signature step outside a commitment. C: one call every 5 steps during an activation. Budget: mean controller time per step ≤ 10 ms and the 99th percentile of planner calls ≤ 100 ms on this laptop |
| **Candidate manoeuvres** | turn on the spot at ±1 rad/s for k ∈ {0, 4, 8, 12, 16, 24, 31} steps, then drive at 1 m/s with turn rate ∈ {0, ±0.5, ±1} rad/s for the rest of the horizon. Score: cost-to-go of the end point; reaching the goal beats everything; ties go to the lower index |
| **Activation** | a contiguous block of steps in which the recovery module, not the policy, proposes the action. A: back-to-back commitments are one activation |
| **Recovery success** | the goal is reached during the activation, or the geodesic cost-to-go drops by ≥ 1.0 m between its start and end. The geodesic cost is used for *scoring* every method, including B and C_euclid, which never use it for decisions |
| **Episode outcomes** | success (goal within 0.5 m), collision, timeout (400 steps, expert) |
| **Stall classes** | round-2 rules on the proposed action: last 100 steps, net displacement < 0.10 m; filter-blocked if the filter limited the speed in ≥ 50 % of those steps; else self-stop if the proposed speed was ≤ 1e-3 in ≥ 50 %; else other stall. Otherwise "moving" |
| **Total deadlock** | filter-blocked + self-stop + other stall (share of episodes). "Moving" timeouts are failures but not deadlock; they are reported next to it, so that deadlock turned into wandering is visible |
| **Safety constraint** | 0 collisions and minimum obstacle clearance ≥ 0.30 m in every episode. Any collision stops the experiment for debugging |

## Agents and scenarios

- **Agents.** The 10 committed agents (`results/checkpoints/`): 5 trained **with** the filter (self-stops dominate) and 5 trained **without** it (filter-blocked stalls dominate). Checkpoint `best`, policy mean. No training in this round.
- **Scenarios.** Expert stage, the corrected generator of E3 (`env.goal_sampling=rejection`), filter on. None of these seeds had been used before; checked with `git grep`.
  - **Pilot (exploratory):** seeds 8,300,000 + i, i < 40.
  - **Main (confirmatory):** seeds 8,500,000 + i, i < 500.

  All methods run on the same scenarios with the same initial conditions. An episode under recovery is identical to the baseline episode up to the first activation.
- **Separation.** No round-3 parameter was fitted on these scenarios. The parameter values above were set from the round-2 analysis and the step limits of the stage. The tests use seeds 8,900,000+, and the pilot uses its own seeds.

## Endpoints and hypotheses (confirmatory, main set)

**Primary comparison: C vs. baseline**, separately for the two agent types T ∈ {trained with the filter, trained without it}. That gives four tests, Holm-adjusted together:
- **H1_T (deadlock):** total deadlock(C) − total deadlock(baseline) ≤ −10 percentage points, with the 95 % interval below 0 and Holm-adjusted p < 0.05.
- **H2_T (success, safeguard):** success(C) − success(baseline) > 0, with the 95 % interval above 0 and Holm-adjusted p < 0.05.

**Claim rule.** "C reduces total deadlock without compromising safety" for type T only if H1_T and H2_T hold **and** there are 0 collisions with a minimum clearance ≥ 0.30 m. A reduction in one stall class alone is not a claim.

**Secondary (reported, not decisive; unadjusted unless stated):**
- A vs. baseline and B vs. baseline, on the same two endpoints (Holm within this family of 8 tests);
- C vs. B (does planning matter?), C vs. A (does the trigger matter?), C vs. C_euclid (does global information matter?);
- self-stop, filter-blocked, other stall and moving rates;
- collisions and minimum clearance;
- recovery success rate (successful activations / activations);
- activations per episode and repeated activations within 0.5 m of an earlier one (loops);
- filter intervention rate (intervened steps / steps);
- mean activation duration;
- share of the last 100 steps under recovery;
- computational overhead: planner time per call (mean, 99th percentile), controller time per step, episode wall time relative to baseline, and field set-up time per episode.

## Statistics

- **Paired hierarchical bootstrap** (10,000 replicates). Scenarios are paired, and seeds are paired, because the same agents run under both methods. The intervals therefore include seed-level uncertainty. The p-value is two-sided, from the bootstrap distribution.
- **Seed level, next to every bootstrap result:**
  - per-seed rates, mean ± SD and t-interval over the 5 seeds;
  - a paired t-test over seeds;
  - the paired effect size d_z (mean difference / SD of the differences);
  - Hedges' g_av (mean difference / average SD of the two conditions, small-sample corrected).
- **Failure-case analysis** of C's remaining deadlocks and loops:
  - exit reasons;
  - geodesic progress;
  - whether the geodesic path from the stall position exists;
  - five trajectories plotted per stall class, chosen by the lowest scenario seed (no selection by appearance).

## Pilot (exploratory)

Seeds 8,300,000 + i, i < 40, all five methods, the 10 agents (2,000 episodes). It checks:
1. triggers fire;
2. activations end;
3. recovery acts on both stall types;
4. 0 collisions;
5. overhead within budget;
6. no runaway loops (activations per episode).

It does **not** decide anything about the hypotheses.

**Permitted changes after the pilot:** only technical defects, such as a bug, a trigger that cannot fire, or a runtime outside the budget. Each must be described in *Amendments* with its reason and committed before the main run. A change motivated by success or deadlock rates is **not** permitted. It may only be run as a separately labelled exploratory variant, in addition to the pre-registered methods.

## Commands

```bash
python research/experiments/round3/run_recovery.py --tag pilot --agent filtered=results/checkpoints/filtered \
    --agent unfiltered=results/checkpoints/unfiltered --stages expert --seed-base 8300000 --n 40 \
    --set env.goal_sampling=rejection
python research/experiments/round3/run_recovery.py --tag main --agent filtered=results/checkpoints/filtered \
    --agent unfiltered=results/checkpoints/unfiltered --stages expert --seed-base 8500000 --n 500 \
    --set env.goal_sampling=rejection            # repeat until complete (resumable)
python research/experiments/round3/analyze_round3.py
```

## Not in this round

- Changes to training, a learned recovery policy, sampled actions, and the E4 agents.
- A physical robot.
- Any change to the filter, its margin or its γ.

## Amendments

**Amendment 1 (2026-11-13, after the pilot, before the main run; measurement only).** The protocol caps the 99th percentile of planner-call time at 100 ms. The pilot runner recorded only the mean time per call within an episode. The 99th percentile of that mean over pilot episodes was 24.7 ms, but this is not the quantity specified.
- The controller now records the longest single planner call per episode (`planner_max_call_ms`).
- The budget check uses the 99th percentile of that maximum over episodes, which is conservative (an upper bound on the per-call percentile).

Nothing that affects the actions taken was changed. No other change was made after the pilot.

**Pilot assessment (2026-11-13; exploratory).**
- **Technical criteria:**
  1. triggers fired for both stall types;
  2. activations ended (at most 4 per episode for C, 17 for A);
  3. recovery acted on both stall types;
  4. 0 collisions in 2,000 episodes, minimum clearance 0.300 m;
  5. mean controller time ≤ 3.3 ms per step, measured on battery;
  6. no runaway loops.
- **Conclusion:** the pilot is technically valid, and the main run proceeds unchanged. Its outcome numbers are in `results/pilot/summary.json` and are **not** used for any decision.

---

## Results

**Status (2026-11-14):** pilot **completed (exploratory)**; main experiment **completed (confirmatory)**: 25,000 episodes (5 methods × 10 agents × 500 scenarios). Numbers come from `results/main/episodes.csv` through [`analyze_round3.py`](analyze_round3.py), which writes `results/main/summary.json`. The mechanism analysis is in [`MECHANISM.md`](MECHANISM.md), and the round-2 audit in [`audit/README.md`](audit/README.md).

**Execution.**
- The protocol and analysis (`c1abd93`) and amendment 1 (`5d2838d`) were pushed before the main run.
- The 8 recorded invocations ran at `914eda8`, which adds only the figure script, with unmodified code and on mains power (2.66 h).
- The first 776 rows were written by an earlier invocation at `5d2838d` on battery. That invocation ended with the session and left no record entry. Rows are keyed, so no episode is counted twice; outcomes do not depend on power state, only timings do.

### Safety

There were **0 collisions in 25,000 episodes** (2,000 more in the pilot), and the minimum obstacle clearance was 0.3000 m in every method. Every proposed action passed through the unchanged filter.

### Primary result: C vs. baseline (Holm over the 4 tests)

| Agents | Endpoint | Baseline | C | Difference [95 % CI] | p (Holm) | Criterion |
|---|---|---:|---:|---:|---:|---|
| Trained with the filter | total deadlock | 58.8 % | 0.6 % | **−58.2 points [−64.3, −52.4]** | < 0.001 | **H1 holds** |
| Trained with the filter | success | 39.2 % | 85.9 % | **+46.7 points [+40.6, +52.6]** | < 0.001 | **H2 holds** |
| Trained without the filter | total deadlock | 53.4 % | 0.4 % | **−53.0 points [−57.5, −48.5]** | < 0.001 | **H1 holds** |
| Trained without the filter | success | 46.4 % | 85.2 % | **+38.8 points [+34.4, +43.4]** | < 0.001 | **H2 holds** |

- **Statistics.** Paired hierarchical bootstrap (scenarios and seeds paired), so the intervals include seed-level uncertainty. Bootstrap p = 0 in 10,000 replicates, reported as < 0.001.
- **Seed level (5 seeds).** The difference has the same sign in every seed.
  - Paired t-tests: p < 10⁻⁴ for all four.
  - d_z: −9.4 and +7.3 (with the filter); −19.9 and +11.5 (without).
  - Per-seed rates are in `summary.json`.
- **Decision (pre-registered claim rule).** H1 and H2 hold for both agent types with 0 collisions, so **C reduces total deadlock without compromising safety in this setting.**

**What the claim does not cover.** Part of the removed deadlock became timeouts in which the robot is still moving:

| Agents | Method | Success | Deadlock | of which filter-blocked / self-stop | Moving timeout | All timeouts |
|---|---|---:|---:|---:|---:|---:|
| With filter | baseline | 39.2 ± 6.3 | 58.8 ± 6.1 | 20.4 / 38.4 | 2.0 | 60.8 |
| With filter | A | **87.9 ± 4.7** | 7.9 ± 3.9 | 5.4 / 2.6 | 4.2 | 12.1 |
| With filter | B | 69.6 ± 3.4 | 9.8 ± 3.7 | 9.4 / 0.4 | 20.6 | 30.4 |
| With filter | **C** | 85.9 ± 1.8 | **0.6 ± 0.2** | 0.5 / 0.1 | 13.5 | 14.1 |
| With filter | C_euclid | 72.9 ± 1.5 | 25.3 ± 0.9 | 21.8 / 3.5 | 1.8 | 27.1 |
| Without filter | baseline | 46.4 ± 2.7 | 53.4 ± 2.7 | 48.4 / 4.9 | 0.3 | 53.6 |
| Without filter | A | **94.1 ± 0.8** | 2.4 ± 0.7 | 2.1 / 0.3 | 3.5 | 5.9 |
| Without filter | B | 74.6 ± 1.0 | 22.9 ± 1.0 | 21.8 / 1.1 | 2.5 | 25.4 |
| Without filter | **C** | 85.2 ± 1.2 | **0.4 ± 0.1** | 0.3 / 0.1 | 14.5 | 14.8 |
| Without filter | C_euclid | 72.2 ± 1.1 | 27.0 ± 0.9 | 27.0 / 0.0 | 0.8 | 27.8 |

Values are percentages of 2,500 episodes; ± is the SD over the 5 seeds.

- C's moving timeouts rose by 11.5 points [8.6, 14.6] (with filter) and 14.2 points [11.2, 17.4] (without).
- Of C's 700 moving timeouts, 686 were deadlocks under the baseline. The robot was freed but did not finish within 400 steps: the median remaining path fell from 8.6 m to 4.5 m after a median of 3 activations.
- C turned only 2 of the 2,139 baseline successes into failures. For comparison: A 11, B 27, C_euclid 9.

### Secondary results

**A and B vs. baseline (Holm over 8 tests; all p < 0.001).**

| Method | Agents | Deadlock | Success |
|---|---|---:|---:|
| A | with filter | −50.9 [−55.1, −46.4] | +48.7 [+43.4, +53.5] |
| A | without filter | −51.0 [−55.6, −46.6] | +47.8 [+43.2, +52.4] |
| B | with filter | −49.0 [−55.7, −41.0] | +30.4 [+26.7, +34.2] |
| B | without filter | −30.4 [−34.7, −26.4] | +28.2 [+24.2, +32.4] |

**Method contrasts (unadjusted).**

| Contrast | Agents | Deadlock | Success | Moving timeouts |
|---|---|---:|---:|---:|
| C − B (planning vs. fixed behaviour) | with filter | −9.3 [−13.1, −6.2] | +16.4 [+12.4, +20.3] | −7.1 [−12.3, −0.7] |
| C − B | without filter | −22.6 [−26.1, −19.0] | +10.6 [+7.9, +13.4] | +12.0 [+9.1, +15.0] |
| C − A (persistent vs. immediate trigger) | with filter | −7.4 [−11.2, −4.2] | −2.0 [−5.6, +1.8] | +9.3 [+7.0, +12.0] |
| C − A | without filter | −2.0 [−3.2, −0.9] | **−9.0 [−11.7, −6.3]** | +11.0 [+8.4, +13.7] |
| C − C_euclid (global vs. local cost) | with filter | −24.8 [−28.3, −21.3] | +13.0 [+10.2, +16.0] | +11.8 [+9.2, +14.6] |
| C − C_euclid | without filter | −26.6 [−30.4, −22.8] | +13.0 [+10.3, +15.8] | +13.7 [+10.8, +16.7] |

**Reading.**
- **Global information matters.** With the straight-line cost, the same trigger and planner leave 25–27 % deadlock, and most of its activations end with "no improving manoeuvre". This agrees with the mechanism analysis: the path out usually starts away from the goal.
- **Planning beats the fixed tangent behaviour** on deadlock and success for both agent types.
- **The immediate trigger (A) is at least as successful as the persistent one (C).** For agents trained without the filter, it is 9 points more successful. C has the lowest deadlock, but it waits 3 s before every activation and so loses time, which shows up as moving timeouts. A has more residual deadlock (2–8 %) and takes control far more often.
- C was the pre-registered primary method, and the claim rule is evaluated on C only. A's higher success is a secondary, unadjusted finding.

### Recovery statistics

| Method | Agents | Episodes with an activation | Activations per episode (max) | Recovery success rate | Mean activation (steps) | Repeated activations | Filter intervention rate |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | with / without | – | – | – | – | – | 45.7 % / 78.4 % |
| A | with filter | 88 % | 3.07 (25) | 3.8 % | 13.9 | 2,612 | 62.9 % |
| A | without filter | 84 % | 2.77 (15) | 6.5 % | 17.6 | 2,448 | 68.3 % |
| B | with filter | 66 % | 1.70 (6) | 17.4 % | 41.6 | 1,130 | 62.5 % |
| B | without filter | 55 % | 1.58 (5) | 11.0 % | 36.1 | 2,077 | 74.3 % |
| C | with filter | 66 % | 1.15 (9) | **85.9 %** | 48.8 | 110 | 58.7 % |
| C | without filter | 55 % | 1.00 (9) | **86.0 %** | 51.9 | 129 | 69.3 % |
| C_euclid | with filter | 66 % | 2.51 (11) | 18.2 % | 19.1 | 3,804 | 63.1 % |
| C_euclid | without filter | 55 % | 2.66 (11) | 10.4 % | 11.8 | 4,941 | 75.9 % |

- **Recovery success** means the activation ends at the goal or with ≥ 1.0 m less geodesic cost-to-go. This measure is not meaningful for A: its activations are short nudges (14–18 steps) that rarely gain 1 m each, although A's episodes succeed most often.
- **Repeated activations** start within 0.5 m of an earlier one, which indicates a loop. They are rare for C (110–129 in 2,500 episodes) and frequent for A, B and C_euclid.
- **Filter intervention rate.** C raised it for filter-trained agents (+13.0 points [6.7, 18.2]): the planner drives at full speed along boundaries. C lowered it for agents trained without the filter (−9.2 [−10.5, −7.7]).

### Computational overhead (laptop CPU, one thread, below-normal priority)

| Method | Planner time per call (mean) | Controller time per step (mean) | Map set-up per episode | Episode wall time (baseline 0.19–0.24 s) |
|---|---:|---:|---:|---:|
| A | 12.7–17.9 ms | 2.0–4.7 ms | 14–21 ms | 0.44–1.08 s |
| B | – | 0.2–0.3 ms | 14–24 ms (scoring only) | 0.15–0.35 s |
| C | 13.1–15.8 ms | 1.0 ms | 14–17 ms | 0.33–0.39 s |

**Budget (pre-registered).**
- **Mean controller time ≤ 10 ms per step: met** (≤ 4.7 ms).
- **99th percentile of planner calls ≤ 100 ms: not met** on the conservative measure of amendment 1. The longest call per episode has a median of 17.7 ms and a 95th percentile of 36 ms, but its 99th percentile is **189 ms**. 1.7 % of episodes contain a call above 100 ms.
  - The maximum is 15.4 s, which is almost certainly the machine being suspended or pre-empted: the process runs at below-normal priority on a laptop in use.
  - These are wall-clock times of an unoptimised NumPy planner under background load. They show that the method is cheap on average, but they are **not** a real-time guarantee.
- One control step is 100 ms of simulated time.

### Failure cases of C

- **Residual deadlocks:** 23 of 5,000 episodes (19 filter-blocked, 4 self-stop). All had activations, and 22 had repeated activations. Their activations mostly ended with "no improving manoeuvre" (109 exits): no 4 s manoeuvre from the library gained ≥ 0.1 m of geodesic cost. These are tight spots where the manoeuvre set is too coarse. The goal was reachable in all of them.
- **Moving timeouts:** 700, described above. The time is lost to the 3 s trigger delay before each activation, to turning on the spot, and to the policy steering back after it regains control.
- **Trajectories.** `paper_draft/figures/r3_trajectories.pdf` shows the cases selected by rule: the lowest-seed baseline deadlocks of seed-0 agents, plus the lowest-seed moving timeout of C.

### Pilot (exploratory, 2,000 episodes, seeds 8,300,000+)

The technical criteria were met; see *Amendments*. Its outcome numbers are in `results/pilot/summary.json` and pointed the same way as the main run (C: 0 % deadlock in both agent types). They are not part of the confirmatory evidence.

### Limitations

- **Privileged information.** C and A use the exact obstacle map and the exact state, as the filter does. With only local information (B, C_euclid), a quarter of the episodes still deadlock. A robot that senses obstacles only by range readings would need mapping first.
- **Not a liveness guarantee.** 0.4–0.6 % deadlock and 13.5–14.5 % moving timeouts remain. Nothing is proved about reaching the goal.
- **Agents.** The 10 committed agents (5 seeds per type); expert stage; corrected generator; policy mean; static circular obstacles; simulation only.
- **Parameters.** The recovery parameters were set a priori and not tuned. A's advantage in success suggests the 3 s trigger delay of C is not optimal, but changing it now would be post hoc.
- **Reference point.** A planner with the full map can solve these scenarios by itself (the oracle finds a path in practically all of them). The result shows that deadlock is removable *when the policy is kept in control outside stalls*; it does not show that the learned policy is needed.
- **Timing.** Measured on a laptop under background load; see above.

