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

*(added after the runs)*
