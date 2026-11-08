# Round 2: corrected generator (E3), causes of self-stops (E4), DWA-style baseline

This protocol was committed **before** any of the runs below. Results are added later, below the line marked *Results*. E1a, which runs in parallel, has its own record: [`../e1a/README.md`](../e1a/README.md).

Code used here is opt-in, and the defaults are unchanged. A 4,096-step default training run gives bit-identical weights to `exp/e1-pilot` (`bad41e9`). The new code:
- `env.goal_sampling: rejection`;
- `network.distribution: beta`;
- `DynamicWindowPolicy` in `safenav/agents/baselines.py`;
- the runner [`run_eval2.py`](run_eval2.py).

All episodes use the filter on, unless a row says otherwise.

## Scenario sets (all previously unused; checked with `git grep` before this commit)

| Set | Seed base | Generator | Use |
|---|---|---|---|
| DWA tuning | 1,300,000 | default | 100 expert scenarios; choose the DWA weights. Validation-type, never a test set |
| E3 intermediate | 7,500,000 | **rejection** | 1,000 scenarios |
| E3 expert | 7,600,000 | **rejection** | 1,000 scenarios |
| E3-dense (exploratory) | 6,000,000 | rejection, expert with **15–25 requested obstacles** | 500 scenarios. This is the "Test-OOD-dense" set reserved in `research/EXPERIMENT_PLAN.md` |
| E4 primary | 6,500,000 | default | 500 expert scenarios |
| Fresh (existing) | 9,000,000 | default | 500 expert scenarios, used in L3; re-run here only for DWA and for descriptive E4 diagnostics on the committed agents |

The rejection generator redraws start, heading and goal together until the goal lies inside the workspace at a distance in the stage range. The start stays in the original square [1.5, 5]², so only the goal-distance artefact changes (a single-factor change). This is a documented deviation from the "start anywhere" sketch in `paper_draft/minimum_experiments.md`. In the intermediate and expert stages, the joint redraw shifts starts slightly away from (5, 5), where no goal 5 m away fits.

## E3: do the findings hold on a corrected generator? (evaluation only, confirmatory)

- **Agents:** the 10 committed agents (`results/checkpoints/`), with mean and sampled actions. Baselines: potential field, go-to-goal, and DWA (frozen weights; see below).
- **Primary hypothesis H3:** in the expert stage, on 1,000 rejection-generator scenarios, the four expert stall contrasts behave as on the original test set. The contrasts and the criterion for each:
  1. **Type I, trained with − without the filter (mean actions):** keeps its negative sign; the 95 % interval excludes 0.
  2. **Type II, trained with − without the filter (mean actions):** keeps its positive sign; the interval excludes 0.
  3. **Type II, filter-trained, sampled − mean:** keeps its negative sign; the interval excludes 0.
  4. **Type I, filter-trained, sampled − mean:** this was a **null** on the original test set (+0.8 [−7.6, 10.0]), so the criterion is that its interval **includes** 0.

  The plan in `minimum_experiments.md` required all four to exclude 0. That cannot apply to an original null, so it is corrected here, before any run.
- **Statistics:** paired hierarchical bootstrap with 10,000 replicates and scenarios paired.
  - Across agent types, seeds are resampled independently (contrasts 1–2).
  - Within an agent type, seeds are paired (contrasts 3–4).

  The intermediate stage is reported as secondary, with the same four contrasts.
- **H3 outcome:** supported if all four criteria hold in the expert stage; partly supported if 2–3 hold; not supported otherwise.
- **Secondary:**
  - success by stage, with seed-level t-intervals over 5 seeds and the bootstrap;
  - success by start-goal distance bin;
  - collisions, which must be 0 with the filter on;
  - comparison with the clipped generator on the fresh set (9,000,000, from L3).
- **E3-dense (exploratory):** the same agents and baselines, mean actions, 500 scenarios. Descriptive only.

## E4: what causes the self-stops? (training; confirmatory if all runs complete)

The two candidate explanations come from the manuscript (Discussion, Hypothesis 1):
- **aliasing:** clipped speed samples `v ≤ 0` all execute as 0;
- **the reward:** the risk penalty.

| Arm | Change from control | Runs |
|---|---|---|
| control | none: this **is the E1a control** (same code path, seeds 0–4, 300k steps, filter on in training) | shared |
| riskfree | `reward.risk_penalty_beta=0.0` | 5 × 300k |
| beta_policy | `network.distribution=beta` (actions always inside the bounds, so no clipping of `v ≤ 0`) | 5 × 300k |

- **Training:** as E1a, with the same driver, queue, stop rules and checkpoint rule (`best.pt`).
- **Evaluation:** 500 expert scenarios with seed base 6,500,000, mean actions (primary) and sampled actions (secondary). Runner: `run_eval2.py`. Arms: control, riskfree, beta_policy.
- **Primary endpoint: policy-side stall rate.** This is the share of episodes that end in a self-stop or an "other stall", that is, a stall where the filter limited the speed in fewer than half of the last 100 steps. A Beta policy never commands exactly 0, so its stops can appear as "other stall"; Type II alone would be biased against it. Type II is reported separately.
- **Hypotheses:**
  - **H4a (aliasing):** beta_policy − control ≤ −10 points, with the 95 % interval excluding 0.
  - **H4b (reward):** riskfree − control ≤ −10 points, with the interval excluding 0.
- **Interpretation:**
  - **Aliasing supported:** H4a holds and H4b does not.
  - **Reward supported:** H4b holds and H4a does not.
  - **Both:** both hold.
  - **Neither:** neither holds.
  - Holm adjustment over the two primary tests.
- **Statistics:** paired hierarchical bootstrap (scenarios paired, seeds resampled independently per arm). Also reported: per-seed rates, mean ± SD over seeds, and seed-level t-intervals.
- **Secondary:** success; Type I; Type II; collisions; sampled actions.
- **Caveats, stated in advance:**
  - The Beta policy changes the exploration distribution, not only the clipping.
  - Removing the risk penalty changes the incentive to keep clearance.
  - Neither arm was tuned.
- **If the runs cannot all finish** (the laptop is on battery; the queue pauses below 35 %): E4 is reported as **not executed** or **partly executed**, with no confirmatory reading. Unfinished arms are not replaced by shorter runs without a new, labelled pilot protocol.

**E4-diagnostics (descriptive, no training).** The 5 committed filter-trained agents are re-run on the fresh set (9,000,000, mean actions) with extra logging. In self-stops, the following are recorded: the raw policy speed command before clipping (`tail_mean_raw_v`, `tail_share_raw_v_nonpositive`), the risk score at the end, and the barrier value. These describe *where* in action space the policy stops; they do not establish a cause.

## DWA-style baseline (evaluation only)

The planner is described in the `DynamicWindowPolicy` docstring. It uses the same observation as PPO and the potential field (16 range readings, goal distance and bearing; no map).
- **Tuning (validation only):** 12 configurations: `w_heading ∈ {0.2, 0.5, 1.0}` × `w_clearance ∈ {0.2, 0.5}` × `clearance_min ∈ {0.05, 0.15}` m; other weights at their defaults. Each runs 100 expert scenarios from seed base 1,300,000, with the filter on. The configuration with the highest success is selected; ties go to fewer timeouts, then to the lower index. It is then frozen.
- **Evaluation:**
  - the 300 committed test scenarios (filter on and off), comparable with `results/benchmark/`;
  - fresh expert scenarios (9,000,000; filter on), comparable with L3;
  - the E3 and E3-dense sets.
- **Reporting:** success, collision and timeout with Wilson intervals over episodes (DWA is deterministic), plus the stall taxonomy. The comparison with PPO is **descriptive**:
  - PPO intervals are over seeds;
  - PPO hyperparameters were not tuned;
  - the DWA weights were tuned on 100 validation scenarios.

  No claim of superiority is made in either direction without overlapping-interval caveats.

## Not executed in this round, with reasons

- **PPO-Lagrangian:** it needs a cost signal design and about 10 training runs, comparable to E1a. It also optimises a different objective: expected constraint cost instead of per-step filtering, so a fair comparison needs its own protocol.
- **MPC with a DT-CBF:** a substantial solver implementation, not feasible in this round.
- **E1b:** E1a must come first; it is also expensive (5 × 1M steps).

## Commands

```bash
# DWA tuning (one invocation per configuration, i = 0..11, see tune_dwa.py)
python research/experiments/round2/tune_dwa.py
# E3 (example for one stage; the remaining invocations are in results/*/run_record.json)
python research/experiments/round2/run_eval2.py --tag e3_expert_mean --agent filtered=results/checkpoints/filtered \
    --agent unfiltered=results/checkpoints/unfiltered --baseline potential_field --baseline gotogoal --baseline dwa \
    --stages expert --seed-base 7600000 --n 1000 --set env.goal_sampling=rejection
# E4 training
python research/experiments/tools/run_queue.py --queue research/experiments/round2/e4_queue.json --max-hours 8
```

---

## Results

All numbers below come from `results/<tag>/episodes.csv` through [`analyze_round2.py`](analyze_round2.py), which writes `results/summary.json`. Figures come from [`make_figures.py`](make_figures.py). Plots are in `paper_draft/figures/`: `e3_outcomes.pdf`, `dwa_filter.pdf` and `e4_seeds.pdf`.

**Executed code.** Every evaluation ran at commit `3aed35e` or later, as recorded in each `run_record.json`. An invocation that was killed when a session ended leaves no record entry. Its rows were kept and the run resumed: rows are keyed by (agent, seed, stage, scenario), so no episode is counted twice. Peak memory was under 210 MB per process.

**Status of each experiment**

| Experiment | Status |
|---|---|
| DWA tuning (12 configurations) and DWA evaluation | **completed** |
| E3, expert and intermediate, mean and sampled actions | **completed (confirmatory)** |
| E3-dense | **completed (exploratory)** |
| E4-diagnostics | **completed (descriptive)** |
| E4 training arms and evaluation | **completed (confirmatory)**; see below |
| Exploratory DWA margin ([note](exploratory_dwa_margin.md), committed before its runs) | **completed (exploratory)**; see the DWA section |

### Check across runners

The E4-diagnostic run of the 5 filter-trained agents on the fresh set reproduces L3 exactly:
- 63.9 % success;
- 577 self-stops;
- 297 filter-blocks;
- 29 moving timeouts.

L3 used `research/experiments/lightweight/run_eval.py`; this run used `run_eval2.py`. The new runner therefore does not change outcomes.

### E3: corrected generator (start-goal distances honoured)

- **The generator works as intended.**
  - Expert start-goal distances run from 5.00 to 9.00 m, median 5.76 m.
  - Intermediate distances run from 5.00 to 7.98 m, median 5.74 m.
  - The joint redraw never fell back (0 fallbacks).
  - There were **0 collisions** across all 52,500 E3 and E3-dense episodes (46,000 + 6,500). The filter is on in all of them, and the minimum clearance is 0.30 m.

**H3 contrasts.** Paired hierarchical bootstrap with 10,000 replicates; the intervals include seed resampling. Values are in points.

| Contrast | Expert | Criterion met? | Intermediate (secondary) | Criterion met? |
|---|---:|---|---:|---|
| 1. Type I, trained with − without the filter (mean) | −29.7 [−37.7, −20.6] | yes | −14.3 [−19.0, −9.5] | yes |
| 2. Type II, trained with − without the filter (mean) | +34.1 [+22.9, +42.9] | yes | +18.9 [+12.9, +24.0] | yes |
| 3. Type II, filter-trained, sampled − mean | −35.8 [−43.5, −25.7] | yes | −21.1 [−26.0, −15.3] | yes |
| 4. Type I, filter-trained, sampled − mean (the interval must include 0) | **+7.3 [+2.5, +14.3]** | **no** | −0.7 [−3.5, +2.0] | yes |
| **H3** | **partly supported (3 of 4)** | | supported (4 of 4) | |

**Reading.**
- The two-mechanism picture holds on the corrected generator:
  - training with the filter trades Type I stalls for Type II stalls;
  - sampled actions remove most Type II stalls.
- The original null in contrast 4 does **not** replicate in the expert stage. With honest long goal distances, sampling turns part of the self-stops into filter-blocks: the filter-trained agents' Type I rises from 21.0 % to 28.3 %. On the original test set this effect was not detectable (+0.8 [−7.6, +10.0]).
- H3 is therefore only partly supported. The claim that "sampling does not affect Type I" should be dropped.

**Success.** Seed-level mean ± SD over 5 seeds, t-interval over seeds; baselines with Wilson intervals over episodes.

| Agent | Expert, mean | Expert, sampled | Intermediate, mean | Intermediate, sampled |
|---|---:|---:|---:|---:|
| PPO trained without the filter | 43.1 ± 4.0 [38.2, 48.1] | 53.2 ± 3.1 [49.3, 57.0] | 73.8 ± 3.0 [70.1, 77.6] | 80.1 ± 2.1 [77.4, 82.7] |
| PPO trained with the filter | 37.2 ± 6.7 [29.0, 45.5] | 55.6 ± 7.9 [45.7, 65.4] | 66.0 ± 5.6 [59.1, 73.0] | 78.5 ± 6.1 [70.9, 86.1] |
| Potential field | 57.0 [53.9, 60.0] | – | 83.5 [81.1, 85.7] | – |
| DWA-style (frozen) | 15.3 [13.2, 17.7] | – | 36.9 [34.0, 39.9] | – |
| Go-to-goal | 9.8 [8.1, 11.8] | – | 29.4 [26.7, 32.3] | – |

**Further observations.**
- **Effect of the generator.** On the generator that clips goals (fresh set, L3), the same agents reached 63.9 % (with filter) and 68.3 % (without filter) expert success. With goals at their requested distance they reach 37.2 % and 43.1 %.
  - A large part of the published expert success came from goals that clipping moved closer. Success also falls with distance: for filter-trained agents, 38.8 % at 5–7 m against 21.1 % at 7–9 m.
  - The two sets differ in scenarios as well as in generator, so this comparison is descriptive.
- **Success, trained with − without the filter (mean actions):**
  - expert: −5.9 points [−12.5, +0.3], p = 0.064;
  - intermediate: −7.8 [−13.2, −2.3], p = 0.003.

  The filter-trained agents are not more successful, and in the intermediate stage they are less successful.
- **The untuned potential field is the strongest method here.** Its timeouts are mostly "moving" (oscillation or wandering), not stalls; its success interval lies above both PPO intervals. This is descriptive: PPO intervals are over seeds and the baselines' over episodes, and no method was tuned on these sets.

**E3-dense (exploratory; 15–25 requested obstacles, expert, 500 scenarios, mean actions).** All methods lose success relative to the E3 expert set, and the ranking is unchanged:
- potential field: 50.8 % [46.4, 55.2];
- PPO trained without the filter: 37.9 %;
- PPO trained with the filter: 31.1 %;
- DWA: 10.8 %;
- go-to-goal: 7.4 %.

There were 0 collisions.

### E4-diagnostics (descriptive; 5 committed filter-trained agents, 500 fresh expert scenarios)

- In **all 577 self-stops**, the policy's raw speed command was ≤ 0 in at least half of the last 100 steps (100 %). The median raw command over those steps was −0.20 (IQR −0.43 to −0.06), and in 20.8 % it was below −0.5.
- In 70.7 % of the self-stops the risk score at the end was exactly 0. The risk-penalty term was therefore not active at the stall position in most of them. The median barrier value at the end was 0.41 m.
- **Reading.** The policies stop by commanding speeds below the action bound, which clipping maps to exactly 0. This is consistent with the aliasing explanation. That the risk term is mostly inactive at the end argues against a *current* reward incentive, although the reward could still have shaped the policy during training. This analysis shows *where* in action space the policies stop; it does not establish the cause. That is the job of E4.

### DWA-style baseline

**Tuning (100 validation scenarios, filter on).** All 12 configurations reached 37–39 % success with 0 collisions, and every timeout was filter-blocked. The rule selected configuration 2 (`w_heading = 0.2`, `w_clearance = 0.5`, `clearance_min = 0.05`). The grid barely mattered, a first sign that the filter, not the weights, limits this planner.

**Evaluation (frozen configuration).** Wilson 95 % intervals.

| Set | Filter | Beginner | Intermediate | Expert | Collisions | Timeout classes |
|---|---|---:|---:|---:|---:|---|
| Committed test (100 per stage) | off | 100 % [96.3, 100] | 94 % [87.5, 97.2] | **96 % [90.2, 98.4]** | 0 | 10 self-stop |
| Committed test (100 per stage) | on | 95 % [88.8, 97.8] | 64 % [54.2, 72.7] | **48 % [38.5, 57.7]** | 0 | 93 filter-blocked |
| Fresh (500 expert) | on | – | – | 50 % [45.6, 54.4] | 0 | 250 filter-blocked |

**Reading.**
- Without the filter, this planner solves 96 % of the committed expert test scenarios with no collisions. Its minimum clearance, however, was 0.027 m, against the filter's 0.30 m margin.
- With the filter, every timeout is filter-blocked. The planner steers into the band the filter forbids, and then stays there.
- So **Type I deadlock is not specific to learned policies**: a classical local planner stalls in front of this filter in the same way. This motivated the exploratory margin follow-up below.
- The DWA-off result is **not** a fair safety comparison with filtered PPO: it keeps a much smaller margin.

**Exploratory: a DWA whose own margin respects the filter margin** ([note](exploratory_dwa_margin.md); committed in `8702fb7` before these runs; post hoc, descriptive). The frozen weights are kept and only `clearance_min` changes. The filter is on throughout; Wilson 95 % intervals.

| Set | Frozen (`clearance_min` 0.05 m) | 0.35 m | 0.45 m |
|---|---:|---:|---:|
| Committed test, expert (100) | 48 % [38.5, 57.7]; 52 filter-blocked | **63 % [53.2, 71.8]; 0 filter-blocked, 37 self-stop** | 55 % [45.2, 64.4]; 0 filter-blocked, 45 self-stop |
| Committed test, intermediate (100) | 64 % [54.2, 72.7] | 82 % [73.3, 88.3] | 76 % [66.8, 83.3] |
| Fresh expert (500) | 50.0 % [45.6, 54.4]; 250 filter-blocked | **67.8 % [63.6, 71.7]; 0 filter-blocked, 160 self-stop** | 59.0 % [54.6, 63.2]; 0 filter-blocked, 204 self-stop |
| E3 expert, corrected generator (1,000) | 15.3 % [13.2, 17.7]; 847 filter-blocked | **43.6 % [40.6, 46.7]; 0 filter-blocked, 564 self-stop** | 29.1 % [26.4, 32.0]; 0 filter-blocked, 702 self-stop |

There were 0 collisions throughout. The minimum clearance was 0.31 m with a margin of 0.35 m, and 0.38 m with 0.45 m.

**Reading (exploratory).**
- Once the planner's own margin exceeds the filter margin, **filter-blocked stalls disappear completely** in all three sets, and success rises by 15–28 points.
- The timeouts that remain are all "self-stops". Here this means the planner found no admissible rollout and turned in place (commanded speed 0). This is the planner's own local minimum, the classical counterpart of a policy-side stop. A larger margin makes it more frequent.
- So for a classical planner, Type I deadlock behind this filter is a **margin mismatch** that a filter-aware design avoids. What remains is a planning problem: no global route, local minima.
- This parallels the PPO results: training with the filter removes most Type I stalls, and Type II stops take their place.
- **Limitations.** The variants were chosen after seeing the frozen DWA's results and were not tuned. The test, fresh and E3 sets were all already used in this round. Only one weight configuration was tried.

### E4: what causes the self-stops? (completed, confirmatory)

**Execution.**
- All 10 new runs (riskfree and beta_policy, seeds 0–4) reached 301,056 steps on mains power. Each records exactly one override, and none had modified code (`records/`).
- Nine ran at `3aed35e`. Beta seed 4 ran at `8702fb7`, which only adds a note and a statistics tool, so training code is unchanged.
- The queue took 3.7 h of wall time, including waits.
- The control arm is the E1a control (`runs/e1a/control`, copied to `../e1a/checkpoints/control`). The new checkpoints are in `checkpoints/<arm>/seed_<s>/`, and their SHA-256 hashes are in `results/e4_*/run_record.json`.
- Evaluation: 500 expert scenarios (seed base 6,500,000) × 15 agents × {mean, sampled} = 15,000 episodes. There were **0 collisions**.

**Primary endpoint: policy-side stall rate** (self-stop + other stall; mean actions). Paired hierarchical bootstrap, Holm over the two tests.

| Arm | Policy-side stall, per seed 0–4 | Mean ± SD | Difference to control [95 % CI] | p (Holm) | Criterion (≤ −10 points, CI < 0) |
|---|---|---:|---:|---:|---|
| control | .188 .390 .334 .344 .436 | 33.8 ± 9.3 % | – | – | – |
| riskfree (`risk_penalty_beta = 0`) | .062 .010 .020 .010 .008 | 2.2 ± 2.3 % | **−31.6 [−39.4, −22.9]** | < 0.001 | **H4b holds** |
| beta_policy | .000 .000 .000 .000 .000 | 0.0 ± 0.0 % | **−33.8 [−41.3, −25.2]** | < 0.001 | **H4a holds** |

**Pre-registered reading: "both".** Each intervention on its own removes almost all policy-side stalls. Seed-level Welch tests agree: p = 0.001 for both, Hedges' g −4.2 (riskfree) and −4.6 (beta).

**Secondary (mean actions; unadjusted).**

| Arm − control | Success | Type I (filter-blocked) | Type II (self-stop) | Total deadlock |
|---|---:|---:|---:|---:|
| riskfree | **+6.2 [+1.2, +11.2]** (p = 0.016) | **+26.2 [+21.0, +31.4]** | −31.6 [−39.3, −22.9] | −5.4 [−10.7, 0.0] (p = 0.051) |
| beta_policy | −0.2 [−6.3, +5.5] | **+33.6 [+27.3, +40.1]** | −33.8 [−41.2, −25.2] | −0.3 [−6.6, +6.4] |

| Arm | Success, mean actions (seed mean ± SD, t-CI) | Success, sampled |
|---|---:|---:|
| control | 59.0 ± 5.2 [52.5, 65.5] | 71.3 [68.2, 74.4] |
| riskfree | 65.2 ± 2.1 [62.6, 67.9] | 70.4 [67.1, 73.7] |
| beta_policy | 58.8 ± 4.4 [53.3, 64.2] | 66.8 [60.2, 73.3] |

**Check of the classification.**
- The Beta agents' 985 timeouts are filter-blocked stalls at the boundary: end barrier median 0.000 m, 90th percentile 0.000 m. The policy's raw speed command there has a median of +0.52.
- They are therefore not hidden stops. The Beta agents push forward into the filter.
- The control's self-stops end at a median barrier of 0.53 m, with a raw command of −0.13.

**Reading.**
- Both candidate causes are sufficient to remove the self-stops. Bounded actions (no clipping of `v ≤ 0`) and a reward without the risk term each remove them on their own. So the explanations are not exclusive, and neither alone is *the* cause.
- More important, **neither intervention removes the deadlock**. The stalls move from before the boundary (self-stop) to the boundary (filter-blocked):
  - total deadlock is unchanged for the Beta policy (39.4 % against 39.7 %);
  - it is only slightly lower without the risk term (34.2 %);
  - success rises by 6 points only in the riskfree arm.
- The same substitution appeared with the correction penalty (E1a) and, in reverse, with the margin-aware DWA. In this setting, self-stopping looks like the policy's learned alternative to being held at the boundary, not an independent failure.
- **Caveats (stated in advance).**
  - The Beta policy also changes exploration.
  - Removing the risk term also removes an incentive to keep clearance.
  - Neither arm was tuned.
  - These are 300k-step agents.
