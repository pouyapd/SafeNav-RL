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

*(added after the runs)*
