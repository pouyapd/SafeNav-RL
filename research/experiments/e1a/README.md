# E1a: correction penalty vs. control, 5 seeds × 300k steps per arm (confirmatory)

This execution record was committed **before** the first E1a training step. The scientific protocol is the pre-registered one in [`../E1_correction_penalty.md`](../E1_correction_penalty.md), and it is **unchanged**:
- the question and hypotheses H1/H0;
- arms, seeds and budget;
- evaluation scenarios;
- primary and secondary endpoints;
- statistics and decision table.

This file adds only execution details. Results are added later, below the line marked *Results*.

## Design (copied from the protocol for convenience; the protocol prevails)

| Item | Value |
|---|---|
| Arms | **control**: default config, filter on during training. **penalty**: identical plus `safety.correction_penalty=1.0` (α = 1), the only difference |
| Seeds, budget | training seeds 0, 1, 2, 3, 4 in both arms; 300,000 steps per run; 10 runs |
| Checkpoint | `best.pt` (existing validation rule; validation every 50k steps, 60 episodes) |
| Evaluation | `research/experiments/e1_evaluate.py`, unchanged. Arms: penalty, control, and the committed 1M-step filtered agents as a reference. Every agent is evaluated with the filter on, with mean and with sampled actions, on two sets: **500 fresh expert scenarios, seed base 5,000,000 (primary; never used before)** and the 300 committed test scenarios (secondary) |
| Primary endpoints | penalty − control on the fresh scenarios with mean actions: **deadlock rate** (Type I + Type II + other stalls) and **success**; paired hierarchical bootstrap (scenarios paired, seeds resampled), Holm adjustment over these two tests |
| Decision table | as in the protocol: H1 needs a deadlock reduction of at least 10 points with the Holm-adjusted 95 % interval excluding 0, success rising, and 0 collisions |
| Secondary | sampled actions, committed scenarios, Type I and Type II separately, penalty − committed |

## Execution (no scientific effect)

- **Driver:** `research/experiments/e1_pilot/train_pilot.py`. Its trained weights were verified bit-identical to `python -m safenav train` in the E1 pilot. Chunks of 2 updates with exact resume. One process, one torch thread, below-normal priority.
- **Order:** the runs go through `research/experiments/tools/run_queue.py` in the order control 0, penalty 0, control 1, …, penalty 4. The order does not affect results, because each run is deterministic given its seed.
- **Stop rules (resources only):**
  - free memory below 1,000 MB;
  - process memory above 1,200 MB;
  - a chunk slower than 25 steps/s. The pilot used 60; it is lowered here because validation chunks on battery run at about 60 steps/s.
  - on battery below 35 % charge.

  A stopped run resumes exactly.
- **No peeking:** no evaluation of any E1a agent before all 10 runs have finished. Training logs are read only for safety checks: the logged `correction_penalty_mean` must be > 0 in the penalty arm and 0 in the control.
- **Outputs:**
  - Runs go to `runs/e1a/<arm>/filtered/seed_<s>/`, which is git-ignored. The `best.pt` and `final.pt` checkpoints, about 0.6 MB each, are copied to `checkpoints/<arm>/seed_<s>/` in this folder so that the evaluation can be reproduced.
  - Run records go to `records/`.
  - The evaluation goes to `eval/` (`episodes.csv`, `summary.json`).
- **Not part of this round:** E1b (penalty arm at 1M steps). It costs about 5 × 1M steps and is run only if E1a is positive or inconclusive.

## Commands

```bash
python research/experiments/tools/run_queue.py --queue research/experiments/e1a/queue.json --max-hours 8
# after all 10 runs:
python research/experiments/e1_evaluate.py --arm penalty=runs/e1a/penalty/filtered \
    --arm control=runs/e1a/control/filtered --arm committed=results/checkpoints/filtered \
    --out research/experiments/e1a/eval --jobs 1
```

---

## Results

*(added after the runs)*

**Status: completed (confirmatory), 2026-11-08.** All 10 runs reached 301,056 steps (the last PPO update ends just past 300k). The evaluation of all 15 agents (24,000 episodes) was run once with the pre-registered command, after all training had finished.

### Execution facts

- **Training.** 1.75 h of summed wall time, laptop on mains power, one process at a time. The control-0 run was interrupted by the end of a session and resumed exactly from its last chunk; `records/train_control_seed0.json` lists every invocation.
- **Code.** Training ran at `d65e576`/`3aed35e`; the default code path is bit-identical to `bad41e9`. The run records show `code_modified_since_commit: false`.
- **Sanity checks (pre-registered).**
  - `correction_penalty_mean` is > 0 in the penalty arm (seed 0, last update: 0.028) and 0 in the control.
  - There were **0 collisions in all 24,000 episodes.**
- **Evaluation.** About 2.3 h of wall time in one process, sharing the CPU with the round-2 evaluations. Outputs:
  - [`eval/episodes.csv`](eval/episodes.csv);
  - [`eval/summary.json`](eval/summary.json);
  - [`eval_log.txt`](eval_log.txt);
  - the checkpoints, in [`checkpoints/`](checkpoints/).

### Primary result (500 fresh expert scenarios, mean actions, penalty − control)

| Endpoint | Penalty (mean of 5 seeds) | Control | Difference [95 % CI] | p (bootstrap) | p (Holm) |
|---|---:|---:|---:|---:|---:|
| Deadlock rate (Type I + II + other stall) | 35.5 % | 40.6 % | **−5.1 points [−11.6, +1.7]** | 0.14 | 0.28 |
| Success | 58.7 % | 58.2 % | **+0.5 points [−5.8, +6.3]** | 0.87 | 0.87 |
| Collisions | 0 | 0 | 0 | – | – |

- Intervals come from the paired hierarchical bootstrap: 10,000 replicates, with scenarios paired and seeds resampled within each arm. **These intervals therefore include seed-level uncertainty; they are not scenario-only intervals.**
- **Decision (pre-registered table):** the deadlock reduction is below 10 points and its interval includes 0, so **H1 is not supported and H0 is retained.** In the decision table this is the row *"No significant change: the penalty does not resolve deadlock in navigation"*, which here means at 300k steps with α = 1 and this filter.

### Secondary results (not part of the decision; unadjusted)

| Contrast (fresh, mean actions) | Difference [95 % CI] | p (bootstrap) |
|---|---:|---:|
| Type I (filter-blocked), penalty − control | **+14.0 points [+2.4, +26.6]** | 0.016 |
| Type II (self-stop), penalty − control | **−19.7 points [−32.2, −6.9]** | 0.002 |
| Deadlock, sampled actions | +2.1 [−2.9, +7.2] | 0.44 |
| Success, sampled actions | −4.1 [−9.1, +1.0] | 0.11 |
| Success, penalty (300k) − committed (1M), mean actions | −5.0 [−9.1, −0.8] | 0.02 |

**Seed-level view.** This is a supplementary analysis added after the runs, using `research/experiments/tools/seed_level_stats.py`: per-seed rates, Welch's t-test over seeds, and Hedges' g. The full output is in [`eval/seed_level_fresh_mean.json`](eval/seed_level_fresh_mean.json).

| Metric (fresh, mean) | Penalty, per seed 0–4 | Control, per seed 0–4 | Mean ± SD (penalty / control) | Welch p | Hedges' g |
|---|---|---|---|---:|---:|
| Deadlock | .330 .328 .400 .380 .338 | .308 .466 .366 .408 .484 | .355 ± .033 / .406 ± .072 | 0.20 | −0.82 |
| Success | .574 .614 .592 .606 .548 | .676 .532 .608 .586 .508 | .587 ± .026 / .582 ± .066 | 0.89 | 0.09 |
| Type I | .064 .108 .400 .332 .148 | .138 .038 .070 .054 .052 | .210 ± .147 / .070 ± .039 | 0.10 | 1.17 |
| Type II | .260 .214 .000 .044 .176 | .170 .428 .296 .354 .430 | .139 ± .112 / .336 ± .108 | 0.022 | −1.62 |

**Reading.** The penalty did not reduce the total deadlock rate by the pre-registered margin, and it did not raise success. What it changed is the *type* of stall:
- With mean actions, self-stops fell by about 20 points and filter-blocks rose by about 14 points.
- In seeds 2 and 3 the self-stops almost vanished (0.0 % and 4.4 %), but filter-blocks rose to 40 % and 33 %.

This fits the second row of the decision table ("the penalty removes one mechanism only"). It is consistent with the two-mechanism taxonomy: a policy that is penalised for commanding actions the filter must correct stops commanding `v ≤ 0`, and is then held by the filter instead.

These are secondary, unadjusted contrasts from 5 seeds with large between-seed spread (Type I SD 0.15 in the penalty arm). They suggest, and do not establish, a substitution of one mechanism for the other.

With sampled actions, neither the deadlock rate nor the stall types differ between the arms.

### Limitations

- The training budget is 300k steps against 1M for the committed agents; the penalty arm is 5 points less successful than the committed agents.
- Only α = 1 was tested, with no tuning of α.
- The penalty uses the one-step filter of this repository, not the setting of the original paper; there are no safe resets.
- E1b (1M steps) was not run. Under the protocol, E1b runs "only if E1a is positive or inconclusive". The primary result is a clear null for total deadlock, but the secondary shift between stall types may justify E1b as an **exploratory** follow-up, not as a confirmation of H1.
