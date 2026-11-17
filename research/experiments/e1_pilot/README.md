# E1 pilot: correction penalty vs. control, one seed, 150k steps per arm

Branch `exp/e1-pilot`. This protocol was written and committed **before** the two pilot training runs. Results are added in a later commit, below the line marked *Results*.

## Purpose

The pilot is **not** a test of the penalty's effect. It has one training seed per arm, so it cannot separate a penalty effect from seed-to-seed variation. It answers three practical questions before the full E1a (2 arms × 5 seeds × 300k steps; [`../E1_correction_penalty.md`](../E1_correction_penalty.md)):

1. Does the penalty arm train normally: does it reach the expert stage, with no collisions under the filter?
2. Does the control show the stalls at this reduced budget? If not, a short-budget E1a could not test the penalty against them.
3. Is there a **large** difference, larger than the observed spread between training seeds?

## Configuration audit (done before this protocol)

| Check | Result |
|---|---|
| Where the penalty is implemented | `SafetyFilterWrapper` (`safenav/safety/cbf_filter.py`). With α > 0 the reward becomes r − α‖u − u_exec‖². `u` is the raw policy sample; `u_exec` is the executed action after the action bounds **and** the filter. |
| Training only | `make_env(..., training=True)` is used only by `PPOTrainer`; evaluation environments get α = 0. Existing test `test_correction_penalty_value_and_training_only`. |
| The hazard noted earlier | The code exists only on `exp/e1-correction-penalty` (commit `0b05aa3`). This branch carries it as cherry-pick `d8ccdca` (`-x`, so the origin is recorded). On branches without it, the config silently ignores the key. |
| The penalty reaches the PPO reward | New: the trainer now logs `correction_penalty_mean` and `filter_gap_sq_mean` (‖clip(u) − u_exec‖², the filter's share, logged in both arms) per update in `progress.csv`. New test `test_correction_penalty_reaches_training_reward`: the control logs 0, the penalty arm logs > 0 at every update, both see the same first rollout, and the final weights differ. |
| Control = original training code | A 4,096-step control run with the pilot driver, chunked into one-update pieces with exact resume, produced **bit-identical weights and validation results** to `python -m safenav train --seed 0 --timesteps 4096` on `main` (`d257eca`). |
| Treatment config | A 4,096-step penalty smoke run recorded `correction_penalty: 1.0` in `run_info.json`. The logged penalty was 0.25 and 0.21 per step; the control logged 0. |
| Tests | `tests/test_ppo.py`, `tests/test_safety_filter.py`, `tests/test_heading_escape.py`: 29 passed. |

**Seen in the smoke run (before this protocol).** At the start of training, the filter-only part ‖clip(u) − u_exec‖² is about 0.046 per step, while the full penalty is about 0.23. So **about 80 % of the early penalty comes from raw Gaussian samples outside the action bounds** ([0, 1] for v, [−1, 1] for ω), not from the filter. This follows the published definition, which includes the input constraints, and the E1 plan. It is kept as specified, but it matters for interpretation: at α = 1 the penalty also acts as a strong push to keep the Gaussian inside the bounds.

## Resource smoke test (measured 2026-10-26, mains power)

- One process, one torch thread, below-normal priority, peak memory **≈ 320 MB**.
- Free system memory was 1.6–3.6 GB.
- Speed depends strongly on machine load: 25–64 steps/s while other work ran, 350–790 steps/s when idle.
- Estimate for 150k steps per arm, including 3 validations: **≈ 5–10 min when idle, up to ≈ 60 min under load**.

## Design (fixed before training)

| Item | Value |
|---|---|
| Arms | control (α = 0, default config) and penalty (α = 1.0, the E1 plan's value); filter on during training; curriculum on; default config otherwise |
| Seed, budget | training seed 0, 150,000 steps per arm; learning rate annealed linearly over the 150k steps in both arms |
| Order | control first, then penalty; **never concurrently** |
| Driver | `train_pilot.py`: chunks of 10 PPO updates (20,480 steps), each ending in an exact resume checkpoint |
| Stop rules | Between chunks, stop if the invocation deadline (1,500 s) is reached, process peak memory > 1,200 MB, or free system memory < 600 MB; the run is then resumed by calling the driver again. **Abort the pilot** if total training wall time exceeds 75 min per arm, or if the machine becomes unresponsive. |
| Checkpoints | `best.pt`: the existing rule, validation at 50k, 100k and 150k steps (60 episodes over the 3 stages). `final.pt`: the 150k-step policy. Both stay local in `runs/e1_pilot/` (git-ignored). |
| Evaluation | `../lightweight/run_eval.py` (plain filter, κ = 0, policy mean), one process: the 500 fresh expert scenarios with seeds 9,000,000–9,000,499, for both arms and both checkpoints |
| Why these scenarios | The committed agents were already evaluated on them (L3), which gives a reference. The seed base 5,000,000 stays **unused**, reserved for the primary test of E1a. |
| Stall classes | As in L3 (last 100 steps, net displacement < 0.10 m): filter-blocked (Type I), self-stop (Type II), other stall. **Deadlock rate** = share of episodes ending in one of these three classes. |

### Pre-registered analysis (`analyze_pilot.py`)

- Rates per arm and checkpoint: success, collisions, minimum clearance, deadlock rate, and timeouts by class.
- Penalty − control on the same 500 scenarios: deadlock, Type I, Type II and success. The paired bootstrap (10,000 replicates) resamples **scenarios only**. With one seed per arm, the intervals do not include training-seed variability, so they are reported but **not** read as significance.
- Training: when each stage was first reached; validation; the best checkpoint's step; the penalty and the filter gap over training; entropy.
- Reference only: the committed filtered seed-0 agent (best checkpoint at 100k steps of a 1M-step run) on the same scenarios.

### Decision rules (`best` checkpoints)

- **Collisions.** Any collision under the filter is an implementation error: stop and debug.
- **G1, both arms reach the expert stage.** If the penalty arm does not, the penalty slows the curriculum at α = 1. Report it; E1a would then need a longer budget or a different α.
- **G2, the control shows the stalls.** If the control's deadlock rate is below 10 %, the reduced budget does not reproduce the phenomenon, and E1a's budget must be reconsidered.
- **G3, effect screen** (penalty − control deadlock rate). For reference, the committed filtered agents' deadlock rates on these scenarios span 30.8–39.0 % across seeds (a range of 8.2 points), and Type II spans 10.2–29.0 %.
  - **≤ −10 points:** a large reduction in this seed. E1a is justified as the confirmation.
  - **Between −10 and +10:** no large effect in this seed; **inconclusive**. One seed cannot rule out a smaller effect, so this does not count as evidence against the penalty.
  - **≥ +10 points:** the penalty is worse in this seed.

**What justifies the full E1a.** If the run has no collisions and G1 and G2 pass, E1a is **technically feasible**. Whether it is **worth running** follows from G3 and is stated in the results, together with its cost.

Secondary, reported but not decisive: final checkpoints, Type I and Type II separately, success, and the share of the penalty due to the filter.

## Commands

```bash
python research/experiments/e1_pilot/train_pilot.py --arm control --seed 0 --timesteps 150000 --deadline-s 1500
python research/experiments/e1_pilot/train_pilot.py --arm penalty --seed 0 --timesteps 150000 --deadline-s 1500
# repeat a command to resume after a stop; then, for arm in control penalty and ck in best final:
python research/experiments/lightweight/run_eval.py --tag eval_<arm>_<ck> --agents runs/e1_pilot/<arm> \
    --variants filtered --checkpoint-name <ck> --stage expert --seed-base 9000000 --n 500 --kappa 0 \
    --deadline-s 600 --out-root research/experiments/e1_pilot/results
python research/experiments/e1_pilot/analyze_pilot.py
```

---

## Results

Run on 2026-10-26 from commit `3fcc5c5`. The training code (`safenav/`, `configs/`) was unmodified in every invocation (`code_modified_since_commit: false`). All numbers below come from `results/summary.json` (`python research/experiments/e1_pilot/analyze_pilot.py`).

### Runs and resources

| Run | Invocations | Wall time of the processes | CPU | Peak memory |
|---|---|---|---|---|
| Training, control (α = 0) | 3 (see deviations) | ≈ 630 s + 130 s + 126 s. In the first invocation, ≈ 320 s after its last resume point were discarded. The trainer's own elapsed time is 505 s | ≥ 150 s recorded (invocation 1 not recorded) | 322 MB |
| Training, penalty (α = 1) | 1 | 203 s | 185 s | 324 MB |
| Evaluation: control best / final, penalty best / final (500 episodes each) | 1 each | 42 / 29 / 34 / 35 s | 38 / 27 / 31 / 33 s | 204 MB |

- **One process at a time** throughout, one torch thread, below-normal priority.
- **No collisions in training or in evaluation.**
- Mains power except where not recorded (invocation 1).

### Deviations from the protocol

1. **Chunk size.** Chunks were 5 updates (invocation 1), then 2, not 10, so that stops happen sooner. Resumption is exact (`tests/test_ppo.py`; also checked bit-identically in the smoke run), so this does not change the trained weights.
2. **Memory incident in control invocation 1.** Another application (Blender, ≈ 3.4 GB of RAM and 90–100 % CPU) reduced free system memory to 830 MB.
   - My process was paged out (working set 28 MB) and slowed to 9–238 steps/s.
   - The process had ended by 18:05, after its last resume point at update 50, but it **did not write its record**. Its 10 completed chunks are reconstructed from the console output (`train_control_seed0.json`, marked `reconstructed_from_console_output`).
3. **Driver changes after the protocol commit.** These affect stopping only, not training:
   - the record is written after every chunk;
   - the default free-memory floor was raised from 600 MB to 1,000 MB;
   - a new stop rule ends the run when a chunk is slower than 60 steps/s.
4. **Pause.** Invocation 2 stopped itself after one chunk at 32 steps/s (the throughput rule) while Blender kept all cores busy. Training then **waited until Blender had closed**: about 70 min of clock time in total for the control arm, of which about 15 min were training. The 75-min limit in the protocol refers to training time; it was not exceeded.
5. **Missing hard limit.** The outer `timeout 570` of Git Bash did not reliably stop the native Windows Python process. The stops relied on the driver's own checks between chunks.

### Training (seed 0, 150k steps per arm)

| | Control | Penalty |
|---|---:|---:|
| First step at the intermediate / expert stage | 47,104 / 61,440 | 63,488 / 79,872 |
| Validation success at 51k / 100k / 152k steps (60 episodes over the 3 stages) | 0.73 / 0.82 / 0.80 | 0.58 / 0.75 / 0.70 |
| `best.pt` taken at | 100,352 | 100,352 |
| Correction penalty per step (mean over training; first 10 / last 10 updates) | 0 | 0.066 (0.134 / 0.048) |
| Filter gap ‖clip(u) − u_exec‖² per step, last 10 updates | 0.130 | 0.026 |
| Expert stage, last 10 updates: intervention rate / policy entropy | 0.45 / 1.75 | 0.29 / 0.68 |

The penalty did what it is designed to do at the level of actions. By the end of training:
- the policy asked for actions that the filter changed **about 5× less** (in squared size);
- the filter intervened in 29 % instead of 45 % of steps;
- the policy's entropy was much lower, consistent with the push to keep samples inside the action bounds.

The penalty arm also advanced through the curriculum later (about 18k more steps to reach expert).

### Evaluation: 500 fresh expert scenarios (seeds 9,000,000–9,000,499), mean actions, plain filter

Rates are per 100 episodes. Penalty − control is paired over the same 500 scenarios; the intervals resample **scenarios only**, so they do not include training-seed variability (see Purpose).

| Checkpoint | Arm | Success | Deadlock | Type I (filter-blocked) | Type II (self-stop) | Collisions |
|---|---|---:|---:|---:|---:|---:|
| best | control | 58.6 | 39.8 | 24.2 | 15.6 | 0 |
| best | penalty | 55.4 | 38.6 | 17.2 | 21.0 | 0 |
| best | **penalty − control** | −3.2 [−7.0, 0.6] | **−1.2 [−5.2, 2.8]** | −7.0 [−11.6, −2.4] | +5.4 [1.2, 9.6] | — |
| final | control | 61.2 | 37.6 | 12.6 | 25.0 | 0 |
| final | penalty | 58.6 | 34.2 | 12.6 | 21.6 | 0 |
| final | penalty − control | −2.6 [−6.2, 1.0] | −3.4 [−7.4, 0.6] | 0.0 [−4.0, 4.0] | −3.4 [−7.8, 1.2] | — |
| reference | committed filtered seed 0 (1M-step run, best at 100k) | 60.2 | 39.0 | 11.2 | 27.8 | 0 |

The other timeouts were still moving at the end (control 8 / 6; penalty 30 / 36), plus 2 "other stall" episodes in the penalty arm. The minimum clearance was 0.300 m in every condition.

### Decisions (pre-registered rules, `best` checkpoints)

| Rule | Result |
|---|---|
| Collisions | 0 in both arms: **pass** |
| G1, both arms reach the expert stage | **pass** (the penalty arm 18k steps later) |
| G2, the control shows the stalls | **pass**: control deadlock rate 39.8 %, similar to the committed seed-0 agent (39.0 %) |
| G3, effect screen | **−1.2 points: no large effect in this seed; inconclusive** |

### What the pilot shows, and what it does not

- **Shown.**
  - The penalty is applied and changes training. The control is the unchanged original training code.
  - The two-arm design runs safely on this laptop: about 3.5 min of training per arm when the machine is idle, about 320 MB of memory, and no collisions.
  - At 150k steps the control already reaches the stall level of the committed agents. A reduced-budget E1a can therefore test the penalty against the stalls.
- **Suggested, single seed only (not evidence).**
  - The penalty sharply reduced how often and how much the filter had to correct the policy, but this did **not** show up as a large drop in deadlocks: −1.2 points at the best checkpoint, −3.4 at the final one.
  - At the best checkpoint, deadlocks shifted from Type I to Type II. At the final checkpoint they did not. This inconsistency is what one seed and one checkpoint can produce by chance.
- **Not shown.**
  - Whether the penalty reduces deadlocks by a smaller amount, or not at all.
  - Any effect on success.
  - Anything about other seeds, α values or budgets.
  - The rule allows only "large effect" or "inconclusive", and the result is inconclusive.
- **Interpretation caveat.** Most of the early penalty came from samples outside the action bounds, not from the filter (see the audit). A reduced filter gap together with unchanged deadlocks is compatible with a penalty that mainly shapes the action distribution. E1a cannot separate these parts. A filter-only penalty, ‖clip(u) − u_exec‖², would be a separate, non-published variant.
