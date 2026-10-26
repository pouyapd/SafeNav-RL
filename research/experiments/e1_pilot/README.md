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

*(added after the runs)*
