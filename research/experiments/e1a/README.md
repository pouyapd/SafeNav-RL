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
