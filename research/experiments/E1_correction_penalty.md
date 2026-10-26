# Experiment E1: does the published correction penalty alone remove the deadlocks?

Status (2026-10-26):
- **E1 pilot: done.** One seed, 2 arms × 150k steps, on branch `exp/e1-pilot`; see [`e1_pilot/README.md`](e1_pilot/README.md).
  - The configuration was verified: the penalty reaches the PPO reward, and the control is bit-identical to `main`.
  - Result: no collisions; the control shows the stalls; **no large effect in this seed (deadlock −1.2 points): inconclusive.**
- **E1a: not run** (needs the author's approval). E1b: not run.
- The code was first written on branch `exp/e1-correction-penalty` (`0b05aa3`); `exp/e1-pilot` carries it as cherry-pick `d8ccdca`.

## Question and hypothesis

- **Research question.** Does adding the correction penalty of Pizarro Bejarano, Brunke & Schoellig (RA-L 2025, doi:10.1109/LRA.2024.3512374) to filter-in-the-loop PPO remove the stalls (Type I filter-blocked, Type II self-stop) that limit filtered success on the expert stage?
- **H1 (penalty resolves deadlock).** With α = 1, the expert deadlock rate (Type I + Type II + other stalls) with mean actions falls by at least 10 percentage points against an identical control without the penalty, and expert success rises, with zero collisions.
- **H0.** The paired difference is smaller than 10 points, or its 95 % interval includes 0.

## What was implemented (opt-in; defaults unchanged)

- `safety.correction_penalty` (default 0). During **training only**, the reward becomes `r − α‖u − u_exec‖²`. Here `u` is the policy's raw Gaussian sample and `u_exec` is the executed action after the action bounds **and** the filter. This matches the paper, whose certified action includes the input constraints. Note that the penalty therefore also acts on `v < 0` samples, the region behind Type II self-stops.
- Evaluation environments never apply the penalty (`make_env(..., training=False)`).
- Checks:
  - 3 new tests (penalty value, training-only, default off; negative values rejected); `tests/test_safety_filter.py`: 16 passed;
  - with the default config, a 4,096-step training run gives **bit-identical weights** to `main` (checked on 2026-10-25).

## Procedure

Two stages. Run E1a first, and E1b only if E1a is inconclusive or positive.

### E1a: screening at a reduced budget (cheapest valid test)

Same code, same seeds, same 300k-step budget for both arms. The control is needed because the committed checkpoints were trained for 1M steps.

Run E1a only from a branch that contains the penalty code (`exp/e1-pilot` or `exp/e1-correction-penalty`). On any other branch `safety.correction_penalty` is silently ignored. On `exp/e1-pilot`, check that `correction_penalty_mean > 0` in the penalty runs' `progress.csv`.

**Revised after the pilot (2026-10-26): one process at a time.** Use the pilot's driver: chunked, below-normal priority, with memory and throughput stops. Repeating the loop resumes interrupted runs and skips finished ones.

```bash
for s in 0 1 2 3 4; do
  python research/experiments/e1_pilot/train_pilot.py --arm control --seed $s --timesteps 300000 \
      --runs-root runs/e1a --deadline-s 1500
  python research/experiments/e1_pilot/train_pilot.py --arm penalty --seed $s --timesteps 300000 \
      --runs-root runs/e1a --deadline-s 1500
done
python research/experiments/e1_evaluate.py --arm penalty=runs/e1a/penalty/filtered \
    --arm control=runs/e1a/control/filtered --arm committed=results/checkpoints/filtered \
    --out runs/e1a_eval --jobs 1
```

The original commands are kept for reference. They run processes in parallel and need more free memory than this laptop usually has:

```bash
# penalty arm (alpha = 1), 5 seeds, 3 runs in parallel (keeps 9 of 12 logical cores free)
python -m safenav train-many --out runs/e1_penalty --variants filtered --seeds 0 1 2 3 4 \
    --timesteps 300000 --jobs 3 --set safety.correction_penalty=1.0
# control arm: identical, without the penalty
python -m safenav train-many --out runs/e1_control --variants filtered --seeds 0 1 2 3 4 \
    --timesteps 300000 --jobs 3
# evaluation (no training): committed test scenarios + 500 fresh expert scenarios, mean and sampled actions
python research/experiments/e1_evaluate.py --arm penalty=runs/e1_penalty/filtered \
    --arm control=runs/e1_control/filtered --arm committed=results/checkpoints/filtered \
    --out runs/e1a_eval --jobs 4
```

- Runs can be interrupted and resumed: `train-many ... --resume`.
- Seeds, scenarios and episodes:
  - training: 5 seeds per arm, 10 runs;
  - evaluation per agent: 300 committed test scenarios (secondary; already inspected) plus 500 fresh expert scenarios, seed base 5,000,000 (**primary**; never used before), each with mean and sampled actions.

### E1b: confirmation at the full budget (only after E1a)

As E1a with `--timesteps 1000000` and **only the penalty arm**. The control is then the committed filtered checkpoints (same code path, seeds 0–4, 1M steps).

## Cost (measured where possible; laptop on mains power)

**Measured in the pilot (2026-10-26, one process).** 150k steps took 203 s of wall time (185 s CPU) with the machine idle, and peak memory was 324 MB. When another program used all cores, speed fell to about 30 steps/s; when it also used most of the RAM, the process was paged out.

**E1a estimate (10 runs × 300k steps, one at a time).**
- Training: about 7–8 min per run when idle, so **about 1.3–1.5 h** in total.
- Evaluation: 15 agents × (500 fresh + 300 committed scenarios) × 2 action modes, about 30–60 min in one process.
- Under load it takes much longer; the driver then stops and resumes.

The original estimates follow:

| Part | Runs | Measured basis | Estimated CPU | Estimated wall time |
|---|---:|---|---|---|
| E1a training | 10 × 300k steps | 20,480 penalty steps took 65 s in one process (≈ 330 steps/s, 2026-10-25) | ≈ 3–5 core-hours | ≈ 1.5–2.5 h with 3 parallel runs |
| E1a evaluation | 15 agents × 1,600 episodes | stall taxonomy: 6,600 episodes in 5 min on 8 processes | ≈ 1 core-hour | ≈ 15–30 min with 4 processes |
| E1b training | 5 × 1M steps | as above; the README's 2.6–3.7 h per run was with 10 runs competing for 10 cores | ≈ 5–9 core-hours | ≈ 2–4 h with 3 parallel runs |

- **Laptop load.** 3 single-threaded runs use about 25 % of the 12 logical cores. Run on mains power; the machine stays usable.
- **Cloud.** Not needed. A free Colab or Kaggle CPU session could run the same commands, but free sessions disconnect; use `--resume`.

## Decision rules (fixed before running)

| Result on fresh expert scenarios, mean actions | Interpretation | Effect on the paper |
|---|---|---|
| Deadlock rate falls ≥ 10 points (paired 95 % interval excludes 0, Holm-adjusted), success rises, 0 collisions | **The published penalty largely resolves deadlock here.** | The method contribution (heading-aware filter) loses its motivation. The paper becomes an analysis paper whose result is "deadlocks exist, and here is which published remedy removes which type", with E1 as a central result. |
| Type II falls but Type I does not (or vice versa) | the penalty removes one mechanism only | supports the two-mechanism taxonomy; motivates a targeted remedy for the remaining type (E2) |
| No significant change | the penalty does not resolve deadlock in navigation | strengthens the case for a structural remedy; report as a negative result for the penalty |
| Collisions > 0 with the filter on | implementation error (the filter guarantees 0) | stop and debug |

Secondary results, reported but not decisive: sampled actions, the committed scenarios, Type I and Type II separately, and time to goal.
