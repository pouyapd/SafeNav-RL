# Lightweight experiments (evaluation only): protocol

Branch `research/lightweight-experiments`. Written and committed **before** running L2–L5. Results are added in a later commit, below the line marked *Results*.

## Resource limits applied to every run

- One process at a time, one torch thread, below-normal CPU priority (`psutil`), and a hard internal deadline. Results are written to CSV one episode at a time, so an interrupted run keeps its rows and resumes.
- No training, and no new dependencies.
- Measured before planning (smoke test and L1):
  - **≈ 0.055 s per episode**: 1,000 expert episodes in 54.6 s wall, 50.3 s CPU;
  - **peak memory ≈ 207 MB**;
  - the machine had 2.4 GB of RAM free and was on mains power.

## Candidate experiments: decision

| ID | Question | Needs training? | Cost (measured rate) | Decision |
|---|---|---|---|---|
| L1 | Does the evaluation harness (with κ = 0) reproduce the committed benchmark exactly? | no | 1,000 episodes, 55 s | **done (before this protocol)**: 1,000 / 1,000 outcomes and step counts identical; stall classes identical to the paper (67 / 122 / 6; 141 / 34) |
| L2 | E2, selection of κ on **validation** scenarios | no | 3 × 1,000 episodes, ≈ 3 min | run |
| L3 | E2 main test + E3-lite: fresh expert scenarios, κ = 0 vs. κ*, policy mean | no | 2 × 5,000 episodes, ≈ 10 min in 2 invocations | run |
| L4 | as L3 with **sampled** actions (do sampling and escape act on different mechanisms?) | no | 2 × 5,000 episodes, ≈ 10 min | run |
| L5 | checkpoint selection: do the **final** 1M-step checkpoints (local `runs/benchmark/*/seed_*/checkpoints/final.pt`, not committed) show the same mechanisms as the selected `best.pt`? | no | 5,000 episodes, ≈ 5 min | run |
| E1a (full) | correction penalty, 2 arms × 5 seeds × 300k steps | **yes**: 10 runs, 3.0M steps, ≈ 2.5–3 h in one process | **not run**: needs approval (see below) |
| E1-pilot | 1 seed × 2 arms × 150k steps | yes: ≈ 15–20 min in one process | **proposed, not run**: needs approval |
| E3 (full) | fresh scenarios from a *corrected* generator | no, but needs an opt-in generator change | ≈ 2–3 h of code; evaluation minutes | deferred. L3 uses fresh seeds of the *unchanged* generator, so the goal-clipping caveat remains |
| E4 | cause of self-stops (squashed policy, risk penalty off) | yes, 10 runs | ≈ 3–5 core-h | skipped: training |
| E6 | DWA / MPC / PPO-Lagrangian baselines | implementation ± training | days | skipped: out of scope for a lightweight session |

## Pre-registered design for L2–L5

**Intervention (E2).** `safenav/safety/heading_escape.py`. When the CBF filter reduces the commanded speed, the turn rate is shifted by κ·ω_max in the direction that rotates the heading away from the binding barrier, toward its tangent. The speed filter is unchanged. Safety follows from Prop. 1 of the manuscript and is checked by `tests/test_heading_escape.py` (adversarial policy, κ ∈ {0.5, 1}).

**Scenario sets.**
- Validation (κ selection only): expert stage, seeds 1,200,000–1,200,099. This is the stage-2 validation range used for checkpoint selection during training; it was never used for any analysis.
- Fresh test: expert stage, seeds **9,000,000–9,000,499**. Never used before. Same (v2) generator, so goal clipping still applies.

**Agents.** The 10 committed checkpoints (5 trained with the filter, 5 without). L5 additionally uses the 10 local `final.pt` checkpoints; their SHA-256 hashes are recorded in `run_record.json`.

**κ selection rule (fixed now).** Candidates κ ∈ {0.5, 1.0}. Choose the κ with the higher mean success over all 10 agents on the validation set (policy mean), subject to 0 collisions. If the two are within 0.5 percentage points, choose the smaller κ. κ* is then frozen for L3–L4.

**Hypotheses and decision criteria (fresh test set, policy mean unless stated).**

| ID | Hypothesis | Supported if | Contradicted if |
|---|---|---|---|
| H1 (remedy, Type I) | κ* lowers the Type I rate of the agents trained **without** the filter | paired 95 % interval of the difference (κ* − 0) lies below 0, and its point estimate is ≤ −10 points | interval includes 0 |
| H2 (Prop. 3 prediction) | κ* does **not** remove Type II self-stops of the agents trained **with** the filter | the Type II difference is small (\|point estimate\| < 5 points) or its interval includes 0 | Type II falls by ≥ 10 points with the interval excluding 0. This would contradict the interpretation of Prop. 3, which concerns speed limits only; ψ changes ω, so a change here would need explanation |
| H3 (safety) | 0 collisions and minimum clearance ≥ 0.30 m (up to 1e-9) with κ* | all episodes | any collision: a bug, since it is a theorem |
| H4 (success) | κ* raises the success of the agents trained without the filter | interval of the success difference above 0 | interval below 0 |
| R1 (replication, κ = 0) | Type I is the majority of timeouts for agents trained without the filter, Type II the majority for agents trained with it; both training contrasts (Type I lower, Type II higher when trained with the filter) keep their sign with intervals excluding 0 | all four conditions hold | any contrast changes sign |
| R2 (sampling × escape, L4) | with sampled actions **and** κ*, both stall types are lower than with either alone | reported descriptively with intervals. No decision rule: secondary | — |
| C1 (checkpoint selection, L5) | the final checkpoints of the agents trained with the filter also have Type II as their largest timeout class | Type II is the largest class | another class is larger. Then the self-stop finding is specific to the selected checkpoints and the manuscript must say so |

**Statistics.**
- Paired hierarchical bootstrap, 10,000 replicates, seed 0.
- κ contrasts and best-vs-final contrasts compare the *same* agents, so seeds and scenarios are resampled jointly for both arms.
- Training contrasts (filtered vs. unfiltered agents): scenarios are resampled jointly, seeds independently.
- Rates are per 100 episodes. With 5 seeds the intervals are approximate.
- No other analyses will be presented as confirmatory. Anything else is labelled exploratory.

## E1a: resolution of the command ambiguity (no training run)

- **Why the command names only `--variants filtered`.** `train-many --variants` chooses whether the filter is used *during training* (`filtered` / `unfiltered`). Both E1a arms train *with* the filter. They differ only in `--set safety.correction_penalty=1.0` and in the output root (`runs/e1_penalty` vs. `runs/e1_control`). Omitting `--variants` would also train unfiltered agents (the default is both), doubling the cost for runs the design does not need.
- **Size.** 2 arms × 5 seeds = **10 runs × 300,000 steps = 3.0M environment steps**, plus validation episodes.
  - Measured speed: about 330 steps/s per process (20,480 steps in 65 s, 2026-10-25).
  - One process at a time: about 2.5–3 h of CPU.
  - The earlier plan's `--jobs 3` is **not** recommended with 2.4 GB of free RAM. Each training process is expected to need more memory than the 207 MB of an evaluation process (unmeasured).
- **A hazard found during this review.** `safety.correction_penalty` is implemented only on branch `exp/e1-correction-penalty`. On every other branch the config accepts the key silently and ignores it (verified), so the "penalty" arm would secretly be a second control. Run E1 only from that branch, and check `info["correction_penalty"] > 0` in a smoke run first.
- **What existing results answer.** The committed filtered checkpoints are a control only at the 1M budget, so they are not a same-budget control for a 300k run. Nothing already computed tests the penalty.
- **Smaller preliminary test (proposed, not run).** **E1-pilot**: seed 0, penalty vs. control, 150k steps each (the curriculum reaches the expert stage by about 50–65k steps), run one after the other in a single process. Expected ≈ 8–10 min per run, ≈ 15–20 min in total, then 2 min of evaluation on the fresh set.
  - What it could show: whether the penalty produces a *large* change in Type I/II counts for one seed, which would justify the full E1a.
  - What it could not show: any general effect. One seed, a shorter budget than the committed agents, and no interval estimate across seeds.
