# Lightweight experiments (evaluation only): protocol

Branch `research/lightweight-experiments`. Written and committed **before** running L2–L5. Results are added in a later commit, below the line marked *Results*.

> **Later status (2026-11-01).** This note was added afterwards; the protocol below is unchanged. The E1 pilot proposed here has since been run: [`../e1_pilot/README.md`](../e1_pilot/README.md) (one seed, inconclusive). The correction-penalty code is now also on branch `exp/e1-pilot`. E1a has still not been run.

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

---

## Results (added after running; the protocol above is unchanged since commit 9ced888)

All numbers come from `results/summary.json` (`python research/experiments/lightweight/analyze.py`). Rates are per 100 episodes; intervals are 95 % paired hierarchical bootstrap intervals (10,000 replicates, 5 seeds, so approximate).

### Runs actually executed

| Tag | Episodes | Wall time (s) | CPU time (s) | Peak memory | Power / notes |
|---|---:|---:|---:|---:|---|
| smoke | 15 | 1.7 | 1.5 | 206 MB | mains; 15/15 identical to the benchmark |
| L1_repro_committed_expert | 1,000 | 54.6 | 50.3 | 207 MB | mains; 1,000/1,000 outcomes and step counts identical |
| L2_val_k0 / k0.5 / k1.0 | 3 × 1,000 | 61 / 105 / 93 | 56 / 91 / 82 | 207 MB | mains |
| L3_fresh_mean_k0 | 5,000 | 290 | 269 | 206 MB | mains |
| L3_fresh_mean_k0.5 | 5,000 | 600.8 + 301.2 + 146.9 | (n.r.) + 275 + 144 | 207 MB | first invocation stopped at its 600 s deadline when the previous session ended; resumed on **battery** with 27–44 % background CPU load |
| L5_fresh_final_k0 | 2,500 (filtered only) | 301.5 + 302.5 + 199.3 | 203 + 82 + 67 | 206 MB | battery, then mains; heavy background load. The low CPU time shows the below-normal priority yielding to other work |

Every invocation ran one process with one torch thread at below-normal priority. No training was run.

### Deviations from the protocol (decided during the session, before analysis)

- **L4 (sampled actions × escape): not run.** The laptop was on battery (32 %) with 20–60 % background CPU load, and L4 was secondary and descriptive. Deferred.
- **L5 narrowed to the filter-trained agents.** The run slowed to about 1.5 episodes/s under background load. C1 concerns only the filter-trained agents, so the unfiltered final checkpoints (2,500 episodes) were deferred. The runner evaluates the filter-trained agents first, so no unfiltered episode had been run; the CSV contains 5 × 500 filter-trained episodes only.

### L2: κ selection (validation seeds 1,200,000–1,200,099)

Success over all 10 agents:

| κ | Success | Collisions |
|---|---:|---:|
| 0 | 64.7 % | 0 |
| 0.5 | 68.5 % | 0 |
| 1.0 | 65.6 % | 0 |

By the rule, **κ* = 0.5**.

### L3: fresh expert scenarios 9,000,000–9,000,499, policy mean, 10 committed agents

| Agents | κ | Success | Timeouts | Filter-blocked (Type I) | Self-stop (Type II) | Moving |
|---|---|---:|---:|---:|---:|---:|
| trained without filter | 0 | 68.3 % | 792 | 709 | 81 | 2 |
| trained without filter | 0.5 | 75.6 % | 611 | 523 | 85 | 3 |
| trained with filter | 0 | 63.9 % | 903 | 297 | 577 | 29 |
| trained with filter | 0.5 | 64.0 % | 901 | 244 | 615 | 42 |

There were 0 collisions in all 10,000 episodes, and the minimum obstacle clearance was 0.3000 m.

| Hypothesis | Estimate (points) | Decision (pre-registered rule) |
|---|---|---|
| H1: κ* lowers Type I, agents trained without filter | −7.4 [−10.2, −4.8] | **partly supported**: the interval excludes 0, but the effect is below the 10-point threshold |
| H2: κ* leaves Type II of filter-trained agents unchanged | +1.5 [−0.8, +3.9] | **supported** |
| H3: safety with κ* | 0 collisions, minimum clearance 0.3000 m | **supported** |
| H4: κ* raises success, agents trained without filter | +7.2 [+5.0, +9.7] | **supported** |
| R1: replication of the taxonomy and of both training contrasts at κ = 0 | Type I −16.5 [−22.8, −10.1]; Type II +19.8 [+12.8, +26.1]; Type I is 709/792 and Type II 577/903 of timeouts | **supported** |

Exploratory, not pre-registered:
- success of the filter-trained agents with κ*: +0.1 [−2.2, +2.4];
- their Type I: −2.1 [−4.7, +0.2];
- success, filter-trained − unfiltered-trained at κ = 0: −4.4 [−8.5, −0.2]. On the committed test set this was −4.0 [−10.8, 3.2].

The Type I reduction holds in every seed (131→91, 133→102, 149→108, 162→110, 134→112).

### L5: final (1M-step) vs. selected checkpoints, filter-trained agents, fresh scenarios, κ = 0

| Checkpoint | Success | Timeouts | Type I | Type II | Moving |
|---|---:|---:|---:|---:|---:|
| selected (`best.pt`, committed) | 63.9 % | 903 | 297 | 577 | 29 |
| final (`final.pt`, local only) | 59.1 % | 1,023 | 123 | 870 | 30 |

Final − best:
- Type II **+11.7 [+3.0, +20.8]**;
- Type I −7.0 [−13.0, −1.5];
- success −4.8 [−8.8, −1.2].

**C1 supported**: self-stops remain the largest class, and they become more frequent with more training.

Per seed, self-stops (best → final) were 139→205 (seed 0), 137→137 (seed 1, whose best checkpoint *is* its final one), 51→174 (seed 2), 105→206 (seed 3) and 145→148 (seed 4).

**Caveat:** the `final.pt` files are local training outputs, not in the repository. Their SHA-256 hashes are in `results/L5_fresh_final_k0/run_record.json`. Committing them (10 × 0.58 MB) would make L5 reproducible by others, and needs the author's approval.

### What these results do and do not justify

**Justified:**
- the two-mechanism taxonomy and both training contrasts replicate on 5,000 fresh episodes from the same generator;
- a simple heading escape, safe by construction, lowers filter-blocked stalls and raises success for agents trained without the filter, by about 7 points;
- it does not touch self-stops. This is expected, because it is triggered only by speed reductions, which never occur while the policy commands zero speed. A heading change triggered during a stop was not tested;
- self-stops are not an artefact of early checkpoint selection.

**Not justified:**
- that the escape "solves" Type I: 523 of 611 timeouts are still filter-blocked;
- anything about a corrected generator, other stages, sampled actions, the correction penalty, or real robots;
- a causal explanation of self-stops (more training increases them, but why is untested).
