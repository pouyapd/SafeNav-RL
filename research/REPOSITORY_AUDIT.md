# Repository audit: SafeNav-RL v0.2.0

Branch `research/q1-feasibility-audit` · audited commit `d257eca` (= `origin/main` on 2026-10-25) · audit date 2026-10-25

**Purpose.** To find out whether the code, results and documentation are a sound base for a journal paper, and what must change first. This audit changes nothing in `safenav/`, `results/`, `docs/` or `README.md`. The only new code is a read-only analysis script, [`research/scripts/audit_diagnostics.py`](scripts/audit_diagnostics.py). It writes only to [`research/audit_outputs/`](audit_outputs/).

**How to read the labels.**
- **Verified**: I re-ran or recomputed it, and the output is stated.
- **Code reading**: it follows from the source code as quoted, but was not run.
- **Hypothesis**: a plausible explanation that has not been tested.

---

## 0. Summary

| | |
|---|---|
| **Engineering quality** | High for a research prototype. Seeded, deterministic evaluation; exact resumption; config validation; 90 tests; committed checkpoints that reproduce the per-episode benchmark exactly. |
| **Mathematical claim** | The one-step discrete-time CBF invariance proof is **correct** under its stated assumptions, and the between-sample bound is correct for the simulator. It is a standard result (Agrawal & Sreenath 2017), not a contribution. |
| **Stored results** | All headline numbers in the README match the stored files. **Two documentation statements are contradicted by the data**: the goal-distance ranges (§3.1) and the failure mechanism of the filter-trained agent (§5.1). |
| **Main methodological weaknesses** | (1) The intervals reflect training-seed variation only. In a seed × scenario decomposition, the scenario term accounts for 25–86 % of the sum of squares and the seed term for under 2 % (§6.1). (2) The safety margin and the sampled-action evaluation were chosen after looking at the test scenarios (§6.3). (3) The scenario generator does not produce the documented goal distances; 3 of 300 test episodes are solved at step 1 (§3.1). (4) Weak baselines: no published safe-navigation or safe-RL baseline (§7). |
| **New finding with research value** | Timeouts with the filter come from **two different mechanisms**, depending on how the agent was trained (§5.1). An agent trained *without* the filter and run with it pushes into the keep-out boundary at full commanded speed while its turn rate chatters. An agent trained *with* the filter mostly **stops by itself** (commanded `v ≤ 0`), at a median of 0.35 m before the boundary. The second mechanism is a learned equilibrium of the policy, not of the filter. |
| **Publishability as is** | Not publishable in a Q1 journal as a methods paper. The benchmark and the deadlock finding are a sound starting point; see [`NOVELTY_ASSESSMENT.md`](NOVELTY_ASSESSMENT.md). |

---

## 1. What was run (reproducibility checks)

Environment: Windows 11, Python 3.12.7, torch 2.10.0+cpu, gymnasium 1.2.3, numpy 2.4.1. These are the same versions recorded in `results/benchmark/metadata.json`; the benchmark was produced on this machine. A check on a different machine or OS is still missing.

| Check | Command | Result |
|---|---|---|
| Test suite | `python -m pytest -q` | **Verified**: 90 passed in 62 s |
| Benchmark from committed checkpoints | `python -m safenav benchmark --runs results/checkpoints --out <scratch> --jobs 8` (2 min 25 s) | **Verified**: `episodes.csv` and `scenarios.csv` byte-identical to `results/benchmark/`. `summary.csv` differs only in floating-point rounding (largest absolute difference 1.8e-15, from summation order in parallel workers). The README claims identity only for the per-episode files, which holds. |
| Sampled-action evaluation | `python scripts/stochastic_eval.py --runs <empty> --out <scratch>` (2 min 40 s) | **Verified**: `episodes.csv` and `summary.json` byte-identical |
| Training (10 × 1M steps) | not re-run (about 30 core-hours) | **Not verified.** Determinism and exact resumption are covered by tests at small scale only. The training logs are internally consistent (§4.3). |
| Audit diagnostics | `python research/scripts/audit_diagnostics.py` (2.5 min) | outputs in `research/audit_outputs/` (§3, §5, §6) |

---

## 2. Safety filter (`safenav/safety/cbf_filter.py`, `docs/SAFETY.md`)

### 2.1 The proof (verified by reading, and consistent with the tests)

For `h_i(p) = ‖p − c_i‖ − ρ_i`, convexity of the norm gives `‖p_{t+1} − c‖ ≥ ‖p_t − c‖ − dt·v·a_i` with `a_i = u(θ_t)·(c − p_t)/‖c − p_t‖`. With `a_i v ≤ γ h_i` this gives `h_i(p_{t+1}) ≥ (1 − γ dt) h_i(p_t)`, and wall barriers are linear. The derivation, the closed-form solution of the 1-D QP, and the fallback for `h < 0` are all correct. The adversarial test (40 expert scenarios, full speed toward the nearest obstacle) and the chord bound between samples (≈ 1.4 mm for `ρ ≥ 0.9 m`) are implemented as described.

### 2.2 Properties of the filter that the documentation does not state

These are not bugs. They are consequences of the design that matter for any paper.

1. **Turn rate has no effect on safety within a step (code reading).** The Euler step moves the position along the *current* heading `θ_t` and applies `ω_t` afterwards. So the one-step safe set in action space is `{v ≤ v_lim(x)} × [−ω_max, ω_max]`, and projecting onto it can only change `v`. The docstring's "closest safe action" is therefore exact for the one-step set. However, the filter can never *steer* the robot away from an obstacle; only the policy can. In continuous-time terms, the position barrier has relative degree 2 with respect to `ω` (cf. high-order CBFs, Xiao & Belta 2022).
2. **At the boundary, the admissible headings collapse to a half-plane (derivation).** `v_lim = γ h / a` tends to 0 as `h → 0` for every heading with `a > 0`, that is, any heading within 90° of the obstacle centre. A robot near the boundary can only move at a useful speed if it heads at or beyond the tangent (`a ≤ 0`). Sliding along an obstacle therefore requires the policy to hold a slightly outward heading at every step. This explains the chattering stall in §5.1.
3. **`γ = 0.5 s⁻¹` is conservative (`γ·dt = 0.05`).** Head-on, the allowed speed is `0.5·h`. For example, `h = 0.4 m` allows 0.2 m/s, and `h` can shrink by at most 5 % per step. Neither γ nor `d_safe` was varied in any experiment except the oracle/baseline margin sweep. The effect of γ on time-to-goal and on deadlock is unmeasured.
4. **No reverse motion** (`v ∈ [0, 1]`). The only escape from a boundary is to rotate in place.
5. **Information asymmetry (code reading).** The filter uses the true robot state and the exact obstacle list. The policy sees a noisy 16-beam scan plus a proximity "risk" feature that is computed from the true position and the full obstacle map (§3.3). The filter's guarantee does not carry over to a filter built from LiDAR data (the ROS 2 scaffold) or from estimated state. `docs/SAFETY.md` states this as assumption 2; a paper must keep it explicit.
6. **The bound between samples depends on the generator.** The 1.4 mm figure uses the smallest generated obstacle radius (0.3 m) and `v·dt ≤ 0.1 m`. It is not a property of the filter in general.

---

## 3. Environment, observation model and scenario generator

### 3.1 The goal-distance ranges in the documentation are wrong (verified, documentation bug)

`NavigationEnv.reset()` draws the start in the lower-left square `[1.5, 5]²`, places the goal at a distance drawn from the stage range, and then **clips the goal to `[1.5, 8.5]²`** (`navigation_env.py` lines 419–428). Clipping shortens many goal vectors. [`audit_outputs/goal_distances.csv`](audit_outputs/goal_distances.csv):

| Stage (documented goal range) | Test set: share below documented minimum | Test set: start already inside goal region | Mean / max distance (test) | 3,000 fresh seeds: share below minimum |
|---|---:|---:|---:|---:|
| beginner (3–5 m) | 42 % | 2 / 100 | 3.0 / 5.0 m | 36 % |
| intermediate (5–8 m) | 66 % | 0 / 100 | 4.1 / 7.0 m | 63 % |
| expert (5–12 m) | 53 % | 1 / 100 | 4.5 / 8.9 m | 58 % |

- The README's curriculum table ("goal 5–12 m" for expert) describes the requested range, not the generated one. **No expert test goal is farther than 8.9 m.**
- **3 of the 300 test scenarios start inside the goal region.** Every controller "succeeds" at step 1 in these (verified in `episodes.csv`: 15/15 PPO episodes and 3/3 baseline episodes, 1 step each). This inflates every success rate by 0–2 percentage points per stage, the same for all methods. It does not change any comparison, but it should be removed.
- The start is always in the lower-left quadrant. Because the policy observes absolute `(x, y)`, it can learn layout priors that would not transfer to other start distributions (hypothesis; untested).

### 3.2 The test layouts are fairly open (verified)

The median ratio of oracle shortest path (margin 0) to straight-line distance is 1.06 (beginner), 1.07 (intermediate) and 1.08 (expert); see `scenarios.csv`. At the 0.3 m margin, 100 % of the test scenarios are feasible. The "expert" stage is therefore moderate clutter, not a hard maze. Claims about dense or cluttered environments are not supported by this test set.

### 3.3 Observation model

- **The privileged "risk" feature (code reading).** `risk_score()` uses the true position and *all* obstacles, including those that no beam hits. It encodes the exact distance to the nearest obstacle surface, saturated at 0.8 m. This is mild privileged information. The README describes it as a "proximity risk", without saying it is computed from ground truth.
- **Range-sensor coverage (verified, [`audit_outputs/sensor_coverage.csv`](audit_outputs/sensor_coverage.csv)).** 200 expert scenarios × 20 random poses. Every obstacle within 1.0 m of the robot body is hit by at least one of the 16 beams. Obstacles at 1.0–1.5 m are missed by all beams 3 % of the time; at 1.5–2.0 m, 13.5 %; at 2.0–2.7 m, 43 %. So the policy perceives near obstacles reliably and far ones poorly.
- **The "sensor noise" does not affect safety.** Position noise (σ = 0.02 m) and range noise (σ = 0.02 m) enter only the policy's observation. The heading is noise-free, and the filter uses ground truth. No experiment varies the noise level.

### 3.4 Obstacle generator

The minimum surface-to-surface gap between obstacles is 1.0 m. With `d_safe = 0.3`, a robot needs 2·(0.3 + 0.3) = 1.2 m, so the margin closes some gaps. The oracle still finds a path in 100 % of the test scenarios. Obstacles that cannot be placed after 200 attempts are skipped: the expert stage requests 8–15 obstacles and gets 8–12 in the test set (7–13 in 3,000 fresh seeds in this audit, while the README reports 7–14 for its own 3,000 seeds; the seed sets differ, so this is not a contradiction).

---

## 4. PPO training and checkpoint selection

### 4.1 Correct (verified by reading and by tests)

- The buffer stores the policy's own sample and its log-probability, with the filter treated as part of the environment. This is the same choice as the "filtering training actions" variant of Pizarro Bejarano et al. (2025) for on-policy methods.
- GAE uses each transition's own `done`. Time-limit truncation bootstraps from `V(s_T)` by adding `γ V` to the last reward and then cutting the recursion, which is correct.
- Advantage normalisation, clipping, value loss, entropy bonus and linear LR decay are standard. No `target_kl` is used.

### 4.2 Weaknesses

1. **Unbounded Gaussian policy with clipping in the environment (code reading, standard but consequential).** The policy is a diagonal Gaussian. `v` is clipped to `[0, 1]` by the environment, and deterministic evaluation uses the mean. Any mean `≤ 0` gives `v = 0`: the robot stops. All actions with `v ≤ 0` lead to the same outcome, so their advantages cannot separate them. *Hypothesis H-A1:* this flat region allows the speed mean to drift below 0 in states near obstacles, which would produce the "self-stop" stalls of §5.1. Bounded distributions (tanh-squashed Gaussian, Beta) are the usual remedy; testing this is an experiment, not a fix.
2. **Validation is too small to choose checkpoints reliably (verified from `results/training_logs/`).** The validation set has 60 episodes (20 per stage), so its standard error is about ±4–5 percentage points. All five unfiltered runs reach 100 % validation success (first at 0.30–0.50M steps), so their best checkpoint is decided by the tie-break on mean return. Choosing the best of 20 noisy validation scores also biases the selected validation score upward (winner's curse). For the filtered runs, the selected scores are 0.87–0.92, while expert test success is 0.61. Part of this gap is the stage mix (validation pools all stages), and part is selection.
3. **Filtered runs pick early checkpoints.** The best validation scores were reached at 0.10M, 0.20M, 0.30M, 0.70M and 1.0M steps. The README reports this correctly. Together with intervention rates of 36–61 % of training steps at the end of training (`progress.csv`, last update), it suggests that filtered training is not converging to a better policy within 1M steps. Hypothesis; learning-curve uncertainty was not analysed.
4. **One hyperparameter set for both variants**, tuned for neither. The README states this.
5. **Selection protocol for "unfiltered agent + filter at test time".** This checkpoint was selected with the filter *off*. That is a reasonable deployment scenario, but it should be stated; a filter-on selection would be a separate condition.

### 4.3 Training logs agree with the README (verified)

Every one of the 10 runs reached the expert stage by 43k–76k steps, so about 93 % of training ran on the hardest stage. Final validation success: unfiltered 0.97–1.0 ("about 99 %"); filtered 0.77–0.88 ("near 80 %"). Three of the five filtered seeds chose checkpoints within the first 0.31M steps (seeds 0, 2 and 3).

---

## 5. Stored results: claims against data

### 5.1 The failure mechanism is described wrongly for the filter-trained agent (verified, documentation bug)

README: *"Replaying the failed expert episodes of the filter-trained agent shows the mechanism: in most of them the robot reaches a keep-out boundary early and stops there … The deterministic policy keeps steering into the boundary."*

The diagnostics re-ran every expert test episode of the 10 committed agents with the filter on and recorded the commanded speed, the speed limit and the turn rate. A timeout counts as a **stall** if the robot moved less than 0.10 m in its last 100 steps. A stall is **filter-blocked** if the filter cut the speed in at least half of those steps, and **self-stop** if the policy itself commanded `v ≤ 0.001` in at least half of them. Results are in [`audit_outputs/timeout_attribution_expert.csv`](audit_outputs/timeout_attribution_expert.csv) (500 episodes per row):

| Agent · actions | Timeouts | Filter-blocked stalls | Self-stop stalls | Still moving |
|---|---:|---:|---:|---:|
| trained **with** filter · mean | 195 | 67 | **122** | 6 |
| trained with filter · sampled | 118 | 71 | 13 | 34 |
| trained **without** filter · mean | 175 | **141** | 34 | 0 |
| trained without filter · sampled | 150 | 147 | 1 | 2 |

Medians over the last 100 steps:

| | trained with filter (mean actions) | trained without filter (mean actions) |
|---|---|---|
| commanded speed | 0.00 m/s | 1.00 m/s |
| barrier `h` at the end | 0.135 m (self-stop class only: 0.35 m) | 0.000 m |
| turn-rate sign changes per 100 steps | 54 | 65 |
| cos(angle between goal direction and nearest-obstacle direction) | 0.79 | 0.61 |

Interpretation:

- **Agent trained without the filter.** The description "keeps steering into the boundary" fits *this* agent. It commands full speed, sits exactly on the boundary (`h = 0`), and its turn rate chatters. This is the behaviour §2.2(2) predicts, and it matches the undesired boundary equilibria of CBF-QPs (Reis et al. 2021).
- **Agent trained with the filter.** It mostly does something different. In 122 of its 195 deterministic timeouts, the policy itself commands zero speed, typically 0.35 m *before* the keep-out boundary, facing the obstacle with the goal behind it. This self-stop occurs in every seed (10–32 episodes each). The filter does not cause it; it is a **learned undesired equilibrium of the policy**. Sampling actions removes most of these stalls (122 → 13). That is why sampling helps the filter-trained agent much more than the unfiltered one (+15.4 vs +5.0 points, §6.2).

The README sentence should be corrected; this needs the author's approval and is roadmap task A3. The finding is also the strongest research lead in the repository; see the novelty assessment.

### 5.2 All headline numbers match the files (verified)

| Claim | Source | Status |
|---|---|---|
| 0 collisions in 3,600 filtered episodes; minimum obstacle clearance 0.300 m | `episodes.csv` (min 0.3000000012) | ✓ |
| Unfiltered PPO, expert: 94.6 % (filter off) → 65.0 % (filter on) | `summary.csv` | ✓ |
| Filter-trained 61.0 %; sampled 76.4 % [68.2, 84.6] vs 70.0 % [66.3, 73.7] | `summary.csv`, `stochastic_eval/summary.json` | ✓ |
| Potential field with filter 72.0 % [62.5, 79.9]; go-to-goal 46 % / 57 %, 43 % collisions | `summary.csv` | ✓ |
| "Training with the filter did not clearly help" | paired re-analysis in §6.2 | ✓ (the conclusion survives a stronger test) |
| Expert goal range "5–12 m" | generator | ✗ (§3.1) |
| Failure mechanism of the filter-trained agent | re-run | ✗ (§5.1) |
| v0.1 safety layer: 0.095 m entry, 2/40 out of workspace | `scripts/v01_safety_layer_check.py` | not re-run in this audit |

### 5.3 Leftover from v0.1

`assets/trajectory_*.png` came from a checkpoint that no longer exists, produced by the old code. The README says so; these images should not appear in a paper.

---

## 6. Evaluation protocol and statistics

### 6.1 The intervals describe training-seed variation only, which is the smaller source (verified)

The README gives t-intervals over 5 seeds for PPO, and says so. A two-way decomposition of expert success, seed × scenario ([`audit_outputs/variance_components_expert.csv`](audit_outputs/variance_components_expert.csv)), shows how small that source is:

| Condition | Mean success | Share of sum of squares: seed | scenario | residual |
|---|---:|---:|---:|---:|
| unfiltered-trained · filter off | 0.946 | 0.008 | 0.248 | 0.743 |
| unfiltered-trained · filter on | 0.650 | 0.003 | 0.863 | 0.134 |
| filter-trained · filter on | 0.610 | 0.007 | 0.714 | 0.279 |
| filter-trained · filter on · sampled | 0.764 | 0.019 | 0.587 | 0.393 |
| unfiltered-trained · filter on · sampled | 0.700 | 0.003 | 0.836 | 0.160 |

The scenario term is 30–300 times larger than the seed term in every condition. (This is a descriptive sum-of-squares split, not a fitted variance-component model; roadmap A6 replaces it with a mixed model.) A seed-only interval such as 65.0 % [61.5, 68.5] is valid for *these 100 scenarios*. It understates the uncertainty about success on the scenario *distribution*. The cheapest way to tighten claims is therefore more test scenarios (evaluation is cheap), not more seeds. More seeds are still needed for claims about training methods (§6.2 note).

### 6.2 Paired comparisons are not used, although every method sees the same scenarios (verified)

[`audit_outputs/paired_comparisons_expert.csv`](audit_outputs/paired_comparisons_expert.csv) gives paired hierarchical bootstrap intervals for the expert stage: scenarios are resampled jointly for both conditions, and seeds are resampled within each condition; 10,000 replicates:

| Comparison (difference in success rate) | Difference | 95 % interval |
|---|---:|---|
| cost of adding the filter at test time (unfiltered agent), off − on | +0.296 | [0.216, 0.384] |
| filter-trained − unfiltered-trained, filter on, mean actions | −0.040 | [−0.108, 0.032] |
| filter-trained − unfiltered-trained, filter on, sampled actions | +0.064 | [−0.016, 0.152] |
| sampled − mean, filter-trained agent | +0.154 | [0.078, 0.240] |
| sampled − mean, unfiltered-trained agent | +0.050 | [0.004, 0.104] |
| filter-trained (sampled) − potential field, filter on | +0.044 | [−0.040, 0.130] |

The README's qualitative conclusions survive. Training with the filter is not shown to help (both intervals include 0). Sampling clearly helps, and much more for the filter-trained agent. *Caveat:* a bootstrap over only 5 seeds underestimates seed variance; at least 10 seeds are needed for method-level claims (see [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)).

### 6.3 Design decisions were made on the test scenarios (verified from the documentation, design-level leakage)

- `d_safe = 0.3` was chosen with the geometric oracle on the **100 expert test scenarios** (`docs/SAFETY.md`, "Choice of the margin").
- The sampled-action evaluation was added after replaying **failed test episodes**.
- The potential-field gains were "tuned briefly by hand"; on which scenarios is not recorded.

There is no leakage in the machine-learning sense: training, validation (`1,000,000+`) and test (`2,000,000+`) scenarios come from disjoint seeds, and the training stream is a separate generator. But the reported test numbers are no longer a clean held-out estimate for the decisions above. A fresh, frozen test set (new seed base, never inspected) is needed for any paper.

### 6.4 Other points

- Different interval types for PPO (t over seeds) and baselines (Wilson over episodes) are not comparable. The README states this.
- SPL uses the oracle path at margin 0 as reference, but filtered agents must keep the 0.3 m margin. An SPL relative to the margin-feasible shortest path (also stored in `scenarios.csv`) would be the fair reference for filtered conditions.
- The timeout class mixes stalls, slow progress and wandering. §5.1 shows these must be separated.
- No test checks the generated goal-distance distribution or excludes trivial scenarios.

---

## 7. Baselines and fairness

| Baseline | Status | Issue |
|---|---|---|
| Go-to-goal | present | obstacle-blind; a floor, not a competitor |
| Potential field | present, hand-tuned, `margin = 0` | tuned on unknown scenarios; known local-minimum failures (Koren & Borenstein 1991) |
| PPO trained without filter + filter at test | present | — |
| PPO trained with filter (action replacement in Krasowski et al.'s terms) | present | — |
| Correction penalty during training (Pizarro Bejarano et al. 2025; Krasowski et al. 2023) | **missing** | the closest published training recipe |
| CMDP / Lagrangian PPO (Achiam et al. 2017; Stooke et al. 2020) | **missing** | standard safe-RL baseline |
| Classical local planner (DWA, Fox et al. 1997) or MPC with DT-CBF (Zeng et al. 2021) | **missing** | needed to show RL is worth it at all |
| CBF-RL training (Yang et al. 2025, preprint) | **missing** | recent direct competitor |

Fairness note: PPO checkpoints are selected on validation scenarios, but the baselines get no comparable tuning budget. A paper needs equal tuning budgets on validation scenarios for all methods, documented.

---

## 8. Tests

The 90 tests cover the claims listed in the README, and I found no false positive. Not covered: the scenario distribution (goal distances, trivial scenarios); the policy's action range at evaluation (for example, how often the deterministic speed is ≤ 0); the information asymmetry between filter and policy; and filter behaviour with noisy state (not modelled).

---

## 9. Issue register

Severity is for a journal submission; none of these issues invalidates the stored per-episode results.

| # | Issue | Type | Severity | Affects current claims? | Roadmap |
|---|---|---|---|---|---|
| 1 | Failure mechanism of the filter-trained agent described wrongly | documentation vs. data | **major** | yes (one README paragraph, and the failure figure caption) | A3 |
| 2 | Generated goal distances do not match the documented ranges; 3 trivial test scenarios | generator / documentation | **major** | numbers shift by ≤ 2 points, uniformly; documentation wrong | A1, A2 |
| 3 | Intervals ignore variance between scenarios; no paired tests | statistics | **major** | conclusions survive (§6.2), widths understated | A6 |
| 4 | Margin and sampled-action evaluation chosen on test scenarios | protocol | **major** | test estimates not cleanly held out | A7 |
| 5 | No published safe-RL or classical baselines | experimental design | **major** | novelty and significance claims | B3 |
| 6 | Validation too small / saturated; best-of-many selection | protocol | moderate | checkpoint choice partly a tie-break | A5 |
| 7 | Unbounded Gaussian + clipping; deterministic mean can be "stop" | algorithm (standard practice) | moderate (research-relevant, H-A1) | explains self-stops (hypothesis) | B1 |
| 8 | Privileged risk feature from ground truth | observation model | moderate | must be disclosed | A11 |
| 9 | Filter uses ground truth, policy uses sensors; noise never varied | scope | moderate | safety claims limited to the simulator (already stated) | C2 |
| 10 | Open layouts (path/straight ≈ 1.07); start always lower-left | scenario design | moderate | "expert" difficulty overstated | A1, B4 |
| 11 | γ and filter design never ablated | experimental design | moderate | — | B2 |
| 12 | `summary.csv` not byte-reproducible (1.8e-15) | numerical | minor | no | A9 (sort before aggregating) |
| 13 | v0.1 demo images in `assets/` | presentation | minor | already disclosed | — |
