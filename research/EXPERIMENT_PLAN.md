# Experiment plan

Branch `research/q1-feasibility-audit` · 2026-10-25 · for the recommended Direction 1 + 2 ([`NOVELTY_ASSESSMENT.md`](NOVELTY_ASSESSMENT.md) §4), with the Direction 3 extensions marked.

**Principles.**
1. Every claim is about a stated scenario distribution in simulation. **No finite simulation study proves real-world safety.** The per-step guarantee is a theorem about the model, and the experiments only check the implementation and measure liveness.
2. The primary hypotheses, contrasts and analysis are fixed **before** any run on the frozen test sets. They are recorded in `research/PREREGISTRATION.md` (roadmap A7) and committed before the test sets are first used.
3. The existing results in `results/` are not changed. New results go to `results_v3/` (or a separate data release) with a hash manifest.

---

## 1. Hypotheses

| ID | Hypothesis | Type | Falsified if |
|---|---|---|---|
| H1 | On oracle-feasible frozen test scenarios, the liveness-preserving filter (LPF), trained in the loop, has a lower Type I deadlock rate than v0.2 filtered PPO **with correction penalty**. | primary | the paired 95 % interval for the difference includes 0 (Holm-adjusted) |
| H2 | LPF + in-loop training reaches a higher success rate than penalty-PPO, at zero collisions. | primary | as H1 |
| H3 | Type II self-stops are caused by action aliasing: a bounded or executed-action parameterisation removes them, while a reward-only change (risk penalty off) does not. | mechanism | the bounded or executed-action variant still has a Type II rate within the interval of v0.2 |
| H4 | The LPF keeps zero violations of the margin and the per-step bound in all exact-state conditions. | implementation check | any violation (it would be a bug, since it is a theorem) |
| H5 (Dir. 3) | With noisy state, a calibrated margin keeps the empirical per-step violation rate ≤ α, at a measurable cost in deadlocks. | extension | the violation rate exceeds the Clopper–Pearson upper bound for α |

## 2. Scenario sets (generator v3)

Generator v3 fixes audit issues 2 and 10:
- goal distance drawn by **rejection sampling** within the documented range, with no clipping;
- start uniform over the free workspace;
- the start must be outside the goal region;
- the v0.2 generator stays available under a flag, so existing results remain reproducible.

| Set | Seed base | Size per stage | Use | Rule |
|---|---|---|---|---|
| Training stream | per run seed | — | training | — |
| Validation | 1,000,000 (v3) | 100 (300 total) | checkpoint selection, hyperparameter tuning **for all methods** | may be inspected |
| Test-ID | 5,000,000 (new) | 1,000 | primary analysis | **frozen**: no inspection before pre-registration; never used for any design decision |
| Test-OOD-dense | 6,000,000 | 500 | 15–25 obstacles | frozen |
| Test-OOD-gates | 6,100,000 | 500 | gaps between `2(r + d_safe)` and `2(r + d_safe) + 0.3 m` on the oracle path | frozen |
| Test-OOD-traps | 6,200,000 | 500 | concave obstacle clusters (U shapes) between start and goal | frozen; local methods expected to fail, reported as a limitation set |
| Probe-collinear | 7,000,000 | 300 | goal directly behind an obstacle (Type II probe) | diagnostic only |
| Calibration (Dir. 3) | 8,000,000 | 500 rollouts | conformal calibration | disjoint from test sets |

For every test scenario, the margin-feasibility oracle (`feasibility.py`, 5 cm grid) is computed for each margin used. Scenarios that are infeasible at a margin are reported separately: success is computed **conditional on feasibility** for the primary analysis, and unconditionally as a secondary result.

## 3. Experiment matrix

### 3.1 Core matrix (Directions 1 + 2)

Factors and levels (static obstacles, exact state):

| Factor | Levels | Why |
|---|---|---|
| Method (trained) | M1 PPO, no filter in training · M2 PPO + v0.2 filter · M3 M2 + correction penalty α=1 · M4 M3 + bounded (tanh-Gaussian) policy · M5 M2 + executed-action parameterisation · M6 PPO + LPF in training · M7 M6 + correction penalty · M8 PPO-Lagrangian (no filter in training) | baselines from the literature (M3: Pizarro Bejarano et al. 2025; M8: Stooke et al. 2020), mechanism ablations (M4, M5), proposed method (M6, M7) |
| Test-time filter | none · v0.2 filter · LPF | 2×2-style ablation of train vs. test filter |
| Deployment actions | mean · sampled | sampling breaks Type II stops (audit §6.2) |
| Non-learned references | potential field + filter · DWA + filter · MPC with DT-CBF (true map) | equal tuning budget on the validation set; MPC as an upper reference |
| Training seeds | 10 per trained method (20 for M3 and M7, the primary contrast) | the seed term limits interval width at ≥ 1,000 scenarios (§5.3) |

Not every cell is run. Each trained method is evaluated with its own training filter, with no filter, and (M1, M2, M3) with the LPF at test time only. That is about 25 method × filter × action conditions per test set.

### 3.2 Secondary sweeps (run only for M3, M6, M7 and the potential field)

| Sweep | Levels | Seeds | Purpose |
|---|---|---|---|
| Safety margin `d_safe` | 0.1, 0.3, 0.5 m | 5 per level (retrained) | trade-off between safety and success; feasibility share shrinks as the margin grows (82 % of v0.2 expert scenarios are feasible at 0.5 m) |
| Barrier rate γ | 0.25, 0.5, 1.0, 2.0 s⁻¹ (γ·dt ≤ 1 holds) | test-time only, frozen agents | γ was never ablated (audit §2.2) |
| Obstacle density | train on stages as now; test on ID / OOD-dense | — | generalisation |
| Look-ahead ℓ (LPF) | 0.1, 0.2, 0.4 m | test-time only | LPF design choice |

### 3.3 Uncertainty and dynamic obstacles (extensions; only if H1/H2 hold)

| Experiment | Levels | Notes |
|---|---|---|
| State-estimation noise given to the filter | pose σ 0, 0.02, 0.05, 0.1 m; heading σ 0°, 1°, 3° | the filter receives *estimates*; violations are measured against ground truth (Dir. 3) |
| Obstacles from scan | circle fitting to range clusters, range σ 0.02 / 0.05 m, 5 % dropouts | removes the known-map assumption |
| Dynamic obstacles | 2–6 moving circles, constant velocity 0.2–0.6 m/s, known velocity in the filter (DT-CBF on predicted positions); then velocity noise | the guarantee extends only for known bounded motion; collision-cone CBFs (Thontepu 2023; Goswami 2024) as a filter baseline |
| Standard simulator | Gazebo or Isaac, TurtleBot3-class robot, ROS 2 (extends the untested `ros2_integration/`) | sim-to-sim transfer check; still simulation |

## 4. Metrics (definitions fixed in advance)

| Metric | Definition | Primary? |
|---|---|---|
| Success | goal within 0.5 m before the step limit, without collision | yes |
| Collision | robot body touches an obstacle, or leaves the workspace (reported separately) | yes (safety) |
| Timeout | neither of the above |  |
| **Type I deadlock** | timeout, net displacement < 0.10 m over the last 100 steps, filter reduced speed in ≥ 50 % of those steps | yes |
| **Type II deadlock** | as Type I, but commanded speed ≤ 0.001 in ≥ 50 % of those steps | yes (H3) |
| Slow progress | timeout that is not a stall | |
| Time to first stall | first step at which the stall criterion holds over a 50-step window | |
| SPL (margin) | success × `l*/max(l, l*)`, with `l*` the margin-feasible oracle path length | |
| Path efficiency | straight-line distance / path length (successes only) | |
| Clearance | minimum and 5th-percentile body-to-surface distance | |
| Margin violation depth | `max(0, −min_t h_t)`; must be 0 for exact-state conditions | yes (H4) |
| Intervention frequency / magnitude | share of steps where `u_safe ≠ u_nom`; mean `‖u_nom − u_safe‖` | |
| Chattering | sign changes of executed ω per second | |
| Compute | filter time per step (µs), training wall time | |

All metrics are stored per episode, together with the per-step nominal action, speed limit and barrier minimum for the first 50 episodes per condition (roadmap A8–A9).

## 5. Statistical analysis

### 5.1 Two different sources of uncertainty, reported separately

1. **Between training seeds** (*is the training method reliably better?*): the unit is a trained agent. On a fixed scenario set, compute each seed's success rate, then report a t-interval and a bootstrap over seeds, plus the interquartile mean and stratified bootstrap (Agarwal et al. 2021). This interval answers "would a new training run of this method do as well *on these scenarios*?"
2. **Between test scenarios** (*does the result hold on the scenario distribution?*): the unit is a scenario. For a fixed set of agents, bootstrap over scenarios. With deterministic baselines this is the only source.
3. **Combined** (*the claim a paper makes*): a **paired hierarchical bootstrap**, in which scenarios are resampled jointly for the methods compared and seeds are resampled within each method. It is cross-checked with a logistic mixed model, `success ~ method + (1 | seed:method) + (1 | scenario)`, reporting risk differences with 95 % intervals.

The audit's re-analysis of the v0.2 data shows why this matters. The scenario term accounted for 25–86 % of the sum of squares in expert success, against under 2 % for seeds (`audit_outputs/variance_components_expert.csv`).

### 5.2 Tests and multiplicity

- Primary contrasts (H1, H2): M7 vs. M3, and M6 vs. M3, on Test-ID expert, conditional on feasibility. Holm correction over these four tests (2 contrasts × 2 hypotheses).
- Secondary contrasts are reported with unadjusted intervals and labelled exploratory.
- Paired scenario-level tests (McNemar on per-scenario majority outcome) serve as a sensitivity check only.
- Report effect sizes (risk differences), not only p-values.

### 5.3 Sample size (estimate from the v0.2 data)

The width of the paired hierarchical-bootstrap interval for "filter-trained vs. unfiltered-trained, sampled actions, expert" was fitted as `var ≈ 0.0041/n_seeds + 0.089/n_scen + 0.103/(n_seeds·n_scen)` by subsampling the stored episodes. This is an approximate fit from a bootstrap over only 5 seeds, so it is rough:

| Seeds per method | Scenarios per stage | Approx. 95 % half-width |
|---:|---:|---:|
| 5 | 100 | ±8.6 points (current) |
| 10 | 300 | ±5.3 |
| 10 | 1,000 | ±4.4 |
| 20 | 1,000 | ±3.4 |
| 10 | 3,000 | ±4.1 |

Beyond about 1,000 scenarios, more seeds help more than more scenarios. Hence the plan: 1,000 scenarios per stage, 10 seeds per method, and 20 seeds for the two primary methods. The minimum detectable effect is then about 4–5 points, which is small compared with the 24–39-point gap to oracle feasibility.

### 5.4 Rare events

Zero observed collisions in N episodes gives a 95 % upper bound of about 3/N (rule of three). For example, 0 in 10,000 gives ≤ 0.03 %. This is reported as a bound on the *simulated* rate under the stated assumptions, not as evidence of real-world safety.

## 6. Checkpoint selection and tuning (fairness)

- The same selection rule for every trained method: best validation success over 300 validation episodes, evaluated with the method's own deployment filter. The final checkpoint is also reported.
- An equal tuning budget for every method, including the baselines (e.g. 20 configurations by random search on the validation set). Record the search space and the result.
- Hyperparameters of PPO, shared by all PPO variants, are tuned once for M3 (the strongest baseline), not for the proposed method.

## 7. Compute budget (estimates, 12-core laptop, CPU)

| Block | Runs | Core-hours (≈ 3.5 h per 1M-step run) | Wall time at 10 parallel runs |
|---|---:|---:|---:|
| Gate G0 (M1, M2 on generator v3, 10 seeds) | 20 | 70 | ≈ 7–8 h |
| Core matrix (M3–M8; 10 seeds, 20 for M3/M7) | 80 | 280 | ≈ 28–30 h |
| Margin sweep (3 levels × 3 methods × 5 seeds) | 45 | 160 | ≈ 16 h |
| Evaluation (all sets, ≈ 25 conditions × 10–20 agents × 4,800 episodes) | — | ≈ 40–60 | ≈ 5 h |
| Extensions (noise, dynamic obstacles) | ≈ 60 | ≈ 210 | ≈ 21 h |
| **Total** | | **≈ 760–780** | **≈ 3–4 days of compute** |

A CPU was enough for v0.2 (small MLP). Performance measurements must be made on mains power; battery runs are labelled.

## 8. What this plan cannot show

- Real-world safety or performance: the dynamics are exact Euler kinematics, and the obstacles are circles.
- Safety under unmodelled dynamics (slip, delay, actuator limits), except through the Gazebo extension, and even that is only another simulator.
- Liveness in concave traps. A local filter cannot provide it; this is reported as a limitation set, not hidden.
