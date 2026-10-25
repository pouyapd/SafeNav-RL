# Novelty assessment and candidate research directions

Branch `research/q1-feasibility-audit` · 2026-10-25 · based on [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md) and [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md)

Labels: **verified** (checked against data or a source), **hypothesis** (plausible, untested). Novelty statements of the form "no prior work does X" are always hypotheses here, because the literature search was targeted, not systematic.

---

## 1. Verdict on the repository as it stands

**The current repository does not contain a novel, publishable methodological contribution for a Q1 journal.** It is a careful, reproducible benchmark of known components. Item by item:

| Element of SafeNav-RL | Status in the literature | Novel? |
|---|---|---|
| 2D unicycle simulator, sparse range sensor, PPO | standard (e.g. Tai et al. 2017; Long et al. 2018) | no |
| One-step discrete-time CBF speed filter with closed-form QP and invariance proof | standard DT-CBF (Agrawal & Sreenath 2017; Ames et al. 2017/2019) | no |
| Filter as part of the environment; buffer stores the policy's own action | studied as "filtering training actions" (Pizarro Bejarano et al. 2025); taxonomy in Krasowski et al. 2023; theory in Oh et al. 2026 | no |
| Finding: the filter removes collisions and costs success (deadlock) | known qualitatively: undesired CBF-QP equilibria (Reis et al. 2021; Tan & Dimarogonas 2024); chattering against the filter (Pizarro Bejarano et al. 2025) | no, as a headline |
| Finding: training with the filter does not clearly help | inconclusive (intervals include 0, audit §6.2), and it **differs** from Pizarro Bejarano et al. (2025), who found that training with the filter *plus a correction penalty* helps. The repository has no penalty, so the two results are not in conflict. | not a contribution until explained |
| Geometric feasibility oracle that separates avoidable from unavoidable timeouts | a sound evaluation device; I found no CBF+RL navigation paper that reports one (hypothesis H-L2) | an evaluation contribution at most |
| **Two distinct stall mechanisms: filter-blocked boundary chattering (agent trained without filter) vs. learned self-stop before the boundary (agent trained with filter)** (audit §5.1) | I found no analysis of filter-induced or filter-learned equilibria of *learned* policies (hypothesis H-L1) | **possibly, as the starting point for a method; not as a result on its own** |

A Q1 journal paper therefore needs a **new method with a precise claim**, tested against the published methods listed below, on a frozen held-out test set, with uncertainty over both seeds and scenarios.

## 2. Closest existing methods and precise differences

| Method | What it does | Difference from what this project could add |
|---|---|---|
| Pizarro Bejarano, Brunke, Schoellig 2025 (RA-L) | PPO with the filter in training, correction penalty `α‖u−u_safe‖²`, safe resets; quadrotor trajectory tracking | No obstacles, navigation or deadlock analysis. Their penalty is a **required baseline** for anything proposed here. If their recipe already removes the deadlocks in this setting, the training-side novelty disappears (gate G1 below). |
| Krasowski et al. 2023 (TMLR) | Compares action replacement, projection and masking; a penalty on interventions helps | Benchmarks on pendulum and quadrotor stabilisation; no navigation liveness. "Action masking" for continuous actions overlaps with the parameterisation idea in Direction 2. |
| Oh, Nguyen, Hu, Fisac 2026 (IASEAI) | With a sufficiently permissive filter, filtered-MDP training loses no asymptotic optimality | An asymptotic statement about optimal policies. The SafeNav filter allows turning in place, so it may be permissive in their sense (to be checked against their definition). The observed gap would then be a **learning failure, not a limit of the filter**: a clean, testable framing for Directions 1–2. |
| Yang, Werner, de Sa, Ames 2025 (**preprint**, CBF-RL) | CBF filtering of rollouts plus a CBF reward term; deploy without the filter; navigation and humanoid | Aims to remove the runtime filter. Deadlock is not reported in the abstract. A required baseline if it is published or has code. |
| Reis, Aguiar, Tabuada 2021; Tan & Dimarogonas 2024 | Characterise undesired equilibria created by CBF-QPs for model-based nominal controllers | Analysis only; the nominal controller is a known stabilising law, not a learned policy, and they give no training-time remedy. |
| Jankovic et al. 2021/2024; Grover et al. 2021/2022; Chandra et al. 2025 | Deadlock analysis and resolution for (multi-robot) CBF controllers; liveness by making deadlock equilibria unstable; deadlock-free DT-CBF navigation | Model-based and mostly multi-agent. A single robot behind a filter with a *learned* nominal policy, trained with that filter in the loop, is a different closed loop. Whether their tools carry over is a research question, not an obvious step. |
| Marzari et al. 2025 (RA-L) | Builds a CBF layer around a trained navigation policy from verified unsafe regions | Targets violations, not liveness. Zero violations is already achieved here by construction. |
| Guo et al. 2025 (Sensors) | PPO + CBF-QP filter in Gazebo | Single runs, no intervals, timeouts not reported (verified, full text). Not a strong competitor, but it shows the evidence level reviewers will contrast with. |
| Daş et al. 2026 (RA-L); Cosner et al. 2021, 2023 | Robust / measurement-robust / stochastic CBFs; online reduction of robust-CBF conservatism in navigation | The closest competitors for Direction 3. |

## 3. Three candidate research directions

Each direction is a research question with a falsifiable prediction. Directions 1 and 2 can be combined into one paper (see §4).

---

### Direction 1: A liveness-preserving discrete-time safety filter for learned navigation policies (filter-level contribution)

1. **Research question.** For a learned policy deployed behind a one-step DT-CBF filter: can a filter that also acts on the heading (a QP over `(v, ω)`) provably avoid creating stable boundary equilibria while keeping the per-step invariance guarantee? And does training with such a filter in the loop remove filter-blocked deadlocks better than the correction penalty of Pizarro Bejarano et al. (2025)?

2. **Gap.** The deadlock and liveness results (Reis; Tan & Dimarogonas; Jankovic; Grover; Chandra) assume model-based nominal controllers. The filter-in-training results (Pizarro Bejarano; Krasowski; CBF-RL; Oh et al.) do not analyse equilibria. **Hypothesis H-L1:** no peer-reviewed work analyses or removes the equilibria created by a filter around a learned navigation policy. Verified fact supporting relevance: in this repository, 141 of 175 expert timeouts of the agent trained without the filter are filter-blocked boundary stalls with chattering heading (audit §5.1).

3. **Proposed technical contribution.**
   (a) A characterisation of the boundary equilibria of the speed-only one-step DT-CBF filter: at `h_i = 0`, forward motion is possible only for headings with `a_i(θ) ≤ 0` (audit §2.2).
   (b) A modified filter whose constraint depends on the heading after the turn, so that `ω` enters the QP. Two candidate forms: a two-step condition, or a look-ahead-point barrier `h̃_i(p, θ) = h_i(p + ℓ·u(θ))`. The filter adds a circulation term that makes boundary equilibria unstable, keeps `h(p_{t+1}) ≥ (1−γ dt) h(p_t)`, and has a proof.
   (c) Training in the loop with this filter, and an empirical test of whether it closes the gap to oracle-feasible success.

4. **Mathematical formulation.**
   - Closed loop: `x_{t+1} = f(x_t, F(x_t, π_φ(o_t)))`, with `x = (p, θ)`, `f` the Euler unicycle and `F` the filter.
   - Equilibrium (practical definition): a state with `‖p_{t+K} − p_t‖ ≤ ε` for all `t ≥ T`, goal not reached.
     - **Type I** (filter-blocked): `h_i(p*) ≈ 0`, `π_v > v_lim`, `a_i(θ_t) > 0` along the heading sequence.
     - **Type II** (policy self-stop, Direction 2): `π_v(o*) ≤ 0`, `h(p*) > 0`.
   - Proposed filter: `min_{v, ω} (v − v_n)² + λ(ω − ω_n)²` subject to
     - `h̃_i(p_t + v dt u(θ_t), θ_t + ω dt) ≥ (1 − γ dt) h̃_i(p_t, θ_t)` (linearised and conservatively bounded so that the convex-norm argument still holds);
     - `h_i(p_{t+1}) ≥ (1 − γ dt) h_i(p_t)` (the original guarantee, unchanged);
     - box constraints;
     - plus a circulation term `ω ← ω + κ σ(h̃_i) s_i(x)`, where `s_i` is the tangential direction chosen consistently (e.g. by goal side).
   - Claims to prove:
     - (i) forward invariance of `{h ≥ 0}` under the same assumptions as v0.2;
     - (ii) for a single obstacle, every boundary equilibrium of the filtered closed loop is unstable for any nominal policy whose commanded speed is positive there;
     - (iii) for several obstacles, identify the remaining equilibria (concave traps) and state them as a limitation.
   - The policy is arbitrary, so no global liveness guarantee is claimed.

5. **Closest competing methods.** Jankovic et al. 2024 (liveness from instability, multi-agent, model-based); Chandra et al. 2025 (deadlock-free DT-CBF); high-order CBFs (Xiao & Belta 2022); Pizarro Bejarano et al. 2025 (penalty); CBF-RL (preprint).

6. **Scenarios.**
   - In-distribution (ID): the current generator **fixed** (goal distances as documented, start anywhere, no trivial scenarios); a frozen test set of ≥ 1,000 scenarios per stage.
   - Out-of-distribution (OOD), with no training on these: denser fields (15–25 obstacles); "gates" (narrow gaps near `2(r + d_safe)`); concave traps built from obstacle clusters; and BARN-style layouts converted to circle sets.

7. **Baselines and ablations.**
   - Baselines:
     - PPO without filter, with filter at test time;
     - PPO with filter (v0.2);
     - PPO with filter + correction penalty (Pizarro Bejarano et al.);
     - PPO-Lagrangian (CMDP) with filter at test time;
     - CBF-RL (if code or a faithful re-implementation is possible);
     - potential field + filter;
     - DWA;
     - MPC with DT-CBF and the true map (an upper reference).
   - Ablations: heading term on/off; circulation term on/off; look-ahead `ℓ`; `γ ∈ {0.25, 0.5, 1, 2}`; train-time vs. test-time-only use of the new filter.

8. **Metrics.**
   - Primary: success; Type I deadlock rate on oracle-feasible scenarios; collisions (must stay 0).
   - Secondary: Type II rate; time to goal; SPL against the margin-feasible oracle path; minimum clearance; intervention frequency and magnitude `‖u_n − u_safe‖`; heading chattering (sign changes per second).

9. **Statistics.**
   - At least 10 training seeds per method.
   - Paired over scenarios, with a hierarchical bootstrap (seeds within method × scenarios) and a logistic mixed model (`success ~ method + (1|seed) + (1|scenario)`).
   - Primary contrasts pre-registered; Holm correction over them.
   - Seed-level and scenario-level uncertainty reported separately (see `EXPERIMENT_PLAN.md`).

10. **Computational cost (estimate).**
    - One 1M-step run took 2.6–3.7 h on one core (README; the ten runs went in parallel on a 10-core laptop).
    - About 8 trained configurations × 10 seeds × ~3.5 core-h ≈ **280 core-hours**, plus about 40 core-h for γ/ℓ sweeps and a few core-hours of evaluation. That is about 1.5–2 days of wall time on this 12-core laptop.
    - The QP over `(v, ω)` replaces a closed form, which adds an unmeasured per-step cost (a 2-variable QP is cheap).

11. **Risks and reasons it may fail.**
    - (a) The correction penalty alone may already remove Type I stalls. Then the filter adds little (gate G1).
    - (b) Reviewers may judge the filter to be Jankovic-style liveness applied to one robot, unless the learned-policy analysis and the training interaction are substantive.
    - (c) Type II self-stops are untouched by any filter change. If they dominate after training with the new filter, the headline improvement may be small; Direction 2 is then needed in the same paper.
    - (d) Concave traps need global reasoning; a local filter cannot solve them, so the claim must exclude them.
    - (e) The proof of (ii) may need assumptions (smoothness, single active constraint) that the closed-form discrete-time setting violates.

12. **Evidence needed for a competitive submission.**
    - A theorem with a full proof, and unit tests for (i).
    - Statistically clear reduction of Type I deadlocks and higher success than the penalty baseline, on frozen ID and OOD test sets.
    - Zero collisions.
    - Results in a standard robot simulator (Gazebo or Isaac, with a TurtleBot3-class model, ROS 2) with a LiDAR-built obstacle set, or an honest limitation statement.
    - A short hardware demonstration is strongly advisable for RA-L or T-ASE and close to mandatory for T-RO.
    - Public code and data with a reproduction manifest.

---

### Direction 2: Learned equilibria under filtered training: why filtered PPO learns to stop, and how to prevent it (training-level contribution)

1. **Research question.** Why does PPO trained with the filter learn to stop *before* obstacles (Type II), and which change to the training removes these stops without losing safety: correction penalty, bounded action distribution, filter-state observation, or reward change?

2. **Gap.** Pizarro Bejarano et al. and Krasowski et al. show that penalties help on stabilisation and tracking tasks. Oh et al. show that permissive filters lose nothing asymptotically. None of these explains a *learned* stopping equilibrium in navigation (hypothesis). Verified motivation: 122 of 195 deterministic expert timeouts of the filter-trained agents are self-stops at a median of 0.41 m before the boundary, in all 5 seeds; sampling removes most of them (audit §5.1).

3. **Proposed technical contribution.**
   - An analysis of **action aliasing** in the filtered MDP: many nominal actions map to the same executed action (environment clipping of `v ≤ 0`, filter clipping of `v ≥ v_lim`), so their advantages are equal.
   - A proof that, for a clipped Gaussian policy, the gradient component along the aliased direction has zero mean but non-zero variance. The mean therefore drifts like a random walk in aliased regions. This is a hypothesis to be proven or refuted.
   - Two remedies, tested as alternatives:
     - (i) a parameterisation in the executed-action space, `v = v_lim(x) · sigmoid(z)`: a continuous form of action masking; overlaps with Krasowski et al.'s taxonomy, which must be acknowledged;
     - (ii) a bounded distribution with an explicit "stop" penalty that depends on oracle-free progress.

4. **Mathematical formulation.**
   - Filtered MDP `M_F = (X, U, P_F, r)` with `P_F(·|x, u) = P(·|x, F(x, u))`.
   - Aliasing classes `[u]_x = {u' : F(x, u') = F(x, u)}`; `Q_F(x, u) = Q_F(x, u')` for `u' ∈ [u]_x`.
   - For `π_φ = N(μ_φ(o), σ²)`, consider `E[∇_μ log π · A]` restricted to `[u]_x`. Show that it is 0 in the interior of the aliased set, and quantify the variance of the stochastic gradient that drives `μ`.
   - Prediction: the measured drift of `μ_v` in states near obstacles correlates with the time the agent spends in aliased regions. This is testable from training logs if `μ_v` is logged.

5. **Closest competing methods.** Pizarro Bejarano et al. 2025; Krasowski et al. 2023 (masking, penalties); squashed-Gaussian and Beta policies (standard in SAC and continuous-control RL); Oh et al. 2026.

6. **Scenarios.** The same as Direction 1; plus "goal behind obstacle" probes (goal, obstacle and start collinear), where Type II stops concentrate (median cos 0.79 between goal and obstacle directions, audit §5.1).

7. **Baselines and ablations.** v0.2 filtered PPO; + correction penalty (α ∈ {0.1, 1, 10}); + tanh-squashed or Beta policy; + speed-limit and barrier values in the observation; + executed-action parameterisation; risk penalty on/off; sampled vs. mean deployment.

8. **Metrics.** Type II rate; distribution of `μ_v` near obstacles over training; success; time to goal; and the same safety metrics as Direction 1.

9. **Statistics.** As in Direction 1. In addition, training-curve uncertainty: interquartile mean and stratified bootstrap over seeds (Agarwal et al. 2021).

10. **Computational cost.** About 7 configurations × 10 seeds × ~3.5 core-h ≈ **250 core-hours**; the logging of `μ_v` is negligible.

11. **Risks.**
    - (a) A bounded distribution may fix the stops trivially. Then the contribution is a diagnosis with a simple fix: useful, but probably not enough on its own for a Q1 journal.
    - (b) The stops may come from the risk penalty (non-potential-based shaping out to 0.8 m) rather than aliasing; this needs the risk-off ablation.
    - (c) The aliasing argument may be known in the RL literature on clipped actions; a broader search is needed.

12. **Evidence needed.** A causal test: an intervention that removes aliasing removes the stops, and one that keeps aliasing but changes the reward does not. Mechanism plots with intervals, and results across seeds and OOD sets. **On its own this is more likely a workshop or conference paper.** Its natural home is as the analysis half of a Direction 1 paper.

---

### Direction 3: Calibrated robust DT-CBF filtering from LiDAR and estimated state: the safety–liveness frontier (uncertainty contribution)

1. **Research question.** When the filter uses an estimated pose and obstacles extracted from a noisy scan, how large must the margin be for a stated per-step violation probability? How much liveness (success, deadlock rate) does that margin cost for learned policies trained with and without the filter?

2. **Gap.** Robust, measurement-robust and stochastic CBFs exist (Cosner et al. 2021, 2023; Kolathaya & Ames 2019), as do conformal methods for dynamic obstacles (Lindemann et al. 2023) and adaptive robust CBFs for navigation (Daş et al. 2026). **Hypothesis H-L3:** the safety–liveness frontier under calibrated margins has not been quantified for learned policies with sparse range sensing. This gap is **narrow** and the competition is strong.

3. **Proposed technical contribution.**
   - A conformally calibrated error bound `q_α` on barrier estimation error (pose and obstacle-fit error), from held-out calibration rollouts.
   - A robust DT-CBF `ĥ − q_α` with a worst-case closing rate over the heading uncertainty.
   - A finite-sample per-step guarantee under exchangeability, and an episode-level bound (union bound, or the martingale argument of Cosner et al. 2023).
   - A measured frontier: violation rate vs. deadlock rate as α and γ vary.

4. **Mathematical formulation.**
   - Scores `s_j = max_i |h_i(x_j) − ĥ_i(ŷ_j)|` on n calibration states; `q_α` = the ⌈(n+1)(1−α)⌉-th smallest score.
   - Then `P(h_i(x) ≥ ĥ_i(ŷ) − q_α ∀i) ≥ 1 − α` (marginal, exchangeable).
   - Filter on `ĥ − q_α` with `a_i` maximised over `θ ∈ [θ̂ − δ_α, θ̂ + δ_α]`.
   - Per-step claim: `P(h(p_{t+1}) ≥ 0 | h(p_t) ≥ 0) ≥ 1 − α`.
   - Episode claim: `P(no violation in T steps) ≥ 1 − Tα` (loose), or a tighter martingale bound.
   - Caveat: closed-loop data break exchangeability; the guarantee needs care (e.g. calibration under the deployed policy, or adaptive conformal inference).

5. **Closest competing methods.** Daş et al. 2026; Cosner et al. 2021, 2023; Lindemann et al. 2023; Keyumarsi et al. 2024 (LiDAR-synthesised CBF); Long et al. 2021.

6. **Scenarios.** Noise levels: pose σ ∈ {0, 0.02, 0.05, 0.1} m; heading σ ∈ {0, 1°, 3°}; range σ ∈ {0.02, 0.05} m with dropouts. Obstacles estimated by circle fitting to scan clusters. The same ID/OOD sets as Direction 1.

7. **Baselines and ablations.** Nominal (non-robust) filter on estimates; a fixed worst-case margin; ISSf-style margin; Daş et al.-style adaptive margin (if re-implementable); and the oracle filter with true state as an upper bound. Ablations: calibration set size n; α; calibration policy vs. deployment policy.

8. **Metrics.** Empirical per-step and per-episode violation rate (against ground truth), success, deadlock rate, time to goal, and margin size.

9. **Statistics.** Violation rates near 0 need many episodes. Use exact (Clopper–Pearson) bounds; to show a 1 % rate with useful precision takes ≥ 3,000 episodes per cell. Calibration coverage is checked on fresh rollouts.

10. **Computational cost.** Simulation is cheap per episode, but the matrix is large: about 4 noise levels × 3 α × 4 methods × 10 seeds. Retraining is only needed for policies trained under noise (about 4 configurations × 10 seeds ≈ **140 core-hours**). Evaluation at ≥ 3,000 episodes per cell adds tens of core-hours.

11. **Risks.**
    - (a) Strong, very recent competition (Daş et al. 2026) may already cover the core idea.
    - (b) The guarantee is only marginal and per step; reviewers may reject episode-level claims.
    - (c) Without a realistic sensor simulator, the noise model is easy to criticise.
    - (d) The result may simply be "a bigger margin gives more deadlocks", which is expected and not novel unless the frontier reveals a non-obvious structure.

12. **Evidence needed.** Coverage verified empirically, and frontier curves with intervals. Comparison with at least one published robust-CBF method; a realistic LiDAR simulation (Gazebo); ideally hardware with motion capture as ground truth.

---

## 4. Recommendation

**Pursue Direction 1, with Direction 2 as its mechanism analysis, in one paper.** Working title: *"Deadlock in safety-filtered reinforcement learning for navigation: filter-induced and learned equilibria, and a liveness-preserving filter."*

Reasons:
- It starts from a **verified, specific phenomenon in this codebase**: two stall mechanisms with different causes. The literature does not appear to treat this (hypothesis, to be confirmed by a systematic search).
- There is measured headroom. At the 0.3 m margin, 100 % of expert test scenarios are feasible, but the best filtered PPO condition reaches 76.4 % (sampled actions) and the deterministic ones reach 61–65 %. The gap is 24–39 points.
- Directions 1+2 need about 500 core-hours, which fits this laptop in about a week of compute. Direction 3 needs more infrastructure (sensor simulation) and faces stronger competition.

**Go/no-go gates** (cheap experiments that can kill the plan early; details in `IMPLEMENTATION_ROADMAP.md`):

| Gate | Experiment | Continue only if |
|---|---|---|
| G0 | Fix the generator (A1–A2), build a frozen test set, re-train v0.2 variants with 10 seeds | the filtered success gap to oracle feasibility stays ≥ 15 points and both stall types persist (each ≥ 20 % of timeouts in at least one variant) |
| G1 | Add the correction penalty (Pizarro Bejarano et al.) and a bounded policy | some deadlock gap remains (≥ 10 points to the oracle-feasible rate) after the best of these; otherwise the remaining contribution is an empirical replication in navigation, and the target should drop to a conference or letter |
| G2 | Prototype the heading-aware filter, test-time only, on frozen agents | Type I stalls fall measurably (paired interval excludes 0) before investing in proofs and full training |
| G3 | Systematic literature search (protocol in `IMPLEMENTATION_ROADMAP.md`, task B0) | no prior paper already proposes a liveness-preserving filter for learned navigation policies |

**Venue realism (opinion, not verified).** With strong simulation results, a theorem and a small hardware demo, IEEE RA-L is a realistic target; it is generally listed in the first quartile, but check the current JCR/SJR year before claiming this. Robotics and Autonomous Systems or Autonomous Robots are alternatives. IEEE T-RO or IJRR would need a stronger theory, larger-scale experiments and hardware. **Nothing in this repository currently shows real-world safety, and finite simulation cannot prove it.**
