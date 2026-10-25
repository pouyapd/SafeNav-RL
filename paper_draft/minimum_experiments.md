# Minimum additional experiments

Nothing in this file has been run, except where it says **done**. Expensive runs wait for the author's approval.

## How costs were estimated (measured on this laptop, 2026-10-25, on mains power)

- **Training speed.** 20,480 PPO steps with the correction penalty took **65 s** in one process, about 330 steps/s including start-up. A 300k-step run therefore takes roughly 15–20 min alone, or about 25–30 min with 3 runs in parallel. The README's 2.6–3.7 h per 1M steps was measured with 10 runs competing for 10 cores.
- **Evaluation speed.** The stall taxonomy (6,600 episodes) took 5 min on 8 processes; the window-length check took 48 s.
- **Laptop load.** Use 3–4 parallel processes (≈ 25–35 % of the 12 logical cores), on mains power. All training commands support `--resume`, so an interrupted run loses at most 50k steps.
- **Free alternatives.** Not needed for anything below. Free Colab or Kaggle CPU sessions can run the same commands, but they disconnect after a few hours, so use `--resume`. No GPU is needed: the networks are small MLPs.

## Priority order

1. **E1a**, the cheapest test of the published penalty (as requested).
2. **E5** (done).
3. **E3**.
4. **E2**.
5. **E1b** and **E4**, depending on E1a.
6. **E6**.
7. **E7** and **E8**: optional.

---

### E1: Does the published correction penalty alone remove the deadlocks? (highest priority)

| | |
|---|---|
| Research question | Does PPO with the filter in training plus the penalty `α‖u − u_exec‖²` (Pizarro Bejarano et al., RA-L 2025) remove Type I and/or Type II stalls? |
| Hypothesis | H1: with α = 1, the expert deadlock rate (mean actions, fresh scenarios) falls by ≥ 10 points against an identical control; success rises; 0 collisions. |
| Procedure | Code and protocol are ready on local branch `exp/e1-correction-penalty`: `research/experiments/E1_correction_penalty.md`, `e1_evaluate.py`. **E1a** (screening): penalty arm vs. identical control, 300k steps each. **E1b** (confirmation, only if needed): penalty arm at 1M steps against the committed filtered checkpoints. |
| Seeds / scenarios / episodes | E1a: 2 arms × 5 seeds = 10 runs. Evaluation per agent: 500 fresh expert scenarios (seed base 5,000,000; primary) + 300 committed test scenarios (secondary), each with mean and sampled actions. |
| CPU | E1a ≈ 3–5 core-h training + ≈ 1 core-h evaluation; E1b ≈ 5–9 core-h. |
| Wall-clock | E1a ≈ 2–3 h in total with 3 parallel runs; E1b ≈ 2–4 h. |
| On the laptop? | Yes, with `--jobs 3`, on mains power, overnight if preferred. |
| Success / failure | Decision table in `E1_correction_penalty.md`. Success = deadlock reduction ≥ 10 points with the paired 95 % interval excluding 0 (Holm-adjusted over the two primary tests). |
| Effect on the paper | If the penalty removes both stall types, the paper's main message becomes "the published penalty fixes the deadlocks; here is why (both aliasing regions are penalised)". The heading-escape result (E2) then becomes secondary. If it removes only one type, the taxonomy is the key insight. If it removes neither, it is a strong negative result that motivates E2. |
| Safeguards | Implementation is opt-in (default 0). Weights are bit-identical to `main` with the default config (checked). 93 tests pass on the branch. |

### E2: Does a heading-level escape remove filter-blocked stalls without retraining?

| | |
|---|---|
| Research question | Can a turn-rate modification ψ applied at test time (safe by Prop. 1) remove Type I stalls of the frozen agents? |
| Hypothesis | H2: ψ reduces the Type I rate of the agent trained without the filter by ≥ 10 points (expert, fresh scenarios) and leaves Type II unchanged (Prop. 3 predicts this). |
| Procedure | Implement `ψ(x, u) = clip(ω_nom + κ·σ(x))` whenever the filter limits the speed (`v̄ < v_nom`), where `σ(x)` is the turn direction toward the tangent of the active obstacle on the goal side. One parameter, κ, chosen on **validation** seeds (base 1,000,000) only, from {0.5, 1.0}; then frozen. Add as an opt-in wrapper (default off) with a unit test of invariance (adversarial policy, as in `tests/test_safety_filter.py`). Evaluate the 10 committed agents, mean and sampled actions, with ψ off and on. |
| Seeds / scenarios / episodes | 10 agents × 2 action modes × 2 (ψ on/off) × (500 fresh expert + 300 committed) ≈ 32,000 episodes; κ selection ≈ 2 × 10 × 100 validation episodes. |
| CPU | ≈ 0.5–1 core-h evaluation; ≈ 0.5 day of implementation and tests. |
| Wall-clock | ≈ 15–30 min with 4 processes. |
| On the laptop? | Yes. |
| Free alternative | Not needed. |
| Success / failure | Success: paired 95 % interval for the Type I reduction excludes 0 and success does not fall. Failure: no reduction, or new stalls appear (e.g. oscillation between two obstacles), reported as such. |
| Effect on the paper | Success turns the analysis into an actionable result: filter-level fixes for Type I, policy-level fixes for Type II. Failure keeps the paper as an analysis and strengthens the case that Type I needs a multi-step or planning-level fix. Close prior art to discuss: Thumm et al. 2023; Jankovic et al. 2024. |

### E3: Do the findings hold on fresh scenarios from a corrected generator?

| | |
|---|---|
| Research question | Are the taxonomy and the paired contrasts artefacts of the 300 inspected test scenarios or of the goal-clipping generator? |
| Hypothesis | H3: on fresh scenarios with documented goal distances, the sign and approximate size of the four expert stall contrasts are unchanged. |
| Procedure | Add an opt-in generator flag (`env.generator: v3`): goal distance by rejection sampling in the stage range, start anywhere in the free workspace, no start inside the goal. The default stays v2, so existing results reproduce. Run `stall_taxonomy.py` (parameterised by seed base and generator) on 1,000 fresh scenarios per stage for intermediate and expert. Evaluation only, no retraining. The agents were trained on the v2 distribution, so this is also a mild distribution-shift test; state it as such. |
| Seeds / scenarios / episodes | 10 agents × 2 modes × 2,000 scenarios + 2 baselines × 2,000 = 44,000 episodes. |
| CPU | ≈ 1–2 core-h; ≈ 2–3 h of implementation and tests. |
| Wall-clock | ≈ 30–40 min with 4 processes. |
| On the laptop? | Yes. |
| Success / failure | Success: all four expert contrasts keep their sign with intervals excluding 0. Failure: any contrast changes sign. The paper must then report the dependence on the scenario distribution. |
| Effect on the paper | Removes the test-set-reuse threat (G3); replaces the inspected test set as primary evidence. |

### E4: What causes the self-stops? (mechanism; run after E1a)

| | |
|---|---|
| Research question | Are Type II stops caused by action aliasing (a flat return over clipped speeds) or by the reward (risk penalty)? |
| Hypothesis | H4: a tanh-squashed Gaussian policy (no aliasing at v ≤ 0) trained with the filter shows a Type II rate below the control. Setting the risk penalty to 0 does not. |
| Procedure | Two arms × 5 seeds × 300k steps, compared against the E1a control. Arm 1: tanh-squashed policy (requires an opt-in network option with its log-probability correction; ≈ 0.5 day of code + tests). Arm 2: `reward.risk_penalty_beta=0` (config only). |
| CPU / wall | ≈ 3–5 core-h; ≈ 2–3 h with 3 parallel runs. |
| On the laptop? | Yes. |
| Success / failure | Success: arm 1 lowers Type II (paired interval excludes 0) and arm 2 does not. Mixed results are reported as mixed. |
| Effect on the paper | Turns the Discussion's hypothesis into evidence, or removes it. |

### E5: Window-length sensitivity of the taxonomy (**done**)

48 s of evaluation (`research/scripts/taxonomy_window_sensitivity.py` → `research/audit_outputs/taxonomy_window_sensitivity_expert.csv`).
- Agent trained without the filter: counts unchanged for K = 50–200 (141 Type I, 34 Type II).
- Agent trained with the filter: Type I 63–69, Type II 119–122.

The manuscript has been updated.

### E6: Classical and safe-RL baselines

| | |
|---|---|
| Research question | How do the PPO conditions compare with a classical local planner and with a CMDP method under the same filter and scenarios? |
| Procedure | (a) DWA with the filter: implementation ≈ 1 day; evaluation minutes. (b) MPC with DT-CBF and the true map as an upper reference: needs an optimisation library (e.g. CasADi). Ask before installing; ≈ 1–2 days. (c) PPO-Lagrangian: ≈ 1 day of code; 5 seeds × 300k–1M steps ≈ 3–9 core-h. All tuned on validation seeds with an equal budget. |
| CPU / wall | (a) under 0.5 core-h; (b) ≈ 1–2 core-h (MPC is slower per step); (c) ≈ 3–9 core-h. |
| On the laptop? | Yes. |
| Effect on the paper | Required by most robotics reviewers. Also tests whether Type I stalls appear with any controller that keeps pushing toward the goal. |

### E7: More training seeds (optional)

Five more seeds per variant at 1M steps: 10 runs ≈ 10–18 core-h, ≈ 5–8 h with 3 parallel runs. It narrows the intervals: at 1,000 scenarios, the approximate half-width falls from ≈ 4.4 to ≈ 3.4 points when going from 10 to 20 seeds. Run only if a reviewer asks or if a primary contrast is borderline.

### E8: Standard simulator or hardware (optional, heavy)

Gazebo or Isaac with a TurtleBot3-class model and the ROS 2 node: several days of engineering, plus a large install (ask first). Hardware needs lab access. Advisable for RA-L, close to mandatory for T-RO, optional for RAS or Autonomous Robots.

## Total for the minimum set (E1a, E2, E3, E4)

| | |
|---|---|
| CPU | ≈ 9–14 core-hours |
| Wall-clock | ≈ 6–9 h, spread over several sessions with 3–4 processes |
| Engineering | ≈ 2–3 days (penalty: done; ψ wrapper; generator flag; squashed policy) |
| Cloud | not needed |

This replaces the earlier 780 core-hour roadmap for the purpose of a first submission. That roadmap remains the plan for a full method paper (Direction B), which is not recommended now.
