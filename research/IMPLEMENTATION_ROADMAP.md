# Implementation roadmap

Branch `research/q1-feasibility-audit` · 2026-10-25 · implements [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for the recommendation in [`NOVELTY_ASSESSMENT.md`](NOVELTY_ASSESSMENT.md) §4.

**Ground rules for every task.**
- `results/` and the v0.2 behaviour stay reproducible: the old generator and filter remain available behind flags, and a hash manifest guards the committed results (A10).
- Each task is a separate branch and pull request with tests.
- No README change may suggest that the project is publication-ready. README edits are limited to factual corrections (A3) and need the author's approval.

Scales used below:
- **Complexity:** S = under 1 day, M = 1–3 days, L = 1–2 weeks, XL = more than 2 weeks (one person).
- **Compute:** core-hours on the 12-core CPU laptop.

---

## A. Essential correctness fixes (before any new experiment)

| ID | Task | Complexity | Compute | Depends on | Acceptance criteria |
|---|---|---|---|---|---|
| A1 | **Scenario generator v3**: rejection-sample the goal distance within the stage range (no clipping); start uniform over the free workspace; configurable via `env.generator: v2 \| v3` (default v2 until the switch) | M | — | — | new test: over 3,000 seeds per stage, 100 % of goal distances lie in the documented range; `generator: v2` reproduces `results/benchmark/episodes.csv` byte for byte |
| A2 | Exclude trivial scenarios (start inside the goal region, or straight-line distance < 1 m) in v3 | S | — | A1 | test: no generated scenario has start-to-goal distance < 1 m |
| A3 | **Correct two README statements** (needs approval): the expert goal range, and the failure mechanism of the filter-trained agent (audit §3.1, §5.1); also `docs/SAFETY.md` "not guaranteed" text if needed | S | — | author approval | text matches `research/audit_outputs/`; no new claims added |
| A4 | Add the deadlock classifier (Type I / Type II / slow progress) to `safenav/evaluation/metrics.py`, from the logic in `research/scripts/audit_diagnostics.py` | S | — | — | unit tests on synthetic trajectories for each class; reproduces the audit counts (122 / 67 / 6 for filter-trained, mean actions, expert) |
| A5 | Validation: 100 episodes per stage, the method's own deployment filter, store the full validation history, report best and final checkpoints | S | — | — | config option; test that selection is deterministic; logs contain both checkpoints |
| A6 | **Statistics module**: paired hierarchical bootstrap, logistic mixed model (statsmodels), seed-only and scenario-only intervals, Holm correction; `summary.csv` gains these columns | M | — | — | tests against hand-computed cases; reproduces `research/audit_outputs/paired_comparisons_expert.csv` within bootstrap error |
| A7 | **Frozen test sets and pre-registration**: new seed bases (EXPERIMENT_PLAN §2); `research/PREREGISTRATION.md` with hypotheses, primary contrasts, metrics and analysis; commit it before first use; a guard that refuses to evaluate frozen sets unless a `--confirm-frozen` flag and a committed pre-registration exist | M | — | A1, A2, A4, A6 | pre-registration committed and tagged; guard tested |
| A8 | Per-step logging (nominal action, speed limit, barrier minimum, executed action) for the first k episodes per condition | S | — | — | npz output; loader test |
| A9 | Deterministic aggregation (sort before summing) so that `summary.csv` is byte-reproducible | S | — | — | regenerating twice gives identical bytes |
| A10 | Hash manifest for `results/` (as in SafeTraj-Experiments) plus a CI check | S | — | — | CI fails if any protected file changes |
| A11 | Make the risk feature configurable: `ground_truth` (v0.2) or `from_ranges` (minimum of the scan); disclose it in the docs | S | — | — | test that `from_ranges` uses only the observation; default unchanged |

**Exit criterion for block A:** all tests pass; the v2 generator still reproduces the committed per-episode results; the pre-registration is committed.

---

## B. Minimum experiments to test the research hypothesis (Directions 1 + 2)

| ID | Task | Complexity | Compute | Depends on | Acceptance criteria |
|---|---|---|---|---|---|
| B0 | **Systematic literature search** (gate G3): Scopus/Web of Science plus arXiv, 2018–2026, queries combining (safety filter OR control barrier OR shield) × (reinforcement learning) × (deadlock OR liveness OR equilibri* OR stall*) × (navigation OR mobile robot); screening log with inclusion criteria | M | — | — | written log in `research/`; no paper found that already proposes a liveness-preserving filter for learned navigation policies (otherwise re-scope) |
| B1 | **Gate G0**: re-train M1 (no filter) and M2 (v0.2 filter) on generator v3, 10 seeds each; evaluate on validation-sized pilot sets (not the frozen ones) | S (code exists) | ≈ 70 | A1–A6 | continue if the filtered gap to oracle feasibility stays ≥ 15 points and each stall type is ≥ 20 % of timeouts in at least one variant |
| B2 | **Gate G1**: implement the correction penalty `α‖u − u_safe‖²` (M3) and a tanh-squashed Gaussian policy (M4); train 10 seeds each on pilot sets | M | ≈ 70 | B1 | continue to B3 if a gap of ≥ 10 points to the oracle-feasible rate remains for the best of M3/M4; otherwise re-scope (see NOVELTY_ASSESSMENT §4) |
| B3 | **Liveness-preserving filter (LPF)**: `(v, ω)` QP with look-ahead barrier and circulation term; closed form or a tiny QP solver; test-time prototype on frozen M1–M3 agents (**gate G2**) | L | ≈ 5 | B2 | unit tests: invariance under the adversarial policy (as in `test_safety_filter.py`), ω changes only when needed, no violation in 10⁵ random steps; G2: Type I stalls fall with a paired interval excluding 0 |
| B4 | **Theory**: proof of forward invariance (same assumptions as v0.2) and instability of single-obstacle boundary equilibria; written as `docs/LPF_PROOF.md` | L–XL | — | B3 | proof checked by a second person (supervisor or co-author); assumptions listed; a counterexample search (numerical) finds no violation |
| B5 | Executed-action parameterisation (M5) and `μ_v` drift logging (Direction 2 mechanism) | M | — | B2 | logs show `μ_v` per state bin near obstacles over training |
| B6 | Baselines: PPO-Lagrangian (M8); DWA with filter; MPC with DT-CBF (CasADi) as upper reference; tune all on validation with an equal budget | L | ≈ 60 (M8) + tuning | A5 | each baseline has tests and a recorded tuning log; DWA and MPC reach 0 collisions with exact state |
| B7 | **Core training matrix**: M6, M7 (+ M3, M7 at 20 seeds), plus the remaining M4/M5 seeds | S (orchestration) | ≈ 280 | B3, B5, B6 | all runs complete, with logs and checkpoints archived with hashes |
| B8 | **Frozen evaluation** on Test-ID and the three OOD sets; pre-registered analysis | M | ≈ 50 | A7, B7 | analysis script runs end to end from raw episodes to tables; primary contrasts reported with Holm-adjusted intervals whatever the outcome (a negative result is reported as such) |

**Decision after B8.** If H1 and H2 hold, go to C. If not, write up the diagnosis (the two stall mechanisms, penalty and parameterisation results) as a shorter empirical paper with honest negative results, and do not claim a method contribution.

---

## C. Additional experiments if the hypothesis holds

| ID | Task | Complexity | Compute | Depends on | Acceptance criteria |
|---|---|---|---|---|---|
| C1 | Secondary sweeps: margin (0.1/0.3/0.5, retrained), γ, look-ahead ℓ, density | M | ≈ 160 | B8 | curves with intervals; feasibility-conditioned and unconditional success |
| C2 | Noisy state and scan-based obstacles given to the filter; robust margin, and calibrated (conformal) margin as the Direction 3 seed | L | ≈ 140 + evaluation | B8 | empirical per-step violation ≤ α within Clopper–Pearson bounds; frontier of violations vs. deadlocks |
| C3 | Dynamic obstacles with known velocities (DT-CBF on predicted positions), plus a collision-cone CBF baseline | L | ≈ 70 | B3 | invariance test with moving obstacles under known motion; liveness metrics |
| C4 | **Standard simulator**: Gazebo or Isaac, TurtleBot3-class model, ROS 2 node completed and tested (extends `ros2_integration/`) | XL | ≈ 50 | B8 | sim-to-sim results on ≥ 200 scenarios; the node has integration tests; limitations stated |
| C5 | Hardware demonstration (if a robot and motion capture are available): ≥ 30 trials, pre-specified scenarios | XL | — | C4, lab access | trials logged with video; collisions and stalls reported; claims limited to the trials run |
| C6 | Reproducibility package: data release with DOI (Zenodo), manifest, one-command reproduction of all tables from raw episodes | M | — | B8 | a fresh clone on another OS reproduces the tables |

## D. Optional extensions (only after C, and only if they serve the paper)

| ID | Task | Complexity | Note |
|---|---|---|---|
| D1 | Multi-robot scenes (link to Grover / Jankovic / Chandra deadlock work) | XL | a separate paper |
| D2 | Learned or LiDAR-synthesised barrier (Keyumarsi et al. 2024; Long et al. 2021) instead of known circles | XL | removes the known-map assumption fully |
| D3 | Non-circular obstacles (polygons, SDF barrier) | L | needed for BARN-style maps |
| D4 | Off-policy algorithms (SAC) with the certified action stored | M | Pizarro Bejarano et al. note that off-policy methods differ here |
| D5 | Dashboard and figure improvements | S–M | **deprioritised**: no research value for the submission |

---

## Order and calendar (single researcher, estimate)

1. **Weeks 1–2:** A1–A11, and B0 started.
2. **Week 3:** B1 (G0), B2 (G1). Stop or re-scope here if a gate fails.
3. **Weeks 4–6:** B3 (G2), B4 (proof), B5, B6.
4. **Weeks 7–8:** B7 training, B8 frozen evaluation.
5. **Weeks 9+:** C-block as needed for the venue.

Writing the paper is out of scope for this audit, as requested.
