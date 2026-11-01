# Evidence-gap analysis and choice of paper direction

Date: 2026-10-25. Inputs: the five audit documents in `research/`, the code, committed checkpoints and results, new offline and evaluation-only analyses (`research/scripts/stall_taxonomy.py`, `paper_figures_stats.py`), and a refreshed literature check (Crossref, publisher and arXiv pages).

## 1. The audit's preliminary conclusions, re-tested

| Preliminary conclusion (treated as a hypothesis) | Verdict | Evidence |
|---|---|---|
| The safety proof alone is not a novel contribution | **confirmed** | It is the standard discrete-time CBF argument (Agrawal & Sreenath, RSS 2017, doi:10.15607/RSS.2017.XIII.073). |
| Training with a safety filter is already in the literature | **confirmed, and stronger than before** | Pizarro Bejarano et al., RA-L 2025 (filter in training + correction penalty + safe resets, PPO). Yang et al., *CBF-RL*, now **published at ICRA 2026** (doi:10.1109/ICRA57385.2026.11695808; it was a preprint at the time of the audit). Krasowski et al., TMLR 2023. Oh et al., IASEAI 2026 (theory). Thumm et al., IROS 2023 (proactive reduction of interventions). |
| Current experiments do not show that filter-based training improves performance | **confirmed** | Paired expert differences: −4.0 [−10.8, 3.2] points (mean actions), +6.4 [−1.6, 15.2] (sampled). |
| Sampled-action evaluation looks more promising, but its generalisation and novelty need scrutiny | **partly confirmed, and narrowed** | Sampling removes **self-stops only** (−21.8 [−31.6, −12.4] points, expert); filter-blocked stalls are unchanged (+0.8 [−7.6, 10.0]). Stochastic deployment is not novel as a method. It was chosen after inspecting test episodes and is untested on fresh scenarios. Its value for the paper is diagnostic: it separates the two mechanisms. |
| The contribution might be a filter that modifies heading and speed to reduce deadlock | **revised** | Proposition 1 (manuscript) shows that *any* heading modification keeps the one-step guarantee. A heading-level remedy is therefore easy to make safe; whether it reduces deadlock is untested, and it can only act on filter-blocked stalls (Prop. 3: self-stops are untouched by speed limits). Close prior art: proactive action modification (Thumm et al. 2023) and liveness through instability (Jankovic et al., TCST 2024). As a stand-alone contribution it is **not yet supported**, so it moves into the paper as experiment E2, not as the headline. |
| Test-scenario design and goal-distance inconsistencies must be addressed | **confirmed; partly addressed** | Documented in README/SAFETY (PR #2). The generator itself is unchanged; fresh-scenario evaluation (E3) is still required. |

## 2. Candidate directions compared

Scores 1 (weak) to 5 (strong) are judgements made from the evidence above, not measurements.

| Criterion | A. Analysis of deadlock mechanisms in filtered RL navigation (theory + taxonomy + protocol) | B. New liveness-preserving heading/speed filter (method paper) | C. Training remedy (penalty, bounded policy) | D. Calibrated, uncertainty-aware filter (safety–liveness frontier) |
|---|---|---|---|---|
| Novelty | 3: no published characterisation of learned-policy deadlock mechanisms found (hypothesis; systematic search pending) | 3: close to Thumm et al. 2023 and Jankovic et al. 2024; risk of being judged incremental | 1–2: the penalty is published (Pizarro Bejarano 2025); bounded policies are standard | 2–3: Daş et al., RA-L 2026, and Cosner et al. are close |
| Significance | 3: explains why filtered RL loses success, and which remedy can act on which cause | 4 if it works | 3 | 4 |
| Theory | 2–3: short but exact propositions (Prop. 1–3, Cor. 1) | 3–4: would need a liveness argument | 2 | 4 |
| Existing evidence | **4**: taxonomy, geometry and paired contrasts already computed from committed checkpoints | 1: none | 1: none | 0: none |
| Compute | **5**: done; the remaining items are minutes to a few hours | 3: test-time version cheap; training version ≈ 10–20 core-h | 3: ≈ 5–10 core-h | 2: sensor simulation, many evaluation episodes |
| Reproducibility | 5 | 4 | 4 | 3 |
| Journal fit (Q1, realistic) | 3: RAS / Autonomous Robots; RA-L if E1/E2 give a clear result | 4 if it works | 2 | 4 |

**Selected: Direction A**, absorbing the cheap, decisive parts of B and C as experiments:
- E1: the published penalty, as the strongest existing remedy;
- E2: a test-time heading escape, safe by Prop. 1.

**Direct statement on novelty.** A *new safety filter* (Direction B) is not sufficiently supported to be the paper's headline. There is no evidence for it yet, and its novelty relative to Thumm et al. (2023) and Jankovic et al. (2024) is uncertain. The defensible contribution is the **analysis**:
- the exact properties of the one-step speed filter;
- the two learned-policy deadlock mechanisms and how training shifts between them;
- an evaluation protocol that makes them measurable.

This is a narrower claim. It is only competitive for a Q1 journal if E1 and E2 turn the analysis into actionable conclusions (which remedy fixes which mechanism), and if E3 shows the findings hold on fresh scenarios.

## 3. Evidence gaps (what is missing, why it matters, what closes it)

| # | Gap | Why a reviewer will care | Closes it | Severity |
|---|---|---|---|---|
| G1 | **Correction-penalty baseline** (Pizarro Bejarano et al. 2025) | It is the closest published remedy; if it removes the deadlocks, the paper's conclusions change | E1a (not run). **Pilot done (2026-10-26)**, one seed per arm:
- The configuration was verified, and the design is feasible: 0 collisions, and the control shows the stalls.
- No large effect: deadlock −1.2 [−5.2, 2.8] points, scenario-only interval. This is **inconclusive**, not evidence against the penalty.
- The gap stays open until E1a. | critical |
| G2 | **Remedy for filter-blocked stalls** | Without it, the analysis ends at "here is a problem" | **partly closed (2026-10-26)**: the test-time heading escape lowers Type I by 7.4 [4.8, 10.2] points and raises success by 7.2 [5.0, 9.7] points (agents trained without the filter). This is below the pre-registered 10-point threshold, and 523/611 timeouts are still Type I. A stronger (multi-step) remedy is still missing | major |
| G3 | **Fresh, held-out scenarios** | All analyses used the 300 test scenarios on which the margin and the sampled-action evaluation were chosen | **partly closed (2026-10-26)**: the taxonomy and both training contrasts replicate on 500 unused scenarios of the *same* generator (pre-registered). Still open: corrected generator, other stages, sampled actions | major |
| G4 | **Systematic literature search** for prior analyses of learned-policy deadlock behind filters | The novelty claim rests on a targeted search | protocol in `research/IMPLEMENTATION_ROADMAP.md` B0; no compute | critical |
| G5 | Classical / safe-RL baselines (DWA or MPC with DT-CBF; PPO-Lagrangian) | RL navigation papers without classical baselines are routinely criticised (Xiao et al. 2022 survey) | E6 | major |
| G6 | Mechanism of self-stops (action aliasing vs. reward) | The Discussion offers a hypothesis only | E4. New evidence: self-stops *increase* with longer training (final vs. selected checkpoints, +11.7 [3.0, 20.8] points), so they are learned, not an early-checkpoint artefact. The cause is still untested | major |
| G7 | ~~Window-length sensitivity of the taxonomy~~ | construct validity | **closed by E5** (counts change by ≤ 4 for K = 50–200) | — |
| G8 | Formal check of filter permissiveness in the sense of Oh et al. 2026 | links the analysis to theory | analytical; no compute | moderate |
| G9 | More seeds (10+) for method-level claims | 5 seeds give approximate intervals | E7 (optional) | moderate |
| G10 | Second reader for Props. 1–3 | the proofs are short but not peer-checked | supervisor or co-author | moderate |
| G11 | Standard simulator (Gazebo/Isaac) or hardware | external validity; expected by RA-L/T-RO reviewers | E8 (optional, heavy) | venue-dependent |
| G12 | Archived release with a DOI | reproducibility requirement of most journals | Zenodo release after the experiments | minor |

## 4. What the current evidence does and does not support

**Supported now:**
- the two-mechanism taxonomy on this benchmark, replicated on 500 unused scenarios (same generator, pre-registered);
- the shift between mechanisms caused by the training setup (paired intervals exclude 0 on two stages and on the fresh scenarios);
- a modest, safe, test-time reduction of filter-blocked stalls by a heading escape, with no effect on self-stops;
- self-stops are not an artefact of early checkpoint selection;
- the mechanism-specific effect of sampling;
- the geometry predicted by Prop. 2;
- the absence of a measurable benefit of training with the filter on success (on the fresh scenarios the filter-trained agents were 4.4 [0.2, 8.5] points *lower*; exploratory).

**Not supported now:**
- any claim about whether the published correction penalty fixes either mechanism. Only a single-seed pilot exists: it showed no large effect, which is inconclusive. E1a has not been run;
- generalisation to other scenario generators, stages or sampled actions in the fresh evaluation;
- the causal explanation of self-stops;
- anything about real robots or other simulators.
