# Q1 readiness review and submission decision (2026-11-17)

> **Update after integration (2026-11-17).** All PRs, #2–#4 and #6–#16, were merged into `main` in stack order with merge commits. The branch and PR audit in §1 and the merge advice in §7 describe the state *before* that integration and are kept as written.

This review covers the state of `paper/submission`, which builds on `paper/round4` (draft PR #15). It is an internal review written as three reviewers and an editor, not an external peer review. **No acceptance is predicted.**

## Decision

**NEARLY READY: only minor work remains.**
- The scientific evidence is complete for an empirical analysis paper, and no further experiment is essential.
- What remains are author-only items:
  - affiliation;
  - declarations;
  - an archived DOI;
  - a check of the JCR quartile in Clarivate;
  - a final read.
- The main risk is not missing work but the reviewers' expectations for a robotics journal: there is no physical robot, the simulation is simple, and the recovery planner is methodologically standard. These cannot be fixed by writing. The manuscript now states them openly.

## 1. Audit of the repository and manuscript

Checked directly on 2026-11-17, not taken from earlier reports.

| Area | State | Finding |
|---|---|---|
| Branches and PRs | Verified | <ul><li>Draft PRs #2–#4 and #6–#15 are open; none is merged. #1 and #5 were merged earlier (#5 into `exp/e1-pilot`).</li><li>`main` is `d257eca`, unchanged, and is the default branch.</li><li>The stack is strictly linear: `main` ⊂ #2 ⊂ #3 ⊂ #4 ⊂ #6 ⊂ … ⊂ #15. Every head contains its base.</li></ul> |
| CI | Verified | Green on the heads of #13, #14 and #15 (push and PR runs). 166 tests pass locally |
| Raw data | Verified | <ul><li>Round 4: 26,600 confirmatory and 4,000 latency rows, with no duplicate keys and no NUL bytes.</li><li>Rounds 1–3 were audited earlier with separate code: 223 checks for round 2 and 68 for round 3, 0 failed.</li></ul> |
| Numbers in the manuscript | Verified | <ul><li>`research/scripts/verify_manuscript_round4.py`: 78 of 78 round-4 numbers match the saved results.</li><li>Round-3 numbers were checked by `audit_round3.py` (section C).</li></ul> |
| Statistical scripts | Present | `analyze_round{3,4}.py`, `tools/verify_primary.py` and `audit_round3.py`, all pre-registered or independent |
| Bibliography | Verified | <ul><li>`research/scripts/verify_bibliography.py`: 52 of 52 DOIs resolve on Crossref, and title, first author and year match. One entry was flagged only because Crossref stores the first author as "Hsu*"; it is correct.</li><li>The 11 entries without a DOI (ICML, NeurIPS, JMLR, TMLR, arXiv) were checked by hand.</li><li>Seven verified references were added.</li></ul> |
| `[EVIDENCE REQUIRED]` markers | Resolved | <ul><li>Before: 2 (missing PPO-Lagrangian/MPC baselines; archive DOI).</li><li>The first is now a stated limitation.</li><li>The second is an "Author to confirm" item, because only the author can create the archive.</li></ul> |
| Figures | Checked visually | <ul><li>All are vector PDFs with readable labels.</li><li>Collisions are shown separately in the fairness figure.</li><li>Uncertainty is given in the tables (CIs) rather than in the stacked-bar figures.</li></ul> |
| Reproducibility | Good | <ul><li>Protocols were committed before every confirmatory run.</li><li>Seeds are fixed; runs are resumable and have run records.</li><li>Committed checkpoints, per-episode CSVs and reproduction commands are available.</li><li>Gap: five final checkpoints are withheld, and the 10 × 1M-step training runs were not repeated.</li></ul> |

### Ten audit dimensions (short form)

1. **Scientific completeness.** The questions RQ1–RQ4 are answered with confirmatory evidence. Open questions are named as such: transfer to real robots, dynamic obstacles and real perception.
2. **Novelty and contribution.** This is an empirical diagnosis plus intervention study. The novelty is in the questions and the evidence:
   - the two-mechanism taxonomy and its dependence on training;
   - the negative results of training interventions;
   - recovery as an intervention, with its limits.

   The filter, the correction penalty and the planner are not new. The manuscript now says this explicitly.
3. **Experimental validity.**
   - **Strengths:**
     - a feasibility oracle;
     - unused and corrected scenario sets;
     - a paired design;
     - safety recorded in every episode.
   - **Weaknesses:**
     - a 2D unicycle with static circles and an exact-state filter;
     - an untuned PPO;
     - 300k-step arms in E1a and E4.
4. **Statistical reliability.**
   - **Design:** seeds are the unit of replication; the hierarchical bootstrap includes scenarios; Holm adjustment is applied within each pre-registered family; seed-level tests are reported.
   - **Limits:** five seeds per arm; the bootstrap with five seeds understates seed variance, which the manuscript states.
5. **Literature coverage.**
   - **Adequate for:** CBFs, deadlock, safe RL with filters, clipping and projection, and evaluation practice.
   - **Added in this pass:** recovery behaviours and hybrid planning.
   - **Not systematic:** the search is described as targeted, not systematic.
6. **Reproducibility.** Strong, as described in the audit table above.
7. **Structure and writing.**
   - Improved in this pass:
     - a title that covers the recovery work;
     - a 250-word abstract;
     - RQ4;
     - five contributions;
     - a new method section with pseudocode;
     - a new discussion paragraph;
     - a rewritten conclusion.
   - The main text is still long: about 15,900 words and 33 pages.
8. **Figures and tables.** Consistent, with a source named in every caption. Supplementary tables S1–S7 and two supplementary figures were added.
9. **Limitations.** Explicit and specific: simulation only, exact map, planner-only result, latency not a real-time guarantee, five seeds, untuned baselines and a privileged feature.
10. **Journal requirements (RAS).**
    - Met:
      - an abstract of 250 words or fewer;
      - highlights (5 bullets of at most 85 characters);
      - keywords;
      - a data availability statement.
    - Open: the declarations, including the generative-AI declaration that Elsevier requires, need the author.

## 2. Three-reviewer critique

Severity: C = Critical, Maj = Major, Min = Minor. "Writing" means fixable by writing or analysis; "Exp." means it needs new experiments.

### Reviewer 1: robotics and motion planning
| # | Weakness | Sev. | Why it matters | Fix | Done? |
|---|---|---|---|---|---|
| 1.1 | Simulation only: 2D unicycle, static circular obstacles, exact state for the filter | Maj | Robotics journals value hardware or at least realistic simulators; external validity is limited | Exp. (B/C); writing: state the scope early and plainly | Scope stated in the abstract, introduction and limitations |
| 1.2 | The recovery planner is methodologically standard: sampled manoeuvres with a navigation-function cost, as in the global DWA, plus stuck detection as in Nav2 | Maj | A planning reviewer may see no technical contribution | Writing: frame the planner as an intervention, not a method; cite the precedents | **Done**: new related-work paragraph and an explicit "not claimed" sentence |
| 1.3 | The planner alone, with the exact map, beats policy plus recovery | Maj | Undercuts any claim that the hybrid is useful in this task | Writing: report it prominently and frame the contribution as diagnosis and remedy for a given policy | **Done**: abstract, highlights, discussion, conclusion |
| 1.4 | Global-map dependency; a wrong map restores about 20 % deadlock | Maj | Real deployments have imperfect maps | Writing (results are in hand) | **Done** |
| 1.5 | Real-time feasibility shown only on one laptop | Min | Not a guarantee | Writing | **Done**: stated in every place it is used |
| 1.6 | No dynamic obstacles; no second robot model | Min | Limits generality | Exp. (C) | Stated as an open question |

### Reviewer 2: RL and safe AI
| # | Weakness | Sev. | Why it matters | Fix | Done? |
|---|---|---|---|---|---|
| 2.1 | No constrained-RL baseline without a filter (PPO-Lagrangian) or MPC with a DT-CBF | Maj | Reviewers will ask whether a filter-free safe-RL policy also deadlocks | Exp. (B): about 1–1.5 h for PPO-Lagrangian, 5 seeds × 300k steps plus evaluation; MPC needs new code. Writing: stated as a limitation | Limitation stated |
| 2.2 | Five seeds per arm; the new arms were trained for 300k steps rather than 10⁶ | Maj | Effects of a few points are not resolvable; the budget may favour some arms | Writing: stated, with seed-level results; more seeds is Exp. (C) | Stated |
| 2.3 | PPO hyperparameters not tuned for the filtered setting | Min | Could change the balance of the mechanisms | Writing | Stated |
| 2.4 | The taxonomy thresholds are heuristic | Min | Construct validity | Done earlier: sensitivity analysis (counts change by at most 4) | In the paper |
| 2.5 | The proximity feature uses the true map | Min | A privileged observation | Writing | Stated |
| 2.6 | Claims vs data: the old conclusion overgeneralised ("removing one outcome does not resolve it") | Maj | Overclaiming | Writing | **Fixed**: the conclusion now separates training interventions from deployment remedies, with numbers |

### Reviewer 3: editor and statistician
| # | Weakness | Sev. | Why it matters | Fix | Done? |
|---|---|---|---|---|---|
| 3.1 | The abstract was about 595 words, above the 250-word limit | C (format) | Desk rejection | Writing | **Fixed**: 250 words |
| 3.2 | The title and contributions did not reflect rounds 3–4 | Maj | Misleading framing | Writing | **Fixed** |
| 3.3 | The recovery method was described only inside Results | Maj | Readers cannot judge the method | Writing | **Fixed**: Section 5 with pseudocode; Table S2 |
| 3.4 | Length: about 15,900 words, 12 result subsections | Maj | Reviewer fatigue; RAS sets no limit, but long papers are reviewed more harshly | Writing: two analyses moved to the supplement in this pass; further condensing is possible (Remaining work) | Partly |
| 3.5 | Many secondary contrasts are unadjusted | Min | Risk of over-reading | Writing: labelled as secondary/exploratory throughout; Holm applied within each primary family | In place |
| 3.6 | The bootstrap with five seeds understates seed variance | Min | Intervals are approximate | Writing | Stated |
| 3.7 | Confirmatory vs exploratory status | Min | Integrity | Writing: every exploratory result is labelled; pre-registration commits listed (Table S3) | **Done** |
| 3.8 | Red `[EVIDENCE REQUIRED]` markers in the PDF | C (format) | Cannot be submitted | Writing / author | **Fixed**; blue author-only markers remain on purpose |

## 3. Reassessed contribution

The strongest defensible contribution, as now written:

**Main contribution.** An empirical analysis of how learned navigation policies interact with a CBF safety filter. It covers:
- two deadlock mechanisms, filter-blocked and self-stop;
- their dependence on training;
- the finding that removing self-stops in training moves the stalls to the boundary rather than removing them.

**Supported by.** Elementary properties of the filter that explain where each mechanism can occur.

**Plus.** Evidence from a recovery intervention:
- multi-step planning at stalls, with the filter unchanged, removes almost all deadlock;
- this works only through global information;
- a planner without the policy does better still;
- the remaining timeouts come from the filter's conservativeness.

**Plus.** An open, pre-registered evaluation framework.

**Not claimed:**
- a new filter, training algorithm or planner;
- that policy plus recovery beats a planner with the same map (it does not);
- real-world safety or real-time guarantees.

How each finding listed in the brief is reflected:

| Finding | Where it appears |
|---|---|
| 1. Correction penalty: primary endpoint null | Abstract, Section 7.10, contribution 3 |
| 2. E3 partly supported | Section 7.8 (3 of 4 criteria in the expert stage) |
| 3. E4 removed self-stops, not deadlock | Abstract, Section 7.10, conclusion |
| 4. Round 3 reduced deadlock | Abstract, Section 7.11 |
| 5. Round 4 efficiency and robustness | Section 7.12 |
| 6. Planner alone 95 % vs 85–86 % | Abstract, highlights, Section 7.12, discussion, conclusion |
| 7. Wrong map degrades | Abstract, Table 7, discussion |
| 8. About 14 % moving timeouts | Abstract, Sections 7.11–7.12, conclusion |
| 9. No collisions | Throughout, with clearance 0.300 m |
| 10. No real-world or real-time guarantee | Abstract ("All results are from simulation"), Section 7.12, limitations |

## 4. Literature

**Verified.** All 52 DOIs in the bibliography were checked against Crossref; see `research/audit_outputs/bibliography_check.csv`.

**Added**, each verified on Crossref and each answering a specific reviewer question:

| Reference | Why |
|---|---|
| Brock & Khatib 1999, global dynamic window approach | Closest classical precedent of the recovery planner |
| Rimon & Koditschek 1992, navigation functions | Why a shortest-path cost-to-go has no spurious minima |
| Marder-Eppstein et al. 2010; Macenski et al. 2020 | Recovery behaviours in navigation stacks |
| Faust et al. 2018, PRM-RL | Hybrid RL and planning |
| Bastani & Li 2021, model predictive shielding | Backup policies for safety |

Thananjeyan et al. 2021 (Recovery RL) was already in the bibliography and is now cited.

**Closest competing work and the difference:**
- *Recovery RL and MPS/DMPS:* recovery is for **safety**. Here safety is left to the unchanged filter, and recovery is for **liveness**.
- *Global DWA and Nav2 recovery:* the same planning ingredients, but used with a classical local planner, not to diagnose a learned policy behind a CBF filter.
- *Undesired CBF equilibria (Reis, Tan):* model-based nominal controllers, not learned ones.
- *Pizarro Bejarano et al.:* the correction penalty in trajectory tracking without obstacles.
- *Yang et al.:* a single integrator, with no heading geometry.

**Wheelchair navigation.** This was not added. No reference on autonomous wheelchairs bears on a claim of this paper, and adding one would only inflate the citation count. It would become relevant only if the paper claimed an application to assistive mobility, which it does not.

**Search type.** The search remains targeted, not systematic, and the manuscript says so.

## 5. Are new experiments necessary? (Phase 7)

**A — essential before submission: none.** Every claim in the manuscript is supported by existing confirmatory data.

**B — strongly recommended.** Optional; none runs without your approval.

| Item | Scientific justification | Expected contribution | Runtime estimate | Essential for RAS? |
|---|---|---|---|---|
| Planner-only with the sensed map (200 expert scenarios, exploratory) | Part F gives the planner the exact map; reviewers will ask whether its advantage survives realistic map information | Closes the one fairness gap named in the paper | about 10–20 min of compute, plus about 1 h of code and text | No |
| PPO-Lagrangian baseline (5 seeds × 300k steps, filter off in training, evaluated with and without the filter) | Answers whether constrained RL without a filter shows the same stall mechanisms | Addresses Reviewer 2.1 | about 1–1.5 h of training and evaluation on mains power, plus about 2 h of code and text | No; the paper does not claim anything about constrained RL |

**C — optional future work:**
- a physical robot or a realistic simulator (Gazebo/Isaac);
- dynamic obstacles;
- MPC with a DT-CBF;
- 10 seeds per arm;
- 10⁶-step E1a/E4 arms;
- a learned recovery policy.

## 6. Journal strategy

**How this was verified (2026-11-17):**
- **SJR:** quartiles come from SCImago pages, read via search snippets, because scimagojr.com refuses automated access.
- **JCR (Clarivate):** quartiles could **not** be verified directly, since JCR needs a subscription. The values below come from a third-party aggregator (Peeref) and must be checked in JCR before relying on them.
- **Requirements:** journal requirements come from the publishers' pages where these were reachable.

| Rank | Journal | Fit | SJR (SCImago) | JCR (unverified; Peeref) | Requirements found | Costs |
|---|---|---|---|---|---|---|
| **Primary** | **Robotics and Autonomous Systems** (Elsevier, ISSN 0921-8890) | Best. Robot control and learning for autonomous systems, including theoretical and computational work. Simulation studies of learned navigation appear regularly | SJR 2025 1.436, best quartile Q1; Control and Systems Engineering Q1 in 2021–2023 | JCR 2026 edition: Robotics **Q1**, Automation & Control Systems Q1, CS-AI Q2. The JCR 2023 edition listed Robotics **Q2**. **Check in Clarivate** | Abstract ≤ 250 words; 3–5 highlights of ≤ 85 characters (mandatory); no article length limit found | Hybrid: subscription publication free of charge; open-access APC USD 2,940 (OA page) or 3,140 (Insights page). **Confirm** |
| Alternative 1 | **Engineering Applications of Artificial Intelligence** (Elsevier, ISSN 0952-1976) | Good. Applied AI with robotics in scope; safe RL for navigation fits. Expects a clear engineering framing | SJR 2025 1.782; Q1 in Artificial Intelligence every year 2014–2025 | JCR 2023 edition: Q1 in every listed category (Automation & Control Systems; CS-AI; Electrical & Electronic Engineering; Engineering, Multidisciplinary) | Elsevier format; highlights are likely (check the guide for authors) | Hybrid; APC not verified |
| Alternative 2 | **Journal of Intelligent & Robotic Systems** (Springer, fully open access since 2024) | Good topical fit; most realistic acceptance prospects | SJR 2025 0.720, best quartile Q1; the AI category was mostly Q2 in earlier years | IF 3.1 (2023). JCR quartile not found; probably **not** JCR Q1 | Not verified (pages require login redirects) | APC applies (open access); amount not verified |

**Excluded:**
- *Autonomous Robots:* its scope gives preference to papers with data from physical robots, and an earlier check found a length limit of about 75,000 characters.
- *IEEE RA-L:* the letter format (6 + 2 pages) would need about a 70 % cut.
- *IEEE T-RO and IJRR:* novelty and hardware expectations are too high for a simulation-only analysis.

**Ranking rationale.** RAS ranks first on scientific fit and on its acceptance of computational studies; its JCR quartile has moved between Q2 and Q1, so check it. EAAI is the safest if your institution requires JCR Q1, but its readership is less robotics-specific. JIRS is the most realistic for acceptance but is likely not JCR Q1.

## 7. Recommended merge sequence (for when you decide to merge; nothing is merged)

1. Merge #2, then #3, then #4 into `main`. They already target `main` and contain each other.
2. Then retarget #6 → `main`, merge; continue in order: #7, #8, … #15, then the submission PR.
3. Alternatively, after reviewing the whole stack, open one PR from `paper/submission` to `main` and close the stacked PRs as "included".
4. **Use "Create a merge commit" (no squash, no rebase).** The manuscript and protocols cite commit hashes (for example `c0629f7`, `f0194a1`, `c1abd93`) as evidence of pre-registration. Squashing or rebasing would remove those commits from `main`'s history.
5. Keep every branch after merging.
