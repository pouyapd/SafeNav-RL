# Journal targeting

Date: 2026-10-25.

## How the journal information was verified

- **Quartiles.** SCImago (SJR 2025) values come from search-engine snippets of the official SCImago journal pages. Direct access to scimagojr.com was refused to automated tools (HTTP 403), so **open each SCImago page in a browser before relying on the quartile**.
- **JCR.** Clarivate (JCR) quartiles were not checked; JCR access needs a subscription. Many institutions use JCR, not SJR, to define "Q1", so check which list your institution or funder uses.
- **Costs and guidelines.** Where only third-party pages were found, this is marked; confirm on the publisher's own page before submitting.
- **No acceptance is promised or predicted.**

## Candidates

| Journal | SJR 2025 (best quartile) | Source | Model / costs | Length |
|---|---|---|---|---|
| **IEEE Robotics and Automation Letters (RA-L)** | 1.589, Q1 (AI; Control and Optimization) | SCImago page via search snippet | IEEE hybrid OA; OA fee **not verified**. Overlength charge $175 per page beyond 6, maximum 8 pages (IEEE RAS author page, undated: verify) | 6 + 2 pages including figures and references |
| **Robotics and Autonomous Systems** (Elsevier) | 1.436, Q1 (e.g. Control and Systems Engineering; Software) | SCImago via snippet | hybrid; optional OA. APC $2,840 from a third-party directory dated 2021 (**outdated**: check the Elsevier Guide for Authors) | full-length articles |
| **Autonomous Robots** (Springer) | 1.502, Q1 (AI) | SCImago via snippet | hybrid; APC $3,390 per a third-party site (**verify** on Springer). OA chosen after acceptance | ≈ 75,000 characters including spaces, figures counted as 2,000 each (mirror of the guidelines: verify) |
| **IEEE Transactions on Automation Science and Engineering (T-ASE)** | 2.122, Q1 (Control and Systems Engineering) | SCImago via snippet | IEEE hybrid; fees not verified | not verified |

## Fit and risks per journal

### Robotics and Autonomous Systems: best fit for the current direction (after E1–E3)

- **Scope fit.** Good. Autonomous navigation, learning-based control and safety analysis are core topics.
- **Novelty expectations.** A sound analysis with clear, generalisable insight and solid experiments can be enough. Pure benchmarking without insight is usually not.
- **Rigour expected.** Multiple seeds, uncertainty estimates, baselines including classical planners, and ablations.
- **Theory expected.** Moderate. The propositions are adequate if correct and clearly connected to the experiments.
- **Main rejection risks.**
  - (1) "Toy 2D environment": mitigate with E3, E6 and, ideally, E8.
  - (2) "Known phenomenon": mitigate with the systematic search (G4) and precise positioning against Reis et al. 2021 and Pizarro Bejarano et al. 2025.
  - (3) Missing remedy results (E1/E2).
- **Length.** A full article allows the analysis, the taxonomy and the remedies together.

### Autonomous Robots: good fit, similar to RAS

- **Scope.** Similar to RAS, with somewhat stronger expectations of robotics relevance.
- **Hardware.** Expected more often than in RAS; a Gazebo or hardware component (E8) would help noticeably.
- **Policy.** No resubmission after a final decision (per the guidelines mirror; verify), so submit only when the evidence is complete.

### IEEE RA-L: possible if E1/E2 give a clear, compact result

- **Scope.** Excellent; the topic is central to RA-L.
- **Novelty expectations.** High. Reviewers look for a new method or a clearly new insight; an analysis-only letter is at risk.
- **Rigour expected.** Strong. A real-robot or standard-simulator validation is common, and its absence is a frequent criticism.
- **Length.** 6–8 pages forces a focused story: the two mechanisms, Prop. 1–3, and the remedy results. The bootstrap details and the full taxonomy go to supplementary material.
- **Advantage.** RA-L papers can be presented at ICRA or IROS.
- **Main rejection risks.** Insufficient novelty, no hardware, a simple environment.

### IEEE T-ASE: weaker fit

- Automation-science framing; it would need emphasis on systematic validation and deployment, so it is less natural for this paper. Listed for completeness.

### Not recommended now

- **IEEE T-RO and IJRR.** They require major theoretical or system contributions and real-robot evidence.
- **ML venues (e.g. TMLR).** Possible for a mechanism paper if E4 yields strong evidence about action aliasing, but that is a different paper.

## Readiness assessment (realistic)

| Item | Status |
|---|---|
| Core findings (taxonomy, contrasts, geometry) | ready, verified, reproducible from committed files |
| Theory (Props. 1–3, Cor. 1) | written; needs a second reader |
| Remedy evidence (E1, E2) | **missing**: decisive for the paper's message |
| Fresh-scenario evidence (E3) | **missing** |
| Classical / safe-RL baselines (E6) | **missing** |
| Systematic literature search (G4) | **missing** |
| Archived release with DOI | missing |

**Overall: not ready for submission.** After E1a, E2, E3 and E6 plus the literature search, which is about 2–3 weeks of part-time work and under 15 core-hours of computation, the manuscript would be a credible submission to **Robotics and Autonomous Systems** or **Autonomous Robots**. RA-L becomes realistic only if E2 (or E1) gives a strong, clean remedy result and the story fits into 8 pages. Even then, the lack of hardware validation is a major risk there.

**Next formatting step.** Only after the target is chosen: Elsevier `elsarticle`, the Springer Nature template, or IEEE `IEEEtran`. The current draft uses the neutral `article` class on purpose.
