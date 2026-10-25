# Figures, tables and claim-to-evidence map

Draft: `paper_draft/manuscript.tex` (branch `paper/manuscript-draft`).

Every figure and table below is generated from files in the repository. Status:
- **exists**: committed and used as is;
- **new**: generated for this draft from stored outputs, without new training;
- **required**: needs a new experiment (see `minimum_experiments.md`).

## Figures

| # | Content | Status | Source data | How to regenerate |
|---|---|---|---|---|
| Fig. 1 `stall_taxonomy.pdf` | timeouts per 100 episodes by stall class, intermediate and expert, 4 PPO conditions + 2 baselines | new | `research/audit_outputs/stall_taxonomy_all.csv` | `python research/scripts/stall_taxonomy.py` (≈5 min, 8 processes; evaluation only) → `python research/scripts/paper_figures_stats.py` (seconds) |
| Fig. 2 `stall_geometry.pdf` | (a) heading angle at Type I stalls; (b) barrier value at Type II stops | new | same | same |
| Fig. 3 `paired_differences.pdf` | forest plot of paired differences (success, stall rates), expert | new | `research/audit_outputs/paired_comparisons_expert.csv`, `paper_draft/generated/stats.json` | `python research/scripts/audit_diagnostics.py` (≈2.5 min) → `paper_figures_stats.py` |
| (candidate) `cbf_speed_limit.png` | speed limit field around obstacles; illustrates Prop. 2 | exists, not yet in the draft | `docs/figures/` | `python -m safenav plot ...` |
| (candidate) `failure_cases_expert.png` | first six failed expert episodes, filter-trained seed 0 (4 self-stops, 1 filter-blocked, 1 moving) | exists, not yet in the draft | `results/benchmark/` + checkpoints | `python -m safenav plot ...` |
| (candidate) `safety_margin_sweep.png` | oracle feasibility vs margin | exists | `results/safety_sweep/` | `python -m safenav plot ...` |
| Fig. R1 | Type I/II rates with the correction penalty (E1) | **required** | E1 outputs | `research/experiments/e1_evaluate.py` (branch `exp/e1-correction-penalty`) |
| Fig. R2 | Type I rates with the heading-level escape ψ (E2) | **required** | E2 outputs | E2 not implemented yet |

## Tables

| # | Content | Status | Source |
|---|---|---|---|
| Table 1 | positioning against related work | new | `research/LITERATURE_REVIEW.md` (reading depth per paper); bibliography verified against Crossref |
| Table 2 | expert success / collision / timeout / interventions, 9 conditions | exists (numbers) | `results/benchmark/summary.csv`, `results/stochastic_eval/summary.json` |
| Table 3 | stall-class counts, intermediate and expert | new | `research/audit_outputs/stall_taxonomy_all.csv` |
| Table R | remedies: success and Type I/II rates for penalty, ψ, stochastic deployment | **required** | E1, E2 |

## Claim-to-evidence map (manuscript sections)

| Claim | Where | Evidence file(s) | Verified how |
|---|---|---|---|
| 0 collisions in 3,600 filtered test episodes; minimum clearance 0.300 m | §6.1 | `results/benchmark/episodes.csv` | recomputed (audit §5.2) |
| 94.6 % → 65.0 % expert success when the filter is added at test time; paired difference 29.6 [21.6, 38.4] | §6.1 | `summary.csv`; `paired_comparisons_expert.csv` | recomputed |
| filter-trained vs. unfiltered-trained: −4.0 [−10.8, 3.2] (mean), +6.4 [−1.6, 15.2] (sampled) | §6.1 | `paired_comparisons_expert.csv` | paired hierarchical bootstrap, 10,000 replicates |
| Type I 141/175 (trained without filter), Type II 122/195 (trained with filter), expert | Abstract, §6.2 | `timeout_attribution_expert.csv`, `stall_taxonomy_all.csv` (identical counts) | two independent scripts agree |
| Type II median barrier 0.41 m (IQR 0.16–0.72), expert, filter-trained, mean | §6.2 | `timeout_attribution_expert.csv` | recomputed; the audit's earlier "0.35 m" was pooled with sampled actions (corrected) |
| 89/122 self-stops with the goal within 45° of the obstacle direction; median \|ω\| 0.04 rad/s | §6.2 | `timeout_attribution_expert.csv` | recomputed |
| go-to-goal: 54/54 expert timeouts filter-blocked; potential field 0 filter-blocked | §6.2 | `stall_taxonomy_all.csv` | recomputed |
| taxonomy insensitive to displacement and share thresholds (≤ 3 change) and to the window length K = 50–200 (≤ 4 change) | §6.2, §8 | `research/audit_outputs/taxonomy_threshold_sensitivity_expert.csv`, `research/audit_outputs/taxonomy_window_sensitivity_expert.csv` | offline recomputation; window check by re-simulation (48 s) |
| 617/618 Type I stalls have heading < 90°; max 91.3°; median 66.8°; barrier ≤ 4.2 mm; no wall barrier < 1 mm | §6.3 | `stall_taxonomy_all.csv`; `stats.json` | recomputed |
| training with the filter: Type I −14.8 [−24.4, −5.6], Type II +17.6 [8.2, 27.0] (expert); intermediate −8.0 / +11.0 | Abstract, §6.4 | `stats.json` (`stall_contrasts`) | paired hierarchical bootstrap |
| sampling: Type II −21.8 [−31.6, −12.4], Type I +0.8 [−7.6, 10.0] (filter-trained, expert) | §6.4 | `stats.json` | same |
| scenario term 25–86 % vs. seed term < 2 % of the sum of squares | §6.5 | `research/audit_outputs/variance_components_expert.csv` | descriptive two-way split |
| ≈ 10 seeds × 1,000 scenarios → half-width ≈ 4–5 points | §6.5 | `stats.json` (`ci_halfwidth_fit`) | approximate fit from 5 seeds; labelled approximate |
| consistency: the taxonomy re-run reproduces the benchmark outcomes for all 3,600 mean-action / baseline filtered episodes | §5 | `stats.json` (`consistency_with_benchmark`: 0 mismatches) | automated check |
| Lemma 1, Props. 1–3, Cor. 1 | §4 | proofs in the manuscript; Lemma 1 also tested in `tests/test_safety_filter.py` | Props. 1–3 are short derivations; not machine-checked; **a second reader should check them** |
| all three stages feasible at 0.3 m; 53/100 expert goals < 5 m; 3 trivial scenarios | §5 | `results/benchmark/scenarios.csv`, `research/audit_outputs/goal_distances.csv` | recomputed |
