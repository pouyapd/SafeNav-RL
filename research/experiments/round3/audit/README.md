# Audit of round 2 (PRs #7–#9), before round 3

The audit script is [`../audit_round2.py`](../audit_round2.py). It shares no analysis code with round 2, and its full output is in [`audit_round2.json`](audit_round2.json). PRs #7–#9 were inspected but not changed, and nothing was merged.

## What was checked

| Section | Checks | Failed | Content |
|---|---:|---:|---|
| A. Contrasts | 44 | 0 | All 20 E1a contrasts, the 10 E3 contrasts, the 12 E4 contrasts, the E1a Holm adjustment and the E4 decision rule, recomputed from the per-episode CSVs with a different bootstrap seed. Point estimates are identical; interval bounds differ by at most 0.44 points |
| B. Seed level | 16 | 0 | Per-seed means and SDs, Welch p, Hedges' g, t-intervals (E1a, E4) |
| C. Baselines | 16 | 0 | Success rates and Wilson intervals (DWA, potential field, go-to-goal) |
| D. Scenario sets | 44 | **1** | Every episode seed belongs to a declared set; the DWA tuning scenarios are used nowhere else |
| E. Run records | 20 | **1** | Steps (301,056), overrides per arm, commits, modified-code flag, power state |
| F. Checkpoints | 50 | 0 | SHA-256 in the E4 records = committed copies; the E1a copies are identical to the run outputs |
| G. Determinism | 4 | 0 | 820 episodes replayed from the committed checkpoint copies are identical in outcome, steps and class: E4 300, E3 mean 220 (including DWA), E3 sampled 100, E1a fresh 200 |
| H. Quoted numbers | 29 | 0 | The E1a, E3 and E4 numbers in README.md, the manuscript and the experiment READMEs match the stored outputs |

## Findings

1. **Reproducibility bug (fixed on this branch).** The README and reproducibility checklist said the evaluations could be re-run from the committed checkpoint copies (`research/experiments/{e1a,round2}/checkpoints/<arm>/seed_<k>/best.pt`). Neither `run_eval2.py` nor `e1_evaluate.py` could load that layout.
   - Both now accept it as a last fallback. The existing layouts are tried first, so no stored result changes.
   - The replays in section G use this fix.
2. **E1a control seed 0 ran partly with uncommitted code.** Its second invocation (steps 40,960–196,608) has `code_modified_since_commit: true`. The round-2 opt-in code was being written in the same working tree at the time. The E1a README wrongly said that no record shows modified code.
   - **Check.** The run was retrained from scratch at commit `30b53d8`, with `safenav/` and `configs/` unmodified, in one invocation (`records/train_control_seed0.json` here). `best.pt` (151,552 steps), `final.pt` (301,056 steps) and `validation.csv` are **bit-identical** to the committed ones, so the uncommitted edits had no effect on this run.
   - The E1a README now has a dated correction.
3. **Power state.** The first invocation of that run (206 s, up to 36,864 steps) was on battery, not on mains power as the E1a README says. This affects only throughput, not results (training is deterministic). It is corrected in the same note.
4. **Round 1, minor.** The heading-escape tuning set (L2, seeds 1,200,000–1,200,099) shares 20 scenarios with the trainer's expert validation set (1,200,000–1,200,019), which also selected the checkpoints. Both are validation uses, so there is no leakage into any test set. This is recorded, and nothing is changed.
5. **Statistical reporting.**
   - The bootstrap p-values are two-sided percentile p-values from the bootstrap distribution. This is an approximation, not an exact test. A stored value of 0 means p < 1/10,000, and the docs report it as "< 0.001".
   - With 5 seeds, the seed-level tests have low power. The docs say so.
   - Nothing is changed.

No other inconsistency was found between the manuscript, the README, the experiment reports and the raw outputs for the quoted round-2 numbers.
