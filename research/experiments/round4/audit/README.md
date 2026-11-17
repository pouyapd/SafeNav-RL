# Audit of round 3 (PRs #10–#12), before round 4

The audit script is [`../audit_round3.py`](../audit_round3.py). It shares no code with `analyze_round3.py`, and its full output is in [`audit_round3.json`](audit_round3.json). PRs #10–#12 were inspected and not changed, and nothing was merged.

**Result: 68 checks, 0 failed.**

| Section | Checks | Content |
|---|---:|---|
| A. Contrasts | 15 | The 4 primary and 8 secondary contrasts recomputed from `results/main/episodes.csv` with another bootstrap seed. Point estimates are identical, and interval bounds are within 1.5 points. Holm adjustment; claim rule for both agent types |
| B. Seed level | 12 | Per-seed rates, paired t-test p, d_z. The direction is the same in every seed for every contrast |
| C. Quoted numbers | 30 | Every row of the recovery table in the manuscript (Table 6), the README and PROTOCOL.md (success ± SD, deadlock ± SD, moving timeouts), recomputed from the raw rows |
| D. Raw data | 6 | 25,000 unique rows; a balanced design (5 methods × 10 agents × 500); seeds 8,500,000–8,500,499 only; 0 collisions and minimum clearance 0.300 m; baseline rows have no recovery activity; the run record covers the end of the run |
| E. The 776 rows without a run record | 1 | They are the first 776 rows: filter-trained seed 0, baseline (500) and A (276). All 776 were **replayed with the current code and are identical** in every non-timing column |
| F. Replay sample | 1 | 200 further episodes (agents trained without the filter, seeds 1 and 3, scenarios 200–219, all 5 methods): identical |
| G. Pairing | 3 | For B, C and C_euclid, every episode without an activation is identical to the baseline episode (1,965 each) |

## Other observations (not failures)

- **Safety-filter interaction.** The code was checked against the tests: every recovery mode only proposes actions, and `SafetyFilterWrapper` checks every one. `tests/test_recovery.py` asserts that the executed action equals the filter applied to the proposed one. The raw data confirm 0 collisions and minimum clearance 0.300 m in all 25,000 episodes.
- **Reproduction and checkpoint loading.** The round-3 commands (`run_recovery.py`, `analyze_round3.py`) ran unchanged for the replays above. The committed checkpoints (`results/checkpoints/<variant>/seed_<k>.pt`) load directly.
- **Timing instrumentation (a limitation of round 3, not an error in its data).**
  - The round-3 budget figure (p99 = 189 ms) is the 99th percentile over episodes of each episode's *longest* call. It is not the 99th percentile over calls. It was measured partly while another analysis process ran on the same laptop.
  - Round 3 recorded no per-call times and no CPU time, so slow calls caused by computation could not be told apart from slow calls caused by the operating system.
  - Round 4 records every call, with wall and CPU time (Part P).
