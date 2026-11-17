# Exploratory follow-up: a DWA whose own margin respects the filter margin (post hoc)

**Status: exploratory, added after the pre-registered DWA results were seen. Descriptive only; no confirmatory claim rests on it.** This note was committed before the runs it describes.

## Why

The frozen DWA (selected configuration 2, `clearance_min = 0.05` m) did this on the 300 committed test scenarios:
- **filter on:** 48 % expert success, and every timeout was filter-blocked (Type I);
- **filter off:** 96 % success with no collisions, but a minimum obstacle clearance of only 0.027 m.

The filter keeps `safety.safe_distance = 0.3` m. So the planner steers into a band the filter forbids, and it then stalls in front of the filter. All 12 tuning configurations used `clearance_min ≤ 0.15` m, below the filter margin, which may explain why the tuning grid hardly mattered (37–39 % success).

**Question:** does the filter-induced deadlock of DWA disappear when the planner's own admissibility margin is at least the filter margin? If it does, the Type I stalls of the classical planner are a margin mismatch. If it does not, they are a property of the filter geometry (heading-dependent admissibility) that a margin alone does not fix.

## Design

- **Variants:** the frozen DWA weights (`w_heading = 0.2`, `w_clearance = 0.5`) with `clearance_min ∈ {0.35, 0.45}` m, that is the filter margin plus 0.05 or 0.15 m (the same offsets as the tuning grid). No other change and no further tuning.
- **Scenario sets, all with the filter on:**
  - the 300 committed test scenarios (beginner, intermediate, expert);
  - the 500 fresh expert scenarios (seed base 9,000,000);
  - the 1,000 E3 expert scenarios (seed base 7,600,000, rejection generator).
- **Reporting:**
  - success, collision, timeout and the stall taxonomy, with Wilson intervals;
  - descriptive comparison with the frozen DWA on the same scenarios.

## Commands

```bash
for m in 0.35 0.45; do t=${m/./p}
  for s in "beginner 2000000" "intermediate 2100000" "expert 2200000"; do set -- $s
    python research/experiments/round2/run_eval2.py --tag dwa_margin_${t}_test_on --baseline dwa \
      --dwa w_heading=0.2 --dwa w_clearance=0.5 --dwa clearance_min=$m --stages $1 --seed-base $2 --n 100
  done
  python research/experiments/round2/run_eval2.py --tag dwa_margin_${t}_fresh_on --baseline dwa \
    --dwa w_heading=0.2 --dwa w_clearance=0.5 --dwa clearance_min=$m --stages expert --seed-base 9000000 --n 500
  python research/experiments/round2/run_eval2.py --tag dwa_margin_${t}_e3_expert --baseline dwa \
    --dwa w_heading=0.2 --dwa w_clearance=0.5 --dwa clearance_min=$m --stages expert --seed-base 7600000 \
    --n 1000 --set env.goal_sampling=rejection
done
```
