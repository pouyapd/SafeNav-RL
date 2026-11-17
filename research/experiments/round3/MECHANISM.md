# Round 3, Phase 2: what do the two stall types have in common? (exploratory)

The script is [`mechanism.py`](mechanism.py). Hypotheses M1–M4 and all feature definitions are in its docstring, which was committed in `4b50b0f` before the run.
- **Data:** replays of scenarios that round 2 had already evaluated. This is exploratory, not a fresh test.
  - E3 expert, corrected generator: the first 300 scenarios × the 10 committed agents.
  - E4: the first 200 scenarios × control, risk-free and Beta (5 seeds each).
- **Replays:** all 6,000 are identical to the stored round-2 rows (outcome, steps, class).
- **Units:**
  - 2,937 stalls: 1,910 filter-blocked, 1,026 self-stop, 1 other;
  - 1,585 close passages: successful episodes whose smallest obstacle barrier was below 0.30 m.
- **Intervals:** 95 % bootstrap over scenarios.
- **Outputs:** [`mechanism/summary.json`](mechanism/summary.json), [`mechanism/states.csv`](mechanism/states.csv) and [`mechanism/episodes.csv`](mechanism/episodes.csv).

## Results

| Feature (share) | All stalls | Filter-blocked | Self-stop | Close passages |
|---|---:|---:|---:|---:|
| Goal reachable through the safe set (oracle) | 100.0 % [99.9, 100] | 99.9 % | 100 % | 99.9 % |
| Shortest safe path starts > 45° away from the goal direction (**M1**) | **58.5 % [54.2, 62.5]** | 63.1 % [58.8, 67.4] | 49.9 % [45.2, 54.6] | **10.9 % [8.8, 13.2]** |
| One 4 s manoeuvre gains ≥ 1 m of geodesic cost-to-go (**M2**) | **46.5 % [43.6, 49.3]** | 38.2 % [34.7, 41.7] | 62.0 % [58.4, 65.4] | 95.6 % [94.3, 96.8] |
| One 4 s manoeuvre gains ≥ 0.5 m of straight-line distance | 49.1 % [46.1, 52.1] | 34.9 % | 75.5 % | 97.0 % |
| Straight-line distance locally minimal under admissible 0.3 m moves (**M3**) | **17.7 % [15.4, 20.0]** | 26.5 % [23.3, 30.0] | 1.4 % [0.7, 2.1] | **0.4 % [0.1, 0.8]** |
| Nearest obstacle within 45° of the goal direction | 52.5 % | 42.3 % | 67.8 % | 20.3 % |

| Median | Filter-blocked | Self-stop | Close passages |
|---|---:|---:|---:|
| Barrier to the nearest obstacle (m) | 0.00 | 0.45 | 0.10 |
| Path deviation from the goal direction (°) | 63 | 45 | 17 |
| Detour ratio (geodesic / straight line) | 1.27 | 1.20 | 1.07 |
| Share of range beams > 1 m | 0.81 | 1.00 | 0.88 |
| Filter interventions in the last 100 steps | 100 % | 0 % | – |
| Turn-rate reversals in the last 100 steps | 58 | 55 | – |

## Hypotheses (as stated before the run)

| | Statement | Outcome |
|---|---|---|
| M1 | Stalls need a deviation > 45° more often than close passages | **supported**: 58.5 % vs. 10.9 %. It holds for both types |
| M2 | ≥ 80 % of stalls have a single manoeuvre with ≥ 1 m geodesic gain | **not supported**: 46.5 %. Filter-blocked 38 %, self-stop 62 % |
| M3 | The straight-line distance is locally minimal in *most* stalls, and much more often than in passages | **partly**: much more often than in passages (17.7 % vs. 0.4 %), but not in most stalls. Almost never in self-stops (1.4 %) |
| M4 | The two types share these conditions (within 15 points) | **not supported**: differences of 13 (M1), 24 (M2) and 25 (M3) points |

## Reading (exploratory)

- **What the stall types share.**
  - The goal is reachable in every stall, so no stall is a dead end of the map.
  - In most stalls the shortest safe path starts well away from the goal direction: 63 % of filter-blocked stalls and 50 % of self-stops, against 11 % of close passages.
  - Both types show many turn-rate reversals (about 55 in 100 steps). The policy dithers in place instead of committing to a detour.
- **Where they differ.**
  - Filter-blocked stalls sit at the boundary (barrier 0), and a quarter of them are local minima of the straight-line distance.
  - Self-stops happen 0.45 m before the boundary in open space. In 98.6 % of them a straight move towards the goal was still admissible.
  - So a self-stop is not a geometric trap for a myopic controller. The policy stops although progress is possible, typically with the obstacle roughly in front of the goal (68 % within 45°).
- **Escape needs more than one short manoeuvre.** A single 4 s manoeuvre from the stall state recovers ≥ 1 m of geodesic cost in only 38 % of filter-blocked stalls. Near the boundary the filter slows every motion that is not tangential, and the robot first has to turn.
  - With a 0.5 m threshold, chosen post hoc as a sensitivity check, the shares are 81 % (filter-blocked) and 96 % (self-stop).
  - Recovery therefore has to be multi-step or receding-horizon, not one-shot. This is what methods A and C do.
- **Status of the evidence.** These are associations in replayed data, not interventions. The "missing manoeuvre" reading is supported for the *deviation* part (M1), not for "one manoeuvre suffices" (M2). The two types are geometrically different, even though they share the need for a detour.
