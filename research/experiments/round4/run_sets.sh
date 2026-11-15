#!/usr/bin/env bash
# run_sets.sh — run round-4 evaluation sets to completion (resumable; one process at a time).
#   bash research/experiments/round4/run_sets.sh pilot        # exploratory pilots (seeds 4,3x0,000+)
#   bash research/experiments/round4/run_sets.sh latency      # Part P (normal priority)
#   bash research/experiments/round4/run_sets.sh confirmatory # Part R + F (seeds 4,4x0,000+)
# Waits (does not run) while the laptop is on battery below 35 %.
set -u
cd "$(dirname "$0")/../../.."
R=research/experiments/round4/run_round4.py
AG="--agent filtered=results/checkpoints/filtered --agent unfiltered=results/checkpoints/unfiltered"
REJ="--set env.goal_sampling=rejection"
ALL="nofilter baseline A C C_loc C_map C_sensed planner_only path_follower"

battery_ok() { python -c "import psutil;b=psutil.sensors_battery();print(int(b is None or b.power_plugged or b.percent>=35))"; }

run() {  # tag expected_rows args...
  local tag=$1 expected=$2; shift 2
  for i in $(seq 1 60); do
    local n=0
    [ -f research/experiments/round4/results/$tag/episodes.csv ] && n=$(($(wc -l < research/experiments/round4/results/$tag/episodes.csv)-1))
    [ "$n" -ge "$expected" ] && { echo "$tag complete ($n)"; return 0; }
    while [ "$(battery_ok)" = "0" ]; do echo "battery low; waiting"; sleep 600; done
    python $R --tag "$tag" "$@" --deadline-s 1500 | tail -1
  done
  echo "$tag NOT complete"; return 1
}

case "${1:-}" in
  pilot)
    run p_expert $((10*30*7 + 2*30)) --methods $ALL $AG --stages expert --seed-base 4300000 --n 30 $REJ --record-calls
    run p_intermediate $((10*30*2)) --methods baseline C $AG --stages intermediate --seed-base 4310000 --n 30 $REJ
    run p_dense $((10*30*2)) --methods baseline C $AG --stages expert --seed-base 4320000 --n 30 $REJ --task min_obstacles=15 --task max_obstacles=25
    run p_heading $((10*30*2)) --methods baseline C $AG --stages expert --seed-base 4330000 --n 30 $REJ --heading-perturb
    ;;
  latency)
    run latency $((10*100*4)) --methods A:reference A:fast C:reference C:fast $AG --stages expert --seed-base 4200000 --n 100 $REJ --record-calls --priority normal
    ;;
  confirmatory)
    run r_expert $((10*200*7 + 2*200)) --methods $ALL $AG --stages expert --seed-base 4400000 --n 200 $REJ --record-calls
    run r_intermediate $((10*200*2)) --methods baseline C $AG --stages intermediate --seed-base 4410000 --n 200 $REJ
    run r_dense $((10*200*2)) --methods baseline C $AG --stages expert --seed-base 4420000 --n 200 $REJ --task min_obstacles=15 --task max_obstacles=25
    run r_heading $((10*200*2)) --methods baseline C $AG --stages expert --seed-base 4430000 --n 200 $REJ --heading-perturb
    ;;
  *) echo "usage: $0 pilot|latency|confirmatory"; exit 2 ;;
esac
