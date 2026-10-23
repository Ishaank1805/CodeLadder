#!/bin/bash
# E1 to E3 with the four Qwen2.5-Coder tiers: each model alone, model selection (singles),
# sub-ladders (subsets) and two-level ladders; abstention price κ = 4 and 64; lower-tier draw
# budget 1 to 3; Pareto Testing; pooled benchmarks with transfer (X6) and learning curves (X5).
# RESULTS.md: "LiveCodeBench with all four Qwen tiers", "ladder design, lower-tier draw budget",
# "X8 Pareto Testing at scale", "32B tier and four-tier ladders", "sub-ladders ...".
source "$(dirname "$0")/common.sh"
S="--score agreement_selfconf"
for k in 4 64; do
  for f in singles subsets two_level; do
    ev --benchmark humaneval mbpp --tiers t1 t2 t3 t4 $S --family $f --abstain-factor $k --tag=-k$k
    ev --benchmark livecodebench --tiers t1 t2 t3 t4 $S --family $f --abstain-factor $k --tag=-k$k
    ev --benchmark livecodebench --tiers t1 t3 t4 $S --family $f --abstain-factor $k --tag=-k$k
    ev --benchmark humaneval mbpp livecodebench --tiers t1 t2 t3 t4 $S --family $f --abstain-factor $k \
      --tag=-k$k --calib-sizes 200 400 600 --calib-on humaneval mbpp
  done
done
# Each model alone: the calibrated single-model baselines stored in every run above.
for b in 1 2 3; do
  for f in subsets two_level; do
    ev --benchmark humaneval mbpp --tiers t1 t2 t3 t4 $S --family $f --abstain-factor 64 --lower-budget $b --tag=-k64-lb$b
    ev --benchmark livecodebench --tiers t1 t2 t3 t4 $S --family $f --abstain-factor 64 --lower-budget $b --tag=-k64-lb$b
    ev --benchmark livecodebench --tiers t1 t3 t4 $S --family $f --abstain-factor 64 --lower-budget $b --tag=-k64-lb$b
  done
done
for f in singles subsets two_level; do
  for B in livecodebench "humaneval mbpp livecodebench"; do
    ev --benchmark $B --tiers t1 t2 t3 t4 $S --family $f --abstain-factor 64 --lower-budget 1 --method pareto --tag=-k64-lb1-pareto
  done
  ev --benchmark humaneval mbpp livecodebench --tiers t1 t2 t3 t4 $S --family $f --abstain-factor 64 --lower-budget 1 --tag=-k64-lb1
done
echo ALLDONE
