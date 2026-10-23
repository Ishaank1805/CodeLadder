#!/bin/bash
# X13: when do ladders pay? Sweep the top tier's price per token (relative to the cheapest
# tier) for two-tier ladders and compare calibrated sub-ladders with model selection; the
# break-even ratio is predicted from the ladder's cheap-tier share and measured on the sweep.
# Real ratios (params_b): 1.5B->32B 21x, 7B->32B 4.6x, 14B->32B 2.3x, 1.5B->14B 9.3x,
# 1.3B->32B 25x, 1.5B->120B 80x, 14B->120B 8.6x. RESULTS.md: "X13 when do ladders pay".
source "$(dirname "$0")/common.sh"
A="--score agreement_selfconf --abstain-factor 64 $ALPHAS --ratios 1 2 4 8 16 32 64 128 256 512 1024"
for pair in "t1 t4" "t2 t4" "t3 t4" "t1 t3" "d1 t4"; do
  for B in "humaneval mbpp" livecodebench; do
    cl economics --benchmark $B --tiers $pair $A --lower-budget 1 --tag=-k64-lb1
  done
done
for pair in "t1 api" "t3 api"; do cl economics --benchmark humaneval mbpp --tiers $pair $A --lower-budget 1 --tag=-k64-lb1; done
cl economics --benchmark humaneval mbpp --tiers t1 t4 $A --tag=-k64  # full cheap-tier budget
echo ALLDONE
