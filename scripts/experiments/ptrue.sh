#!/bin/bash
# X12: P(True) self-evaluation (probability of "Yes" to "Is this solution correct?") as the
# confidence signal, against the verbalised rating. Needs annotate-ptrue on every tier.
# RESULTS.md: "X12 P(True)".
source "$(dirname "$0")/common.sh"
specs=""
for b in humaneval mbpp livecodebench; do for t in t1 t2 t3 t4 d1 d2 d3; do specs="$specs $b/$t"; done; done
py scripts/show_results.py ratings $specs
for B in "humaneval mbpp" livecodebench; do
  for s in agreement_ptrue ptrue; do
    for t in t1 t2 t3 t4 d1 d2 d3; do ev --benchmark $B --tiers $t --score $s --abstain-factor 64 --tag=-k64; done
  done
  for f in singles subsets two_level; do
    ev --benchmark $B --tiers t1 t2 t3 t4 --score agreement_ptrue --family $f --abstain-factor 64 --lower-budget 1 --tag=-k64-lb1
  done
  ev --benchmark $B --tiers t1 t2 t3 t4 --score agreement_ptrue --family singles --abstain-factor 64 --tag=-k64
  for pair in "t1 t4" "t2 t4" "t3 t4"; do
    cl economics --benchmark $B --tiers $pair --score agreement_ptrue --abstain-factor 64 $ALPHAS \
      --ratios 1 2 4 8 16 32 64 128 256 512 1024 --lower-budget 1 --tag=-k64-lb1
  done
done
echo ALLDONE
