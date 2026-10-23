#!/bin/bash
# E6: cost measured as GPU energy (busy seconds only) instead of weighted tokens.
# RESULTS.md: "E6 energy cost of reliability".
source "$(dirname "$0")/common.sh"
py scripts/recompute_energy.py
S="--score agreement_selfconf --abstain-factor 64 --cost-metric energy_j"
for B in "humaneval mbpp" livecodebench; do
  for f in singles subsets two_level; do
    ev --benchmark $B --tiers t1 t2 t3 t4 $S --family $f --tag=-k64-energy
    ev --benchmark $B --tiers t1 t2 t3 t4 $S --family $f --lower-budget 1 --tag=-k64-lb1-energy
  done
  for t in t1 t2 t3 t4; do ev --benchmark $B --tiers $t $S --tag=-k64-energy; done
done
echo ALLDONE
