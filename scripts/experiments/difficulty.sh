#!/bin/bash
# X7 difficulty-aware (Mondrian) calibration and the conditional risk of pooled calibration
# on LiveCodeBench easy / medium / hard. RESULTS.md: "X7 difficulty-aware calibration".
source "$(dirname "$0")/common.sh"
S="--score agreement_selfconf --abstain-factor 64 --group-by difficulty"
for f in singles subsets two_level; do
  ev --benchmark livecodebench --tiers t1 t2 t3 t4 $S --family $f --tag=-k64-diff
  ev --benchmark livecodebench --tiers t1 t2 t3 t4 $S --family $f --lower-budget 1 --tag=-k64-lb1-diff
done
for t in t2 t3 t4; do ev --benchmark livecodebench --tiers $t $S --tag=-k64-diff; done
echo ALLDONE
