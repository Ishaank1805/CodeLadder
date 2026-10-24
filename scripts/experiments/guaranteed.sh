#!/bin/bash
# X16: guaranteed baselines on the same splits as CodeLadder: per-model Learn-then-Test
# abstention and a Trust-or-Escalate cascade (full and lean lower rungs). Compare with the
# CodeLadder runs of the same tiers (qwen_families.sh, api.sh). RESULTS.md: "X16".
source "$(dirname "$0")/common.sh"
A="--score agreement_selfconf $ALPHAS --tag=-k64"
for tiers in "t1 t2 t3 t4" "t1 t3 t4"; do
  for B in "humaneval mbpp" livecodebench "humaneval mbpp livecodebench"; do cl guaranteed --benchmark $B --tiers $tiers $A; done
done
cl guaranteed --benchmark humaneval mbpp --tiers t1 t2 t3 t4 api $A
echo ALLDONE
