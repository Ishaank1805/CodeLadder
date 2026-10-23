#!/bin/bash
# Score ablation: agreement, self-confidence, log-probability and agreement x self-confidence
# for single models, and agreement-only versions of each model at κ = 64.
# RESULTS.md: "14B tier, confidence signals", "first LiveCodeBench results",
# "X1 / X10 corrected" (agreement-only runs).
source "$(dirname "$0")/common.sh"
for t in t2 t3 t4; do ev --benchmark humaneval mbpp --tiers $t --score agreement_selfconf; done
for t in t3 t4; do ev --benchmark humaneval mbpp --tiers $t --score agreement; done
for s in agreement selfconf logprob agreement_selfconf; do
  ev --benchmark livecodebench --tiers t2 --score $s --abstain-factor 4 --tag=-k4
done
for B in "humaneval mbpp" livecodebench; do
  for t in t1 t2 t3 t4 d1 d2 d3; do ev --benchmark $B --tiers $t --score agreement --abstain-factor 64 --tag=-k64-agr; done
done
echo ALLDONE
