#!/bin/bash
# E8 / X3 / X11: the free API top tier (gpt-oss-120b on Groq) alone and as the top rung of
# ladders. LiveCodeBench runs use whatever API pools exist (load_pools keeps the tasks with
# pools for every tier, so the 400-problem subset is evaluated until all 1,054 exist).
# RESULTS.md: "E8 / X3 free API top tier".
source "$(dirname "$0")/common.sh"
for B in "humaneval mbpp" livecodebench; do
  for s in agreement_selfconf agreement selfconf; do ev --benchmark $B --tiers api --score $s --abstain-factor 64 --tag=-k64; done
  for tiers in "t1 t2 t3 t4 api" "t1 t3 api" "t2 api" "t3 api"; do
    for f in singles subsets two_level; do
      ev --benchmark $B --tiers $tiers --score agreement_selfconf --family $f --abstain-factor 64 --lower-budget 1 --tag=-k64-lb1
    done
  done
done
echo ALLDONE
