#!/bin/bash
# E4 repair versus resample (per-step statistics), E7 repair-depth ablation and X4 learned
# repair-or-resample policy. RESULTS.md: "E4 repair versus resample, all four tiers",
# "E7 ablation, repair depth", "X4 learned repair-or-resample policy".
source "$(dirname "$0")/common.sh"
py scripts/analysis_repair.py --benchmarks humaneval mbpp --json results/analysis/e4_repair_evalplus.json
py scripts/analysis_repair.py --benchmarks livecodebench --json results/analysis/e4_repair_lcb.json
for d in 2 3; do
  for a in repair adaptive; do
    for tiers in t2 t3 "t1 t2 t3"; do
      ev --benchmark humaneval mbpp --tiers $tiers --action $a --score agreement --max-repair-depth $d --tag=-depth$d
    done
  done
done
A="--score agreement_selfconf --abstain-factor 64 $ALPHAS --tag=-k64"
cl actions --benchmark humaneval mbpp --tiers t1 t2 t3 t4 $A
cl actions --benchmark livecodebench --tiers t1 t2 t3 t4 $A
cl actions --benchmark humaneval mbpp --tiers t4 --family shared $A
cl actions --benchmark livecodebench --tiers t2 --family shared $A
echo ALLDONE
