#!/bin/bash
# X2 stdin signals (stress inputs) per tier, silent-failure types, confidence-rating AUROC,
# and the guarantee check over every run. RESULTS.md: "X2 stress inputs", "silent-failure
# types", summary finding 1.
source "$(dirname "$0")/common.sh"
py scripts/analysis_stdin_signals.py --tiers t1 t2 t3 t4 d1 d2 d3 --json results/analysis/x2_stdin_signals.json
for B in livecodebench mbpp; do
  short=$B; [ "$B" = livecodebench ] && short=lcb
  for t in t1 t2 t3 t4; do
    py scripts/analysis_silent_failures.py --benchmark $B --tiers $t --json results/analysis/silent_failures_${short}_$t.json
  done
done
py scripts/show_results.py ratings humaneval/t3 humaneval/d1 humaneval/d2 humaneval/d3
py scripts/count_validity.py results
echo ALLDONE
