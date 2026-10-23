#!/bin/bash
# Published baselines (Model Cascading for Code, Bayesian self-escalation) on the same splits,
# from the stored self-tests and greedy traces. RESULTS.md: "published baselines against CodeLadder".
source "$(dirname "$0")/common.sh"
for tiers in "t2 t3" "t3 t4" "t1 t4" "t1 t2 t3 t4"; do
  for B in "humaneval mbpp" livecodebench; do cl baselines --benchmark $B --tiers $tiers $ALPHAS; done
done
cl baselines --benchmark humaneval mbpp livecodebench --tiers t2 t3 $ALPHAS
echo ALLDONE
