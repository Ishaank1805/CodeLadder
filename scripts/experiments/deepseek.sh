#!/bin/bash
# X1 / X10: DeepSeek-Coder tiers alone, as a family, cross-family ladders with Qwen, and model
# selection over all seven local models; each with agreement x self-confidence and with
# agreement alone. Needs DeepSeek ratings made with the prefilled answer (conf_prefill).
# RESULTS.md: "X1 / X10 corrected".
source "$(dirname "$0")/common.sh"
py scripts/show_results.py ratings humaneval/d1 humaneval/d2 humaneval/d3 mbpp/d1 mbpp/d2 mbpp/d3 \
  livecodebench/d1 livecodebench/d2 livecodebench/d3 humaneval/t3
for s in agreement_selfconf agreement; do
  sfx=""; [ "$s" = agreement ] && sfx="-agr"
  for B in "humaneval mbpp" livecodebench; do
    for t in d1 d2 d3; do ev --benchmark $B --tiers $t --score $s --abstain-factor 64 --tag=-k64$sfx; done
    ev --benchmark $B --tiers d1 d2 d3 --score $s --family singles --abstain-factor 64 --tag=-k64$sfx
    for f in subsets two_level; do
      ev --benchmark $B --tiers d1 d2 d3 --score $s --family $f --abstain-factor 64 --lower-budget 1 --tag=-k64-lb1$sfx
    done
    for tiers in "t1 d3" "d1 t4" "t2 d3" "d1 d2 t4" "t1 t2 d3" "d1 t3 t4"; do
      for f in singles subsets two_level; do
        ev --benchmark $B --tiers $tiers --score $s --family $f --abstain-factor 64 --lower-budget 1 --tag=-k64-lb1$sfx
      done
    done
    ev --benchmark $B --tiers t1 d1 t2 d2 t3 t4 d3 --score $s --family singles --abstain-factor 64 --tag=-k64-all7$sfx
  done
done
echo ALLDONE
