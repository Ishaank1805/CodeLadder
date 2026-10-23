#!/bin/bash
# Price of abstaining κ when choosing among certified settings (κ = 0 is the cheapest-setting
# rule, degenerate once cheap sub-ladders exist). RESULTS.md: "sub-ladders, calibrated model
# selection and the price of abstaining", "first LiveCodeBench results".
source "$(dirname "$0")/common.sh"
B="--benchmark humaneval mbpp"
for k in 0 1 2 4; do
  ev $B --tiers t1 t3 t4 --score agreement_selfconf --family singles --abstain-factor $k --tag=-k$k
  ev $B --tiers t1 t2 t3 t4 --score agreement --family singles --abstain-factor $k --tag=-k$k
done
for k in 0 0.5 1 2 4; do
  ev $B --tiers t1 t3 t4 --score agreement_selfconf --family subsets --abstain-factor $k --tag=-k$k
  ev $B --tiers t1 t2 t3 t4 --score agreement --family subsets --abstain-factor $k --tag=-k$k
done
for k in 0 0.5 1 2; do
  ev $B --tiers t1 t3 t4 --score agreement_selfconf --family two_level --abstain-factor $k --tag=-k$k
done
for k in 0 1 16 64; do
  for f in singles subsets two_level; do
    ev --benchmark livecodebench --tiers t1 t2 --score agreement_selfconf --family $f --abstain-factor $k --tag=-k$k
    ev --benchmark humaneval mbpp livecodebench --tiers t1 t2 --score agreement_selfconf --family $f --abstain-factor $k --tag=-k$k
  done
done
echo ALLDONE
