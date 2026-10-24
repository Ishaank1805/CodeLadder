#!/bin/bash
# X17: oracle sensitivity. Run every distinct visible-passing program on all hidden tests
# (data/oracle/<benchmark>/<tier>.json.gz; a compute-node CPU job), then calibrate on labels
# from a random 0 / 10 / 25 / 50 / 100% of each task's hidden tests and measure the true risk.
# RESULTS.md: "X17 oracle sensitivity".
source "$(dirname "$0")/common.sh"
for B in humaneval mbpp; do
  for t in t1 t2 t3 t4; do cl hidden-outcomes --tier $t --benchmark $B --workers 12; done
done
A="--score agreement_selfconf --abstain-factor 64 $ALPHAS --tag=-k64"
for t in t1 t4; do cl oracle-shift --benchmark humaneval mbpp --tiers $t $A; done
for f in singles subsets; do cl oracle-shift --benchmark humaneval mbpp --tiers t1 t2 t3 t4 --family $f $A; done
echo ALLDONE
