#!/bin/bash
# X9 learned up-front router against model selection and sub-ladders (and the oracle router).
# The pooled run needs more than the login node's memory: run this as a CPU job.
# RESULTS.md: "X9 learned up-front router".
source "$(dirname "$0")/common.sh"
A="--score agreement_selfconf --abstain-factor 64 $ALPHAS --subsets --tag=-k64"
cl route --benchmark humaneval mbpp --tiers t1 t2 t3 t4 $A
cl route --benchmark livecodebench --tiers t1 t2 t3 t4 $A
cl route --benchmark humaneval mbpp livecodebench --tiers t1 t2 t3 t4 $A
cl route --benchmark humaneval mbpp --tiers t1 t2 t3 t4 api $A
echo ALLDONE
