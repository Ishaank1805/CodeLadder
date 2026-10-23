#!/bin/bash
# X5 learning curves (fewer calibration tasks), E5 temporal shift on LiveCodeBench (calibrate
# once on the oldest window against sliding recalibration), X6 transfer (calibrate on
# HumanEval+/MBPP+, test on LiveCodeBench; also in qwen_families.sh).
# RESULTS.md: "X5 calibration learning curve", "first LiveCodeBench results, temporal shift, transfer".
source "$(dirname "$0")/common.sh"
for f in singles subsets two_level; do
  ev --benchmark humaneval mbpp --tiers t1 t3 t4 --score agreement_selfconf --family $f --abstain-factor 4 \
    --calib-sizes 50 100 150 200 --tag=-k4-lc
  ev --benchmark livecodebench --tiers t1 t2 --score agreement_selfconf --family $f --abstain-factor 4 \
    --calib-sizes 100 200 300 --tag=-k4
  ev --benchmark humaneval mbpp livecodebench --tiers t1 t2 --score agreement_selfconf --family $f --abstain-factor 4 \
    --calib-sizes 100 200 300 500 --calib-on humaneval mbpp --tag=-k4
done
for tiers in "t1 t2" "t1 t2 t3" "t1 t2 t3 t4"; do
  ev --benchmark livecodebench --tiers $tiers --score agreement_selfconf --family shared --shift 5 --tag=-shift
done
echo ALLDONE
