#!/bin/bash
# X14: paired task-level bootstrap intervals for the headline comparisons (A minus B;
# solved-share difference and cost ratio). Writes results/ci/<name>.json; print them with
# python scripts/show_results.py ci results/ci/*.json. RESULTS.md: "X14 confidence intervals".
source "$(dirname "$0")/common.sh"
Q=resample-agreement_selfconf
pair() { cl compare "$2" "$3" $ALPHAS --out "results/ci/$1.json"; }
pair ladders-vs-selection-evalplus humaneval+mbpp-t1-t2-t3-t4-$Q-subsets-k64-lb1 humaneval+mbpp-t1-t2-t3-t4-$Q-singles-k64
pair ladders-vs-selection-lcb livecodebench-t1-t2-t3-t4-$Q-subsets-k64-lb1 livecodebench-t1-t2-t3-t4-$Q-singles-k64
pair twolevel-vs-selection-lcb livecodebench-t1-t2-t3-t4-$Q-two_level-k64-lb1 livecodebench-t1-t2-t3-t4-$Q-singles-k64
pair api-ladders-vs-selection-evalplus humaneval+mbpp-t1-t2-t3-t4-api-$Q-subsets-k64-lb1 humaneval+mbpp-t1-t2-t3-t4-api-$Q-singles-k64-lb1
pair selfconf-vs-agreement-32b humaneval+mbpp-t4-$Q-shared humaneval+mbpp-t4-resample-agreement-shared
pair all7-vs-qwen-selection-evalplus humaneval+mbpp-t1-d1-t2-d2-t3-t4-d3-$Q-singles-k64-all7 humaneval+mbpp-t1-t2-t3-t4-$Q-singles-k64
pair all7-vs-qwen-selection-lcb livecodebench-t1-d1-t2-d2-t3-t4-d3-$Q-singles-k64-all7 livecodebench-t1-t2-t3-t4-$Q-singles-k64
pair pareto-vs-paths-lcb livecodebench-t1-t2-t3-t4-$Q-subsets-k64-lb1-pareto livecodebench-t1-t2-t3-t4-$Q-subsets-k64-lb1
pair lean-vs-full-budget-lcb livecodebench-t1-t2-t3-t4-$Q-subsets-k64-lb1 livecodebench-t1-t2-t3-t4-$Q-subsets-k64
echo ALLDONE
