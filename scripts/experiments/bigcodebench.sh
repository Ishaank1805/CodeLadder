#!/bin/bash
# X18: the core experiments on BigCodeBench (1,134 tasks, unit tests), once all seven local
# tiers have pools with ratings and P(True). Mirrors the Qwen families, DeepSeek, P(True), X13,
# X14, X16 and X9 runs on the other benchmarks; the API tier on the 400-task subset is in api.sh.
# Run as a CPU job: sbatch -c 8 --mem=40G scripts/ada/cpu_job.sbatch scripts/experiments/bigcodebench.sh
# RESULTS.md: "X18 BigCodeBench".
source "$(dirname "$0")/common.sh"
B=bigcodebench
K="--abstain-factor 64"
specs=""
for t in t1 t2 t3 t4 d1 d2 d3; do specs="$specs $B/$t"; done
# Labels of BigCodeBench/790 and /1028 from pools generated before the 27 Oct fix.
py scripts/relabel_bcb.py
py scripts/show_results.py ratings $specs --json results/analysis/signals_auroc_bcb.json

# Each model alone, with each confidence score.
for s in agreement_selfconf agreement_ptrue agreement; do
  for t in t1 t2 t3 t4 d1 d2 d3; do ev --benchmark $B --tiers $t --score $s $K --tag=-k64; done
done

# Calibrated families over the Qwen tiers (full and lean lower rungs), with both scores.
for s in agreement_selfconf agreement_ptrue; do
  ev --benchmark $B --tiers t1 t2 t3 t4 --score $s --family singles $K --tag=-k64
  for f in subsets two_level; do
    ev --benchmark $B --tiers t1 t2 t3 t4 --score $s --family $f $K --tag=-k64
    ev --benchmark $B --tiers t1 t2 t3 t4 --score $s --family $f $K --lower-budget 1 --tag=-k64-lb1
  done
done

# Second family and all seven models.
S="--score agreement_selfconf"
ev --benchmark $B --tiers d1 d2 d3 $S --family singles $K --tag=-k64
ev --benchmark $B --tiers d1 d2 d3 $S --family subsets $K --lower-budget 1 --tag=-k64-lb1
ev --benchmark $B --tiers t1 d1 t2 d2 t3 t4 d3 $S --family singles $K --tag=-k64-all7

# Conditional risk on the hard subset (BigCodeBench-Hard, 148 tasks) against the rest.
ev --benchmark $B --tiers t4 $S $K --group-by difficulty --tag=-k64-diff
ev --benchmark $B --tiers t1 t2 t3 t4 $S --family singles $K --group-by difficulty --tag=-k64-diff

# X16 guaranteed baselines, X13 price sweeps, X9 router.
cl guaranteed --benchmark $B --tiers t1 t2 t3 t4 $S $ALPHAS --tag=-k64
for pair in "t1 t4" "t2 t4" "t3 t4" "d1 t4"; do
  cl economics --benchmark $B --tiers $pair $S $K $ALPHAS \
    --ratios 1 2 4 8 16 32 64 128 256 512 1024 --lower-budget 1 --tag=-k64-lb1
done
cl route --benchmark $B --tiers t1 t2 t3 t4 $S $K $ALPHAS --subsets --tag=-k64

# X14 paired bootstrap intervals for the headline comparisons.
Q=resample-agreement_selfconf
pair() { cl compare "$2" "$3" $ALPHAS --out "results/ci/$1.json"; }
pair ladders-vs-selection-bcb $B-t1-t2-t3-t4-$Q-subsets-k64-lb1 $B-t1-t2-t3-t4-$Q-singles-k64
pair all7-vs-qwen-selection-bcb $B-t1-d1-t2-d2-t3-t4-d3-$Q-singles-k64-all7 $B-t1-t2-t3-t4-$Q-singles-k64
pair ptrue-vs-selfconf-selection-bcb $B-t1-t2-t3-t4-resample-agreement_ptrue-singles-k64 $B-t1-t2-t3-t4-$Q-singles-k64

py scripts/figures/report_figures.py results
py scripts/report_tables.py results
echo ALLDONE
