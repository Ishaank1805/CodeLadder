#!/bin/bash
# Every batch in order (several hours of CPU; on Ada: sbatch -c 16 --mem=64G
# scripts/ada/cpu_job.sbatch scripts/experiments/all.sh), then the report figures.
source "$(dirname "$0")/common.sh"
for b in qwen_families kappa_sweep calibration_data difficulty energy scores deepseek api baselines repair router signals; do
  echo "##### $b"
  bash "scripts/experiments/$b.sh"
done
py scripts/figures/report_figures.py results
echo ALLDONE
