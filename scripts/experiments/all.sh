#!/bin/bash
# Every batch in order (several hours of CPU; on Ada: sbatch -c 16 --mem=64G
# scripts/ada/cpu_job.sbatch scripts/experiments/all.sh), then the report figures and tables.
# bigcodebench.sh needs the BigCodeBench pools.
source "$(dirname "$0")/common.sh"
for b in qwen_families kappa_sweep calibration_data difficulty energy scores deepseek api baselines repair router economics bootstrap ptrue guaranteed oracle signals bigcodebench; do
  echo "##### $b"
  bash "scripts/experiments/$b.sh" || FAILS=$((FAILS + 1))
done
py scripts/figures/report_figures.py results
py scripts/figures/diagram.py results
py scripts/report_tables.py results
echo ALLDONE
