#!/bin/bash
# Run an evaluation or analysis script on a compute node (no GPU). The login node caps memory
# at 2 GB per user, which large replays (LiveCodeBench with many tiers, pooled benchmarks)
# exceed. The script runs from the repository root; its output goes to the job log.
#
#   sbatch scripts/ada/cpu_job.sbatch scripts/some_batch.sh
#   sbatch -c 4 --mem=16G --dependency=afterany:12345 scripts/ada/cpu_job.sbatch scripts/x.sh
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cl-cpu
#SBATCH --output=logs/cl-cpu-%j.out

set -euo pipefail
source "$HOME/codeladder/scripts/ada/common.sh"
activate_venv
cd "$CL_REPO"
echo "[cpu_job] $(date +%F' '%H:%M) $(hostname): $*"
"$@"
echo "[cpu_job] $(date +%F' '%H:%M) done"
