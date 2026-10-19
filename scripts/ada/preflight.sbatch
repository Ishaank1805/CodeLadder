#!/bin/bash
# CPU-only preflight on Ada: our labels vs the official EvalPlus evaluator.
#   sbatch scripts/ada/preflight.sbatch
# Results: results/preflight/{humaneval,mbpp}_preflight.json
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -c 20
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=02:00:00
#SBATCH --job-name=cl-preflight
#SBATCH --output=%x-%j.out

set -euo pipefail
source "$HOME/codeladder/scripts/ada/common.sh"
activate_venv
cd "$CL_REPO"
for b in humaneval mbpp; do
  python scripts/preflight_evalplus.py "$b" --out results/preflight --workers 12
done
