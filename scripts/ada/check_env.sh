#!/bin/bash
# Day-one environment check. Run it twice:
#   1) on the login node:   bash scripts/ada/check_env.sh
#   2) on a GPU node:       srun -A research --qos=medium -p long --gres=gpu:1 -c 10 \
#                                --exclude=gnode[01-40] --time=00:20:00 bash scripts/ada/check_env.sh
# Paste the output into docs/ada_env.md so the team knows which serving stack to use.

set -u
source "$(dirname "$0")/common.sh"

hr() { printf '\n==== %s ====\n' "$1"; }

hr "host"
hostname; date; uname -r
hr "glibc (PyTorch wheels need >= 2.28; Ubuntu 18.04 has 2.27)"
ldd --version | head -1
hr "slurm account / qos"
sacctmgr -n show assoc user="$USER" format=Account,QOS,DefaultQOS 2>/dev/null | head -5

hr "GPUs and driver"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  nvidia-smi --query-gpu=name,compute_cap --format=csv 2>/dev/null || echo "(compute_cap query unsupported: old driver)"
  nvidia-smi | head -4 | tail -2
  echo "vLLM 0.6.x images need a CUDA 12 driver (roughly >= 525). Older -> use llama.cpp fallback."
else
  echo "no nvidia-smi here (login node?)"
fi

hr "modules"
load_modules
python3 --version
singularity --version 2>/dev/null || echo "singularity module not loaded"

hr "internet access (needed for downloads; if missing on GPU nodes, download on login node)"
for url in https://huggingface.co https://github.com https://pypi.org; do
  if curl -sI -m 8 "$url" >/dev/null; then echo "OK   $url"; else echo "FAIL $url"; fi
done

hr "disk"
df -h "$HOME" 2>/dev/null | tail -1
[ -d /scratch ] && df -h /scratch | tail -1
[ -d /share1 ] && du -sh "/share1/$USER" 2>/dev/null
du -sh "$HOME" 2>/dev/null

hr "codeladder venv"
if [ -x "$CL_VENV/bin/python" ]; then
  "$CL_VENV/bin/python" -c "import codeladder, numpy, scipy, openai; print('venv OK', codeladder.__version__)"
else
  echo "venv not found at $CL_VENV (run scripts/ada/setup_env.sh)"
fi
