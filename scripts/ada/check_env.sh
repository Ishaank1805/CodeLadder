#!/bin/bash
# Environment check. Run on the login node, then on a GPU node:
#   bash scripts/ada/check_env.sh
#   srun -A research --qos=medium -p u22 --constraint=2080ti --gres=gpu:1 -c 4 --mem=8G \
#        --time=00:15:00 bash scripts/ada/check_env.sh

set -u
source "$(dirname "$0")/common.sh"
hr() { printf '\n==== %s ====\n' "$1"; }

hr "host"
hostname; date; grep PRETTY_NAME /etc/os-release
hr "glibc"
ldd --version 2>/dev/null | head -1
hr "slurm accounts"
sacctmgr -n show assoc user="$USER" format=Account,QOS,DefaultQOS 2>/dev/null | head -5

hr "GPUs"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
else
  echo "no GPU on this node"
fi

hr "internet"
for url in https://huggingface.co https://github.com https://pypi.org; do
  if curl -sI -m 8 "$url" >/dev/null; then echo "OK   $url"; else echo "FAIL $url"; fi
done

hr "disk"
df -h "$HOME" | tail -1
[ -d /scratch ] && df -h /scratch | tail -1

hr "environments"
"$CL_VENV/bin/python" -c "import codeladder, evalplus; print('harness OK', codeladder.__version__)" 2>&1 | tail -1
"$CL_VLLM_VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda available:', torch.cuda.is_available())" 2>&1 | tail -1
