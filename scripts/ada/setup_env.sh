#!/bin/bash
# One-time setup on the Ada LOGIN node (it has internet access).
#   bash scripts/ada/setup_env.sh            # venv + python deps + evalplus data
#   bash scripts/ada/setup_env.sh --images   # also pull the vLLM Singularity image (~10 GB)

set -euo pipefail
source "$(dirname "$0")/common.sh"
load_modules

mkdir -p "$CL_ENDPOINTS" "$CL_LOGS" "$CL_IMAGES" "$CL_MODELS_SHARE"

if [ ! -d "$CL_VENV" ]; then
  python3 -m venv "$CL_VENV"
fi
source "$CL_VENV/bin/activate"
pip install --upgrade pip
# --prefer-binary picks the newest dependency versions that ship wheels for this old glibc.
pip install --prefer-binary -e "$CL_REPO[benchmarks,energy,analysis,dev]"
pip install --prefer-binary "huggingface_hub[cli]"

# Pre-download EvalPlus data into ~/.cache/evalplus so compute nodes need no internet.
python - <<'EOF'
from evalplus.data import get_human_eval_plus, get_mbpp_plus
print("HumanEval+", len(get_human_eval_plus()), "MBPP+", len(get_mbpp_plus()))
EOF

if [ "${1:-}" = "--images" ]; then
  export SINGULARITY_CACHEDIR="$CL_SHARE/.singularity_cache"
  export SINGULARITY_TMPDIR="$CL_SHARE/.singularity_tmp"
  mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
  singularity pull "$CL_VLLM_SIF" "$CL_VLLM_IMAGE_URI"
fi

echo "setup done. Next: bash scripts/ada/download_models.sh"
