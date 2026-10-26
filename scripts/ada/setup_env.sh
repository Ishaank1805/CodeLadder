#!/bin/bash
# One-time setup, run on the login node (it has internet access):
#   bash scripts/ada/setup_env.sh
# Installs uv, the harness environment and the vLLM server environment, and caches the
# EvalPlus datasets. Takes ~30 minutes, mostly the vLLM wheel from PyPI (~190 MB at ~100 KB/s).

set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$HOME/venvs" "$CL_ENDPOINTS" "$CL_LOGS"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

[ -d "$CL_VENV" ] || uv venv -p 3.11 "$CL_VENV"
uv pip install -p "$CL_VENV/bin/python" -e "$CL_REPO[benchmarks,energy,analysis,dev]"

# PyPI downloads are very slow from Ada (~100 KB/s) while download.pytorch.org is fast
# (~8 MB/s), so PyTorch and the CUDA libraries come from the PyTorch index. transformers
# and huggingface_hub are pinned to versions from vLLM 0.6.6's era: newer transformers needs
# PyTorch >= 2.6 and silently disables PyTorch support otherwise.
export UV_HTTP_TIMEOUT=300
[ -d "$CL_VLLM_VENV" ] || uv venv -p 3.11 "$CL_VLLM_VENV"
uv pip install -p "$CL_VLLM_VENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match \
  "vllm==0.6.6.post1" "torch==2.5.1+cu124" "transformers==4.47.1" \
  "huggingface_hub[cli]<0.28" "tokenizers<0.22" nvidia-ml-py setuptools

# Cache EvalPlus data in ~/.cache/evalplus so jobs need no downloads.
"$CL_VENV/bin/python" - <<'EOF'
from evalplus.data import get_human_eval_plus, get_mbpp_plus
print("HumanEval+", len(get_human_eval_plus()), "MBPP+", len(get_mbpp_plus()))
EOF

(cd "$CL_REPO" && "$CL_VENV/bin/python" -m pytest -q)
echo "setup done"
