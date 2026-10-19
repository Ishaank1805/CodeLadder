#!/bin/bash
# Shared settings for all Ada scripts. Source this file; do not run it.
#
# Storage layout (from the Ada user guide):
#   /home/$USER   25 GB, NFS, visible on all nodes, backed up  -> code, venv, container images
#   /share1/$USER 100 GB, visible on the LOGIN node only        -> model weights (long-term)
#   /scratch      local disk on each compute node, purged ~7 days -> per-job copy of weights
# Compute jobs therefore copy weights from ada:/share1 into /scratch before serving.

export CL_REPO="${CL_REPO:-$HOME/codeladder}"
export CL_VENV="${CL_VENV:-$HOME/venvs/codeladder}"
export CL_IMAGES="${CL_IMAGES:-$HOME/images}"
export CL_SHARE="${CL_SHARE:-/share1/$USER/codeladder}"
export CL_MODELS_SHARE="${CL_MODELS_SHARE:-$CL_SHARE/models}"
export CL_SCRATCH="${CL_SCRATCH:-/scratch/$USER/codeladder}"
export CL_ENDPOINTS="${CL_ENDPOINTS:-$HOME/codeladder_endpoints}"
export CL_LOGS="${CL_LOGS:-$HOME/codeladder_logs}"

# vLLM image. v0.6.x supports Turing (sm_75) with fp16 + AWQ via the xformers backend.
# It needs a CUDA 12 capable driver; check_env.sh prints the driver version. If the
# driver is too old, use the llama.cpp fallback (build_llamacpp.sh + serve_llamacpp.sbatch).
export CL_VLLM_IMAGE_URI="${CL_VLLM_IMAGE_URI:-docker://vllm/vllm-openai:v0.6.6.post1}"
export CL_VLLM_SIF="${CL_VLLM_SIF:-$CL_IMAGES/vllm-openai-0.6.6.sif}"

load_modules() {
  module load u18/python/3.11.2 2>/dev/null || module load u18/python/3.10.2
  module load u18/singularity-ce/3.9.6 2>/dev/null || true
}

activate_venv() {
  load_modules
  # shellcheck disable=SC1091
  source "$CL_VENV/bin/activate"
}

# Copy one model directory from the login node's /share1 into this node's /scratch.
stage_model() {
  local name="$1"
  local dst="$CL_SCRATCH/models/$name"
  mkdir -p "$dst"
  if [ ! -f "$dst/.staged" ]; then
    echo "[stage] copying $name from ada:$CL_MODELS_SHARE to $dst"
    rsync -a "ada:$CL_MODELS_SHARE/$name/" "$dst/" && touch "$dst/.staged"
  fi
  echo "$dst"
}
