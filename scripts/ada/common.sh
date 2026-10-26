#!/bin/bash
# Shared settings for the cluster scripts. Source this file; do not run it.
#
# Cluster layout (checked Oct 2026):
#   login node   Rocky Linux 9, internet access
#   u22          GPU partition, Ubuntu 22.04, driver 570 (CUDA 12.8); RTX 2080 Ti nodes carry
#                the feature "2080ti" (use --constraint=2080ti); others are GTX 1080 Ti
#   u22-cpu      CPU-only partition, restricted to another lab: run CPU jobs on u22 instead
#   $HOME        NFS, visible on every node -> code and Python environments
#   /scratch     local disk per compute node (purged after ~7 days) -> model weights, HF cache
# Compute nodes have internet access, so model weights download straight into /scratch.
#
# Python environments are built with uv's standalone Python builds, which run on both the
# Rocky login node and the Ubuntu compute nodes:
#   ~/venvs/codeladder   the harness (this repo, editable install)
#   ~/venvs/vllm         the model server (vllm==0.6.6.post1, supports Turing sm_75)

export CL_REPO="${CL_REPO:-$HOME/codeladder}"
export CL_VENV="${CL_VENV:-$HOME/venvs/codeladder}"
export CL_VLLM_VENV="${CL_VLLM_VENV:-$HOME/venvs/vllm}"
export CL_SCRATCH="${CL_SCRATCH:-/scratch/$USER/codeladder}"
export CL_ENDPOINTS="${CL_ENDPOINTS:-$HOME/codeladder_endpoints}"
export CL_LOGS="${CL_LOGS:-$HOME/codeladder_logs}"
export PATH="$HOME/.local/bin:$PATH"

activate_venv() {
  # shellcheck disable=SC1091
  source "$CL_VENV/bin/activate"
}

# Download a Hugging Face model into this node's /scratch (reused by later jobs on the node).
# A complete copy under $CL_MODEL_CACHE (home, visible on every node; marked with .copied) is
# used directly instead, which avoids a slow download on a new node.
export CL_MODEL_CACHE="${CL_MODEL_CACHE:-$HOME/hf_models}"
stage_model() {
  local repo="$1"
  local shared="$CL_MODEL_CACHE/$(basename "$repo")"
  if [ -f "$shared/.copied" ]; then
    echo "[stage] using $shared" >&2
    echo "$shared"
    return
  fi
  local dst="$CL_SCRATCH/models/$(basename "$repo")"
  mkdir -p "$dst"
  for attempt in 1 2; do
    if [ ! -f "$dst/.complete" ]; then
      echo "[stage] downloading $repo -> $dst" >&2
      # A failed or partial download must stop the job: a server started on a partial copy
      # loads and answers with garbage (seen once: identical replies to every prompt).
      if ! HF_HUB_ENABLE_HF_TRANSFER=0 "$CL_VLLM_VENV/bin/huggingface-cli" download "$repo" \
        --include "*.json" "*.safetensors" "*.model" "*.txt" "*.py" \
        --local-dir "$dst" >&2; then
        echo "[stage] download of $repo failed; not serving a partial copy" >&2
        return 1
      fi
      touch "$dst/.complete"
    fi
    # Once per node and copy: every weight file must match the SHA-256 the Hub publishes.
    if [ -f "$dst/.verified" ] || "$CL_VENV/bin/python" "$CL_REPO/scripts/ada/verify_model.py" "$repo" "$dst" >&2; then
      touch "$dst/.verified"
      echo "$dst"
      return
    fi
    echo "[stage] $dst does not match the Hub; deleting it and downloading again" >&2
    rm -rf "$dst" && mkdir -p "$dst"
  done
  return 1
}
