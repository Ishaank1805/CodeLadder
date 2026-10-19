#!/bin/bash
# Serve one model tier with vLLM (OpenAI-compatible) on Ada RTX 2080 Ti nodes.
#
#   sbatch --gres=gpu:1 -c 10 --export=ALL,TIER=t1,MODEL=Qwen2.5-Coder-1.5B-Instruct,TP=1 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:1 -c 10 --export=ALL,TIER=t2,MODEL=Qwen2.5-Coder-7B-Instruct-AWQ,TP=1 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:2 -c 20 --export=ALL,TIER=t3,MODEL=Qwen2.5-Coder-14B-Instruct-AWQ,TP=2 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:4 -c 40 --export=ALL,TIER=t4,MODEL=Qwen2.5-Coder-32B-Instruct-AWQ,TP=4 scripts/ada/serve_vllm.sbatch
#
# Once the server is up it writes "host:port" to ~/codeladder_endpoints/$TIER, which is
# where configs/models.yaml looks. A GPU energy log is written alongside for the
# sustainability metrics. Keep the GPU:CPU ratio at 1:10 as the Ada guide asks.
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --exclude=gnode[01-40]
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cl-serve
#SBATCH --output=%x-%j.out

set -euo pipefail
source "$HOME/codeladder/scripts/ada/common.sh"
load_modules

: "${TIER:?set TIER}" "${MODEL:?set MODEL (directory name under /share1/.../models)}"
TP="${TP:-1}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
MAX_LEN="${MAX_LEN:-8192}"
mkdir -p "$CL_ENDPOINTS" "$CL_LOGS"

MODEL_DIR="$(stage_model "$MODEL")"

# GPU energy log for the whole serving window (needs nvidia-ml-py in the venv).
"$CL_VENV/bin/codeladder" energy-log --out "$CL_LOGS/energy-$TIER-$SLURM_JOB_ID.csv" --interval 1 &
ENERGY_PID=$!
trap 'kill $ENERGY_PID 2>/dev/null; rm -f "$CL_ENDPOINTS/$TIER"' EXIT

QUANT_ARGS=()
case "$MODEL" in *AWQ*) QUANT_ARGS=(--quantization awq) ;; esac

# Turing has no FlashAttention 2 and no bf16: use fp16 + xformers.
export VLLM_ATTENTION_BACKEND=XFORMERS
singularity exec --nv --bind "$CL_SCRATCH" "$CL_VLLM_SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "Qwen/$MODEL" \
    --dtype half "${QUANT_ARGS[@]}" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!

# Publish the endpoint once the server answers.
for _ in $(seq 1 180); do
  if curl -s "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "$(hostname):$PORT" > "$CL_ENDPOINTS/$TIER"
    echo "[serve] $TIER ($MODEL) ready at $(hostname):$PORT"
    break
  fi
  sleep 10
done
wait $SERVER_PID
