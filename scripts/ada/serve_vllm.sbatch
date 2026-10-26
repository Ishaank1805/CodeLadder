#!/bin/bash
# Serve one model tier with vLLM (OpenAI-compatible API) on RTX 2080 Ti nodes.
#
#   sbatch --gres=gpu:1 -c 10 --export=ALL,TIER=t1,MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,TP=1 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:1 -c 10 --export=ALL,TIER=t2,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ,TP=1 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:2 -c 20 --export=ALL,TIER=t3,MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ,TP=2 scripts/ada/serve_vllm.sbatch
#   sbatch --gres=gpu:4 -c 40 --export=ALL,TIER=t4,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ,TP=4 scripts/ada/serve_vllm.sbatch
#
# Weights download into the node's /scratch on first use. Once the server answers, it
# writes "host:port" to ~/codeladder_endpoints/$TIER (where configs/models.yaml looks) and
# logs GPU energy to ~/codeladder_logs.
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cl-serve
#SBATCH --output=%x-%j.out

set -euo pipefail
source "$HOME/codeladder/scripts/ada/common.sh"

: "${TIER:?set TIER}" "${MODEL:?set MODEL (Hugging Face repo id)}"
TP="${TP:-1}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
MAX_LEN="${MAX_LEN:-8192}"
mkdir -p "$CL_ENDPOINTS" "$CL_LOGS"
export HF_HOME="$CL_SCRATCH/hf"

MODEL_DIR="$(stage_model "$MODEL")"

"$CL_VENV/bin/codeladder" energy-log --out "$CL_LOGS/energy-$TIER-$SLURM_JOB_ID.csv" --interval 1 &
ENERGY_PID=$!
cleanup() {
  kill "$ENERGY_PID" 2>/dev/null || true
  # Only remove the endpoint file if it still points at this job's server.
  [ "$(cat "$CL_ENDPOINTS/$TIER" 2>/dev/null)" = "$(hostname):$PORT" ] && rm -f "$CL_ENDPOINTS/$TIER"
  return 0
}
trap cleanup EXIT

QUANT_ARGS=()
case "$MODEL" in *AWQ*) QUANT_ARGS=(--quantization awq) ;; esac

# Turing: no bf16 and no FlashAttention 2, so fp16 with the xformers backend.
export VLLM_ATTENTION_BACKEND=XFORMERS
"$CL_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$MODEL" \
  --dtype half "${QUANT_ARGS[@]}" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!

# Publish the endpoint once the server answers. Give up (and free the GPU) if the server
# process dies or is not ready within 30 minutes: a failed vLLM engine can leave the
# API process hanging without ever serving.
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server process exited before becoming ready" >&2
    exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "$(hostname):$PORT" > "$CL_ENDPOINTS/$TIER"
    echo "[serve] $TIER ($MODEL) ready at $(hostname):$PORT"
    READY=1
    break
  fi
  sleep 10
done
if [ "$READY" -ne 1 ]; then
  echo "[serve] not ready after 30 minutes, stopping" >&2
  kill "$SERVER_PID" 2>/dev/null
  exit 1
fi
wait "$SERVER_PID"
