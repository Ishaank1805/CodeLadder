#!/bin/bash
# One self-contained job per model tier: start vLLM, generate candidate pools for each
# benchmark against it, attach measured GPU energy, then shut down and free the GPUs.
# Chain tiers with dependencies so they run unattended within the QoS limits:
#
#   j1=$(sbatch --parsable --gres=gpu:1 -c 16 --export=ALL,TIER=t1,MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,TP=1,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   j2=$(sbatch --parsable --gres=gpu:1 -c 16 --export=ALL,TIER=t2,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ,TP=1,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   j3=$(sbatch --parsable --dependency=afterany:$j1:$j2 --gres=gpu:2 -c 24 --export=ALL,TIER=t3,MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ,TP=2,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   sbatch --dependency=afterany:$j3 --gres=gpu:4 -c 36 --export=ALL,TIER=t4,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ,TP=4,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch
#
# Generation is resumable, so resubmitting a tier only fills in missing pools.
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cl-tier
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$HOME/codeladder/scripts/ada/common.sh"

: "${TIER:?set TIER}" "${MODEL:?set MODEL}" "${BENCHES:?set BENCHES}"
TP="${TP:-1}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
MAX_LEN="${MAX_LEN:-8192}"
WORKERS="${WORKERS:-$((SLURM_CPUS_PER_TASK - 4))}"
mkdir -p "$CL_ENDPOINTS" "$CL_LOGS"
export HF_HOME="$CL_SCRATCH/hf"
ENERGY_LOG="$CL_LOGS/energy-$TIER-$SLURM_JOB_ID.csv"

MODEL_DIR="$(stage_model "$MODEL")"
"$CL_VENV/bin/codeladder" energy-log --out "$ENERGY_LOG" --interval 1 &
ENERGY_PID=$!

QUANT_ARGS=()
case "$MODEL" in *AWQ*) QUANT_ARGS=(--quantization awq) ;; esac
export VLLM_ATTENTION_BACKEND=XFORMERS
"$CL_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$MODEL" \
  --dtype half "${QUANT_ARGS[@]}" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "$PORT" > "$CL_LOGS/vllm-$TIER-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" "$ENERGY_PID" 2>/dev/null || true
  # Only remove the endpoint file if it still points at this job's server.
  [ "$(cat "$CL_ENDPOINTS/$TIER" 2>/dev/null)" = "$(hostname):$PORT" ] && rm -f "$CL_ENDPOINTS/$TIER"
  return 0
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[tier] vLLM exited during startup" >&2; tail -20 "$CL_LOGS/vllm-$TIER-$SLURM_JOB_ID.log" >&2; exit 1; }
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then READY=1; break; fi
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "[tier] vLLM not ready after 30 minutes" >&2; exit 1; }
echo "$(hostname):$PORT" > "$CL_ENDPOINTS/$TIER"
echo "[tier] $TIER ($MODEL, TP=$TP) serving at $(hostname):$PORT"

activate_venv
cd "$CL_REPO"
for bench in $BENCHES; do
  start=$(date +%s.%N)
  codeladder generate --tier "$TIER" --benchmark "$bench" --workers "$WORKERS" 2>&1 | grep -v "HTTP Request"
  end=$(date +%s.%N)
  codeladder attach-energy --pools data/pools --benchmark "$bench" --tier "$TIER" \
    --log "$ENERGY_LOG" --start "$start" --end "$end" || echo "[tier] energy attach failed for $bench"
done
echo "[tier] $TIER done"
