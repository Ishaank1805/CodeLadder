#!/bin/bash
# One self-contained job per model tier: start vLLM, generate candidate pools for each
# benchmark against it, attach measured GPU energy, ask the model for its confidence in
# each visible-passing candidate (ANNOTATE=0 to skip), then shut down and free the GPUs.
# Chain tiers with dependencies so they run unattended within the QoS limits:
#
#   j1=$(sbatch --parsable --gres=gpu:1 -c 16 --export=ALL,TIER=t1,MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,TP=1,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   j2=$(sbatch --parsable --gres=gpu:1 -c 16 --export=ALL,TIER=t2,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ,TP=1,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   j3=$(sbatch --parsable --dependency=afterany:$j1:$j2 --gres=gpu:2 -c 24 --export=ALL,TIER=t3,MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ,TP=2,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch)
#   sbatch --dependency=afterany:$j3 --gres=gpu:4 -c 36 --export=ALL,TIER=t4,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ,TP=4,BENCHES="humaneval mbpp" scripts/ada/run_tier.sbatch
#
# EXTRAS runs further codeladder commands against the served model (see the end).
# Generation is resumable, so resubmitting a tier only fills in missing pools. To top up
# finished pools after raising n_samples or the repair depth, pass GEN_ARGS="--extend".
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

: "${TIER:?set TIER}" "${MODEL:?set MODEL}"
BENCHES="${BENCHES:-}"  # may be empty when the job only runs EXTRAS
TP="${TP:-1}"
PP="${PP:-1}"  # pipeline stages; TP x PP GPUs (e.g. TP=2 PP=2 when TP=4 breaks AWQ alignment)
# GPU_UTIL / VLLM_EXTRA tune memory for large models (e.g. GPU_UTIL=0.82
# VLLM_EXTRA="--max-num-seqs 16 --enforce-eager" for DeepSeek 33B with pipeline stages).
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
MAX_LEN="${MAX_LEN:-8192}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
WORKERS="${WORKERS:-$(( CPUS > 5 ? CPUS - 4 : 1 ))}"  # leave CPUs for the server and energy log
mkdir -p "$CL_ENDPOINTS" "$CL_LOGS"
export HF_HOME="$CL_SCRATCH/hf"
ENERGY_LOG="$CL_LOGS/energy-$TIER-$SLURM_JOB_ID.csv"

MODEL_DIR="$(stage_model "$MODEL")"
"$CL_VENV/bin/codeladder" energy-log --out "$ENERGY_LOG" --interval 1 &
ENERGY_PID=$!

QUANT_ARGS=()
case "$MODEL" in *AWQ*) QUANT_ARGS=(--quantization awq) ;; esac
export VLLM_ATTENTION_BACKEND=XFORMERS
# NCCL_P2P_DISABLE=1 avoids NCCL hangs over PCIe on consumer GPUs (needed for PP on 2080 Ti).
[ -n "${NCCL_P2P_DISABLE:-}" ] && export NCCL_P2P_DISABLE
"$CL_VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$MODEL" \
  --dtype half "${QUANT_ARGS[@]}" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "${GPU_UTIL:-0.90}" ${VLLM_EXTRA:-} \
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
alive() {
  # A crashed or hung server would otherwise turn every later step into fast request failures.
  if ! kill -0 "$SERVER_PID" 2>/dev/null || ! curl -sf -m 60 "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "[tier] vLLM server is down; stopping (resubmit to resume)" >&2
    tail -20 "$CL_LOGS/vllm-$TIER-$SLURM_JOB_ID.log" >&2
    exit 1
  fi
}
for bench in $BENCHES; do
  alive
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  codeladder generate --tier "$TIER" --benchmark "$bench" --workers "$WORKERS" ${GEN_ARGS:-} 2>&1 \
    | grep -v "HTTP Request" || echo "[tier] generation failed for $bench (resubmit to resume)"
  end=$(date +%s.%N)
  codeladder attach-energy --pools data/pools --benchmark "$bench" --tier "$TIER" \
    --log "$ENERGY_LOG" --start "$start" --end "$end" || echo "[tier] energy attach failed for $bench"
  alive
  if [ "${ANNOTATE:-1}" = "1" ]; then
    # Verbalised confidence for every visible-passing candidate (score "selfconf").
    codeladder annotate-confidence --tier "$TIER" --benchmark "$bench" --workers "$WORKERS" 2>&1 \
      | grep -v "HTTP Request" || echo "[tier] confidence annotation failed for $bench"
  fi
done
# Optional extra commands against this tier's server, separated by ";", for example
# EXTRAS="make-traces --tier t1 --benchmark humaneval; make-selftests --tier t1 --benchmark mbpp"
if [ -n "${EXTRAS:-}" ]; then
  IFS=';' read -ra EXTRA_CMDS <<< "$EXTRAS"
  for cmd in "${EXTRA_CMDS[@]}"; do
    alive
    # shellcheck disable=SC2086
    codeladder $cmd --workers "$WORKERS" 2>&1 | grep -v "HTTP Request" || echo "[tier] extra failed: $cmd"
  done
fi
alive
echo "[tier] $TIER done"
