#!/bin/bash
# Build pools for the free API tier within its daily quota: generation stops cleanly when every
# key is rate-limited for hours, so this loop sleeps and resumes until all pools exist, then
# adds self-confidence ratings the same way. Run on the cluster login node (network-bound):
#
#   nohup scripts/run_api_tier.sh humaneval mbpp > logs/api-tier.log 2>&1 &
#   TASK_IDS=data/lcb/subset_api400.json nohup scripts/run_api_tier.sh livecodebench > logs/api-lcb.log 2>&1 &
#   TASK_IDS=data/bcb/subset_api400.json nohup scripts/run_api_tier.sh bigcodebench > logs/api-bcb.log 2>&1 &
#
# Keys: GROQ_API_KEY_1.. in ~/.codeladder_keys (mode 600, never in git).
set -u
cd "$(dirname "$0")/.."
CL="${CL:-$HOME/venvs/codeladder/bin/codeladder}"
TIER="${TIER:-api}"
# TASK_IDS=<json list> restricts generation to a fixed subset (for example LiveCodeBench).
want() {
  if [ -n "${TASK_IDS:-}" ]; then python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TASK_IDS"; return; fi
  case "$1" in
    humaneval) echo 164 ;; mbpp) echo 378 ;; livecodebench) echo 1054 ;; bigcodebench) echo 1134 ;;
    *) echo "unknown benchmark $1" >&2; exit 1 ;;
  esac
}
# Pools that count towards the target: with TASK_IDS only those of the subset (a full run may
# share the folder).
have_pools() {
  if [ -z "${TASK_IDS:-}" ]; then ls "data/pools/$1/$TIER" 2>/dev/null | wc -l; return; fi
  python3 - "$TASK_IDS" "data/pools/$1/$TIER" <<'PY'
import json, os, sys
ids, d = json.load(open(sys.argv[1])), sys.argv[2]
print(sum(os.path.exists(os.path.join(d, t.replace("/", "__") + ".json")) for t in ids))
PY
}
SUBSET_ARGS=()
[ -n "${TASK_IDS:-}" ] && SUBSET_ARGS=(--task-ids "$TASK_IDS")
for bench in "$@"; do
  while true; do
    have=$(have_pools "$bench")
    need=$(want "$bench") || exit 1
    [ "$have" -ge "$need" ] && break
    echo "[api] $(date +%F' '%H:%M) $bench: $have/$need pools"
    nice "$CL" generate --tier "$TIER" --benchmark "$bench" --workers 4 "${SUBSET_ARGS[@]}" 2>&1 | grep -v "HTTP Request"
    have2=$(have_pools "$bench")
    [ "$have2" -ge "$need" ] && break
    [ "$have2" -eq "$have" ] && sleep 3600  # no progress: quota spent, wait
  done
done
for bench in "$@"; do
  echo "[api] $(date +%F' '%H:%M) annotating $bench"
  until nice "$CL" annotate-confidence --tier "$TIER" --benchmark "$bench" --workers 4 2>&1 | grep -v "HTTP Request" | tee /dev/stderr | grep -q "annotated 0 candidates"; do
    sleep 600
  done
done
echo "[api] done"
