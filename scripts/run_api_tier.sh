#!/bin/bash
# Build pools for the free API tier within its daily quota: generation stops cleanly when every
# key is rate-limited for hours, so this loop sleeps and resumes until all pools exist, then
# adds self-confidence ratings the same way. Run on the cluster login node (network-bound):
#
#   nohup scripts/run_api_tier.sh humaneval mbpp > logs/api-tier.log 2>&1 &
#
# Keys: GROQ_API_KEY_1.. in ~/.codeladder_keys (mode 600, never in git).
set -u
cd "$(dirname "$0")/.."
CL="${CL:-$HOME/venvs/codeladder/bin/codeladder}"
TIER="${TIER:-api}"
want() { case "$1" in humaneval) echo 164 ;; mbpp) echo 378 ;; livecodebench) echo 1054 ;; esac; }
for bench in "$@"; do
  while true; do
    have=$(ls "data/pools/$bench/$TIER" 2>/dev/null | wc -l)
    need=$(want "$bench")
    [ "$have" -ge "$need" ] && break
    echo "[api] $(date +%F' '%H:%M) $bench: $have/$need pools"
    nice "$CL" generate --tier "$TIER" --benchmark "$bench" --workers 4 2>&1 | grep -v "HTTP Request"
    have2=$(ls "data/pools/$bench/$TIER" 2>/dev/null | wc -l)
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
