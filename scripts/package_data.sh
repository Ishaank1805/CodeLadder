#!/bin/bash
# Bundle the replay data for release (run on the cluster from the repository root):
# pools, LLM response caches, stress inputs, BigCodeBench task data, baseline generations,
# per-test outcomes (X17), the demo bundle, energy and vLLM logs, and the result summaries,
# bootstrap intervals, policies and report tables. Nothing here contains credentials (keys
# live in ~/.codeladder_keys).
#
#   scripts/package_data.sh /path/to/codeladder-data-$(date +%F).tar.zst
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:?usage: package_data.sh OUT.tar.zst}"
logs="${CL_LOGS:-$HOME/codeladder_logs}"
manifest=$(mktemp)
trap 'rm -f MANIFEST.txt "$manifest"' EXIT
{
  echo "codeladder replay data, $(date -u +%FT%TZ), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  for d in data/pools/*/*; do echo "pools $d $(find "$d" -name '*.json' | wc -l)"; done
} > "$manifest"
cp "$manifest" MANIFEST.txt
data=(MANIFEST.txt configs data/pools data/llm_cache data/baselines)
for p in data/lcb/stress data/lcb/subset_api400.json data/bcb/tasks.jsonl data/bcb/probes.json \
  data/bcb/excluded.json data/bcb/excluded_reasons.json data/bcb/hard_ids.json \
  data/bcb/subset_api400.json data/oracle data/demo results/analysis results/baselines results/ci \
  results/policies results/tables; do
  [ -e "$p" ] && data+=("$p")
done
mapfile -t summaries < <(find results -name summary.json -not -path 'results/archive/*')
mapfile -t logfiles < <(cd "$(dirname "$logs")" && find "$(basename "$logs")" -maxdepth 1 \
  \( -name 'energy-*.csv' -o -name 'vllm-*.log' \))
# --exclude must come before the paths it applies to; -C applies to the paths after it.
tar --use-compress-program="zstd -T0 -10" --exclude='*.tmp' -cf "$out" \
  "${data[@]}" "${summaries[@]}" -C "$(dirname "$logs")" "${logfiles[@]}"
ls -lh "$out"
