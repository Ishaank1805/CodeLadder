#!/bin/bash
# Bundle the replay data for release (run on the cluster from the repository root):
# pools, LLM response caches, stress inputs, baseline generations, energy and vLLM metric logs,
# and evaluation summaries. Nothing here contains credentials (keys live in ~/.codeladder_keys).
#
#   scripts/package_data.sh /path/to/codeladder-data-$(date +%F).tar.zst
set -euo pipefail
out="${1:?usage: package_data.sh OUT.tar.zst}"
logs="${CL_LOGS:-$HOME/codeladder_logs}"
manifest=$(mktemp)
{
  echo "codeladder replay data, $(date -u +%FT%TZ), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  for d in data/pools/*/*; do echo "pools $d $(ls "$d" | wc -l)"; done
} > "$manifest"
cp "$manifest" MANIFEST.txt
tar --use-compress-program="zstd -T0 -10" -cf "$out" \
  MANIFEST.txt configs data/pools data/llm_cache data/lcb/stress data/lcb/subset_api400.json data/baselines \
  --exclude='*.tmp' \
  -C "$(dirname "$logs")" "$(basename "$logs")/energy-"*.csv \
  -C "$PWD" $(ls -d results/*/summary.json results/*.json 2>/dev/null)
rm -f MANIFEST.txt "$manifest"
ls -lh "$out"
