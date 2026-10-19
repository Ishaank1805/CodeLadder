#!/bin/bash
# Run a command once SLURM jobs have left the queue, on the login node, independent of any
# laptop session:  nohup scripts/ada/after.sh 12345,12346 scripts/some_batch.sh > logs/x.log 2>&1 &
set -u
jobs="$1"; shift
while [ -n "$(squeue -h -j "$jobs" -o %i 2>/dev/null)" ]; do sleep 60; done
echo "[after] $(date +%F' '%H:%M) jobs $jobs done; running: $*"
exec "$@"
