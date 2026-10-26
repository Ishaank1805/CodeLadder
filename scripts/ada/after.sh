#!/bin/bash
# Run a command once SLURM jobs have left the queue, on the login node, independent of any
# laptop session:  nohup scripts/ada/after.sh 12345,12346 scripts/some_batch.sh > logs/x.log 2>&1 &
set -u
jobs="$1"; shift
# squeue fails with "Invalid job id" once SLURM has forgotten a finished job; any other failure
# (a busy controller) is retried rather than taken as "done".
while true; do
  if out=$(squeue -h -j "$jobs" -o %i 2>&1); then
    [ -z "$out" ] && break
  else
    case "$out" in *"Invalid job id"*) break ;; esac
  fi
  sleep 60
done
echo "[after] $(date +%F' '%H:%M) jobs $jobs done; running: $*"
exec "$@"
