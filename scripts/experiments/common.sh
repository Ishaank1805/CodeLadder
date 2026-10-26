#!/bin/bash
# Shared helpers for the experiment batches in this folder. Each batch regenerates the
# results/ folders behind one part of docs/RESULTS.md from the cached pools, without GPUs
# (a few steps also write derived data: energy.sh rewrites pool energy fields, oracle.sh writes
# data/oracle, bigcodebench.sh relabels two BigCodeBench tasks).
#
#   scripts/experiments/<batch>.sh            run it (on Ada: sbatch scripts/ada/cpu_job.sbatch scripts/experiments/<batch>.sh)
#   DRY=1 scripts/experiments/<batch>.sh      print the commands only
#
# CL / PY default to the cluster virtualenv when it exists, else to the active environment.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
if [ -x "$HOME/venvs/codeladder/bin/codeladder" ]; then
  CL="${CL:-$HOME/venvs/codeladder/bin/codeladder}"
  PY="${PY:-$HOME/venvs/codeladder/bin/python}"
else
  CL="${CL:-codeladder}"
  PY="${PY:-python3}"
fi
ALPHAS="--alphas 0.05 0.1 0.2"
# Every command runs even if an earlier one failed; the batch then exits with status 1, so a
# SLURM job running it is marked as failed.
FAILS=0
trap 'if [ "$FAILS" -gt 0 ]; then echo "$FAILS command(s) FAILED"; exit 1; fi' EXIT

# cl <command> <args...>: run one codeladder command, print "== args" and ok / FAILED.
cl() {
  echo "== $*"
  if [ -n "${DRY:-}" ]; then return 0; fi
  if nice "$CL" "$@" > /dev/null; then echo ok; else echo FAILED; FAILS=$((FAILS + 1)); fi
}
# ev <args...>: one evaluation at the three risk levels.
ev() { cl evaluate $ALPHAS "$@"; }
# py <script> <args...>: run an analysis script.
py() {
  echo "== $*"
  if [ -n "${DRY:-}" ]; then return 0; fi
  nice "$PY" "$@" || { echo FAILED; FAILS=$((FAILS + 1)); }
}
