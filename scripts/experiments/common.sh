#!/bin/bash
# Shared helpers for the experiment batches in this folder. Each batch regenerates the
# results/ folders behind one part of docs/RESULTS.md from the cached pools (CPU only).
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

# cl <command> <args...>: run one codeladder command, print "== args" and ok / FAILED.
cl() {
  echo "== $*"
  if [ -n "${DRY:-}" ]; then return 0; fi
  nice "$CL" "$@" > /dev/null && echo ok || echo FAILED
}
# ev <args...>: one evaluation at the three risk levels.
ev() { cl evaluate $ALPHAS "$@"; }
# py <script> <args...>: run an analysis script.
py() {
  echo "== $*"
  if [ -n "${DRY:-}" ]; then return 0; fi
  nice "$PY" "$@" || echo FAILED
}
