#!/bin/bash
# BigCodeBench execution environment on the cluster (run on a compute node, as a CPU job):
#   sbatch -c 8 --mem=32G scripts/ada/cpu_job.sbatch scripts/ada/setup_bcb_env.sh
# Python 3.10 (the benchmark's pins need it). Packages are downloaded through a PyPI mirror
# (files.pythonhosted.org runs at about 50 KB/s from the campus network, the mirror at about
# 1 MB/s), every file is checked against the SHA-256 published by pypi.org, and only then
# installed offline from the verified files.
set -euo pipefail
ENV="${BCB_ENV:-$HOME/venvs/bcb}"
WHEELS="${BCB_WHEELS:-$HOME/bcb_wheels}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REQ="$HOME/codeladder/configs/bigcodebench-requirements.txt"
UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" > /dev/null || UV=uv
echo "[bcb] $(date +%F' '%H:%M) $(hostname): python $(python3 --version 2>&1)"
mkdir -p "$WHEELS"
if [ ! -x "$ENV/bin/python" ]; then
  "$UV" venv --seed --python 3.10 "$ENV"
fi
"$ENV/bin/python" --version
"$ENV/bin/python" -m pip download -r "$REQ" -d "$WHEELS" --index-url "$MIRROR" --timeout 120 --retries 10
"$HOME/venvs/codeladder/bin/python" "$HOME/codeladder/scripts/verify_wheels.py" "$WHEELS"
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEELS" -r "$REQ"
"$ENV/bin/python" -c "import numpy, pandas, sklearn, scipy, matplotlib, tensorflow, nltk, cv2, librosa; print('imports ok', numpy.__version__, tensorflow.__version__)"
du -sh "$ENV" "$WHEELS"
echo "[bcb] $(date +%F' '%H:%M) done"
