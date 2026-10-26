#!/bin/bash
# BigCodeBench execution environment on the cluster (run on a compute node, as a CPU job):
#   sbatch -c 8 --mem=32G scripts/ada/cpu_job.sbatch scripts/ada/setup_bcb_env.sh
# Python 3.10 (the benchmark's pins need it). Packages are downloaded through a PyPI mirror
# (files.pythonhosted.org runs at about 50 KB/s from the campus network, the mirror at about
# 1 MB/s), every file is checked against the SHA-256 published by pypi.org, and only then
# installed offline from the verified files. Packages that only exist as source for 3.10
# (scikit-image 0.18.0 and others) are built without isolation against pinned build tools
# (configs/bigcodebench-build-requirements.txt). Safe to rerun: downloads are skipped when
# present and installs are idempotent.
set -euo pipefail
ENV="${BCB_ENV:-$HOME/venvs/bcb}"
WHEELS="${BCB_WHEELS:-$HOME/bcb_wheels}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REPO="${CL_REPO:-$HOME/codeladder}"
REQ="$REPO/configs/bigcodebench-requirements.txt"
BUILD="$REPO/configs/bigcodebench-build-requirements.txt"
UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" > /dev/null || UV=uv
echo "[bcb] $(date +%F' '%H:%M) $(hostname): $(gcc --version | head -1)"
mkdir -p "$WHEELS"
if [ ! -x "$ENV/bin/python" ]; then
  "$UV" venv --seed --python 3.10 "$ENV"
fi
INC=$("$ENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
if [ ! -f "$INC/Python.h" ]; then
  echo "[bcb] no Python headers in $INC: recreating the environment with a uv-managed Python 3.10"
  rm -rf "$ENV"
  UV_PYTHON_PREFERENCE=only-managed "$UV" venv --seed --python 3.10 "$ENV"
fi
"$ENV/bin/python" --version
PIP=("$ENV/bin/python" -m pip)
"${PIP[@]}" download -r "$REQ" -r "$BUILD" -d "$WHEELS" --index-url "$MIRROR" --timeout 120 --retries 10
"${CL_VENV:-$HOME/venvs/codeladder}/bin/python" "$REPO/scripts/verify_wheels.py" "$WHEELS"
"${PIP[@]}" install --no-index --find-links "$WHEELS" -r "$BUILD"
SRC=$(find "$WHEELS" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.zip' \) -printf '%f\n' \
  | sed -E 's/-[0-9][^-]*\.(tar\.gz|zip)$//' | sort -u)
echo "[bcb] building from source without isolation: $SRC"
# shellcheck disable=SC2086
"${PIP[@]}" install --no-index --find-links "$WHEELS" --no-build-isolation $SRC
"${PIP[@]}" install --no-index --find-links "$WHEELS" -r "$REQ"
"$ENV/bin/python" -c "import numpy, pandas, sklearn, scipy, matplotlib, tensorflow, nltk, cv2, librosa, skimage, Levenshtein; print('imports ok', numpy.__version__, tensorflow.__version__, skimage.__version__)"
# NLTK and TextBlob data that many tasks load at import time (stopwords, punkt, wordnet, ...),
# from the official NLTK data index, into ~/nltk_data (the runner points NLTK_DATA there).
NLTK_DIR="${NLTK_DATA:-$HOME/nltk_data}"
"$ENV/bin/python" - "$NLTK_DIR" <<'PY'
import sys, nltk
for pkg in ("stopwords", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "wordnet", "omw-1.4", "vader_lexicon", "brown", "movie_reviews", "conll2000", "words",
            "maxent_ne_chunker", "maxent_ne_chunker_tab", "names", "gutenberg"):
    print(pkg, nltk.download(pkg, download_dir=sys.argv[1], quiet=True))
PY
du -sh "$ENV" "$WHEELS" "$NLTK_DIR"
echo "[bcb] $(date +%F' '%H:%M) done"
