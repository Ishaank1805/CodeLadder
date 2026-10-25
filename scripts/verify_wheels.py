"""Check every downloaded distribution against the SHA-256 that pypi.org publishes for it.

    python scripts/verify_wheels.py <wheel dir>

Used when packages come from a PyPI mirror (faster than files.pythonhosted.org from the
cluster): each file must be byte-identical to the official PyPI release. Exits non-zero on any
mismatch or unknown file, so an install step that follows never sees an unverified file.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from packaging.utils import parse_sdist_filename, parse_wheel_filename


def official_digests(name: str, version: str) -> dict[str, str]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    return {u["filename"]: u["digests"]["sha256"] for u in data["urls"]}


def main() -> int:
    root = Path(sys.argv[1])
    bad = 0
    files = sorted(p for p in root.iterdir() if p.is_file())
    for f in files:
        if f.name.endswith(".whl"):
            name, version, *_ = parse_wheel_filename(f.name)
        else:
            name, version = parse_sdist_filename(f.name)
        try:
            ref = official_digests(str(name), str(version)).get(f.name)
        except Exception as e:  # noqa: BLE001 (report and count as unverified)
            print(f"UNVERIFIED {f.name}: {e}")
            bad += 1
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if ref != got:
            print(f"MISMATCH {f.name}: pypi {ref} local {got}")
            bad += 1
    print(f"{len(files) - bad} of {len(files)} files match pypi.org")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
