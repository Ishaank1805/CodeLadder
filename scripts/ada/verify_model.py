"""Check a local copy of a Hugging Face model against the SHA-256 of every weight file that
the Hub publishes, and optionally copy a verified copy to a shared folder (marked .copied,
which scripts/ada/common.sh then uses on every node instead of per-node downloads).

    python scripts/ada/verify_model.py <repo> <local dir> [--copy-to <shared dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def hub_hashes(repo: str) -> dict[str, str]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=120) as r:
        items = json.load(r)
    return {i["path"]: i["lfs"]["oid"] for i in items if i.get("lfs")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("local")
    ap.add_argument("--copy-to")
    args = ap.parse_args()
    local = Path(args.local)
    ref = {k: v for k, v in hub_hashes(args.repo).items() if k.endswith(".safetensors")}
    if not ref:  # nothing to compare against: do not report a match
        print(f"no .safetensors files listed for {args.repo} on the Hub")
        return 1
    bad = 0
    for name, oid in sorted(ref.items()):
        f = local / name
        got = sha256(f) if f.exists() else "missing"
        ok = got == oid
        bad += not ok
        print(f"{'ok' if ok else 'BAD'} {name} {got[:16]}")
    if bad:
        print(f"{bad} of {len(ref)} weight files do not match the Hub")
        return 1
    print(f"all {len(ref)} weight files match the Hub")
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for f in local.iterdir():
            if f.is_file() and not f.name.startswith("."):
                shutil.copy2(f, dst / f.name)
        for name, oid in ref.items():
            if sha256(dst / name) != oid:
                print(f"copy of {name} does not match; not marking {dst}")
                return 1
        (dst / ".copied").touch()
        print(f"verified copy in {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
