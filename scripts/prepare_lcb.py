"""Prepare LiveCodeBench (release_v6) for CodeLadder.

Downloads the six JSONL files of ``livecodebench/code_generation_lite`` if missing,
decompresses the private tests, and writes:

    data/lcb/index.json            one entry per problem (statement, starter code, mode, date)
    data/lcb/tasks/<id>.json.gz    {"public": [...], "private": [...]} test cases

Run once (on the cluster login node, ~4.5 GB download at ~10 MB/s):

    python scripts/prepare_lcb.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import pickle
import subprocess
import zlib
from collections import Counter
from pathlib import Path

FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
BASE = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"


def private_tests(row: dict) -> list[dict]:
    blob = row["private_test_cases"]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode()))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data" / "lcb"))
    args = ap.parse_args()
    root = Path(args.root)
    (root / "tasks").mkdir(parents=True, exist_ok=True)

    index, seen, modes = [], set(), Counter()
    for name in FILES:
        path = root / name
        if not path.exists() or path.stat().st_size == 0:
            print(f"downloading {name} ...", flush=True)
            subprocess.run(["curl", "-sSL", "-o", str(path), f"{BASE}/{name}"], check=True)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                qid = str(row["question_id"])
                if qid in seen:
                    continue
                seen.add(qid)
                public = json.loads(row["public_test_cases"])
                private = private_tests(row)
                ttype = (public or private)[0]["testtype"]
                mode = "stdin" if ttype == "stdin" else "method"
                meta = json.loads(row.get("metadata") or "{}")
                fname = f"{qid}.json.gz"
                (root / "tasks" / fname).write_bytes(
                    gzip.compress(json.dumps({"public": public, "private": private}).encode())
                )
                index.append(
                    {
                        "task_id": f"lcb/{qid}",
                        "question": row["question_content"],
                        "starter_code": row.get("starter_code") or "",
                        "mode": mode,
                        "entry_point": meta.get("func_name") if mode == "method" else None,
                        "release_date": row["contest_date"],
                        "difficulty": row.get("difficulty"),
                        "platform": row.get("platform"),
                        "n_public": len(public),
                        "n_private": len(private),
                        "file": fname,
                    }
                )
                modes[mode] += 1
        print(f"{name}: {len(index)} problems so far", flush=True)

    index.sort(key=lambda m: m["release_date"])
    (root / "index.json").write_text(json.dumps(index))
    print(
        f"wrote {len(index)} problems ({dict(modes)}), dates {index[0]['release_date']} .. {index[-1]['release_date']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
