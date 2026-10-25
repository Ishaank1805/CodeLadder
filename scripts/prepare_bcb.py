"""Download BigCodeBench (release v0.1.4, the version on the public leaderboard) and the ids of
its Hard subset into data/bcb/ through the Hugging Face datasets API (JSON rows, no extra
packages):

    python scripts/prepare_bcb.py [--split v0.1.4]

Writes data/bcb/tasks.jsonl (task_id, complete_prompt, instruct_prompt, code_prompt,
canonical_solution, test, entry_point, libs) and data/bcb/hard_ids.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://datasets-server.huggingface.co/rows"
KEEP = ("task_id", "complete_prompt", "instruct_prompt", "code_prompt", "canonical_solution",
        "test", "entry_point", "libs")  # fmt: skip


def rows(dataset: str, split: str) -> list[dict]:
    out, offset = [], 0
    while True:
        q = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
                    data = json.load(r)
                break
            except Exception:  # noqa: BLE001 (retry transient HTTP errors)
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        batch = [x["row"] for x in data["rows"]]
        out += batch
        offset += len(batch)
        if not batch or offset >= data.get("num_rows_total", offset):
            return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v0.1.4")
    args = ap.parse_args()
    full = rows("bigcode/bigcodebench", args.split)
    hard = rows("bigcode/bigcodebench-hard", args.split)
    out = ROOT / "data" / "bcb"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w") as f:
        for r in full:
            f.write(json.dumps({k: r[k] for k in KEEP}) + "\n")
    (out / "hard_ids.json").write_text(json.dumps(sorted(r["task_id"] for r in hard)))
    print(f"{len(full)} tasks, {len(hard)} hard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
