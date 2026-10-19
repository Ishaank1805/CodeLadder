"""A fixed, stratified random subset of a benchmark (for tiers that cannot afford all tasks,
such as the quota-limited free API tier). Strata: difficulty x problem format.

    python scripts/make_subset.py --benchmark livecodebench --n 400 --out data/lcb/subset_api400.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeladder.tasks import load_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = load_tasks(args.benchmark)
    strata = defaultdict(list)
    for t in tasks:
        strata[(t.difficulty or "none", t.mode)].append(t.task_id)
    rng = random.Random(args.seed)
    picked = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        k = round(args.n * len(ids) / len(tasks))  # proportional allocation
        picked += rng.sample(ids, min(k, len(ids)))
    picked = sorted(picked)
    Path(args.out).write_text(json.dumps(picked, indent=0))
    print(
        f"{len(picked)} of {len(tasks)} tasks: "
        + ", ".join(f"{k[0]}/{k[1]} {len(v)}" for k, v in sorted(strata.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
