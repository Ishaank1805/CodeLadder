"""Recompute the hidden-test labels of BigCodeBench pools for some tasks.

    python scripts/relabel_bcb.py [--tasks BigCodeBench/790 ...] [--tiers t1 t2 ...]

Pools generated before 27 Oct labelled two tasks wrongly: BigCodeBench/790 (a second
TestCase class that the benchmark never runs was counted, so every candidate was "wrong") and
BigCodeBench/1028 (two test methods with multi-line signatures were missed, so labels were too
lenient). The visible test was right in both, so only ``hidden_pass`` of visible-passing
candidates changes. Rewrites the pool files in place and prints what changed; safe to rerun.
Run where the BigCodeBench environment exists (the cluster), not while a job writes the pools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from codeladder.bcb import load_bigcodebench, run_bcb, unit_test_methods  # noqa: E402
from codeladder.config import load_experiment  # noqa: E402
from codeladder.generate import pool_path  # noqa: E402

FIXED = ["BigCodeBench/790", "BigCodeBench/1028"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", default=FIXED)
    ap.add_argument("--tiers", nargs="+", default=["t1", "t2", "t3", "t4", "d1", "d2", "d3", "api"])
    args = ap.parse_args()
    cfg = load_experiment()
    tasks = {t.task_id: t for t in load_bigcodebench() if t.task_id in set(args.tasks)}
    changed = 0
    for tier in args.tiers:
        for tid, task in tasks.items():
            p = pool_path(cfg["paths"]["pools"], task, tier)
            if not p.exists():
                continue
            pool = json.loads(p.read_text())
            methods = unit_test_methods(task.tests or "")
            cands = pool["samples"] + [c for ch in pool["repairs"].values() for c in ch]
            for c in cands:
                if c["visible_pass_rate"] < 1:
                    continue  # failing the visible test already means hidden_pass is False
                res = run_bcb(c["code"], task.tests or "", None, None, cfg["execution"])
                ok = not res.get("fatal") and all(
                    res["methods"].get(m, {}).get("status") == "pass" for m in methods
                )
                if ok != c["hidden_pass"]:
                    print(f"{tier} {tid} {c['cand_id']}: hidden_pass {c['hidden_pass']} -> {ok}")
                    c["hidden_pass"] = ok
                    changed += 1
            p.write_text(json.dumps(pool))
    print(f"{changed} labels changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
