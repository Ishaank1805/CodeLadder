"""Why do visible-passing candidates fail the hidden tests? First hidden failure type (too slow,
wrong output, crash) per tier and problem kind, for the samples of each pool.

    python scripts/analysis_silent_failures.py --benchmark livecodebench --tiers t1 t2 t3 t4 --json results/silent_failures_lcb.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeladder.config import load_experiment  # noqa: E402
from codeladder.generate import _limits, _run, hidden_chunks  # noqa: E402
from codeladder.scoring import visible_report  # noqa: E402
from codeladder.tasks import load_tasks, prepare_task  # noqa: E402
from codeladder.types import TierPool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    cfg = load_experiment()
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    out = {}
    for tier in args.tiers:
        files = sorted((Path(cfg["paths"]["pools"]) / args.benchmark / tier).glob("*.json"))

        def work(f: Path) -> tuple[str, list[str]] | None:
            p = TierPool.from_dict(json.loads(f.read_text()))
            bad = [c for c in p.samples if c.passes_visible and not c.hidden_pass]
            if not bad:
                return None
            t = tasks[p.task_id]
            prepare_task(t)
            lim = _limits(t.hidden_ref_times, len(t.hidden_inputs), cfg["execution"], t)
            errs = []
            for c in bad:
                err = "passes now"
                for i, j in hidden_chunks(len(t.hidden_inputs)):
                    ins, exp = t.hidden_inputs[i:j], t.hidden_expected[i:j]
                    run = _run(t, c.code, ins, lim[i:j], cfg["execution"], memory_mb=4096)
                    rate, e, _ = visible_report(run, ins, exp, t.atol, t.oracle)
                    if rate < 1.0:
                        err = "exception" if e.startswith("exception") else e
                        break
                errs.append(err)
            t.hidden_inputs, t.hidden_expected = [], None
            return (t.mode, errs)

        kinds: dict[str, Counter] = {}
        with ThreadPoolExecutor(args.workers) as ex:
            for r in ex.map(work, files):
                if r:
                    kinds.setdefault(r[0], Counter()).update(r[1])
        out[tier] = {m: dict(c) for m, c in kinds.items()}
        for m, c in kinds.items():
            n = sum(c.values())
            print(
                f"{tier} {m}: {n} silent failures, "
                + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common())
            )
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
