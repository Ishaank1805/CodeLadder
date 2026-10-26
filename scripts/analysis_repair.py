"""E4: when does repair beat resampling?

For every tier and error class of the failing candidate (wrong answer, runtime error,
timeout, broken code), compare the success rate of one repair round with the success rate
of a fresh sample from the same tier. Success is measured on the hidden tests (true
correctness) and on the visible tests (what the controller can see).

    python scripts/analysis_repair.py --benchmarks humaneval mbpp --json results/e4_repair.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def error_class(err: str) -> str:
    if err == "pass":
        return "pass"
    if err.startswith("exception:"):
        return "runtime"
    if err in ("syntax_error", "load_error", "missing_entry_point", "crashed"):
        return "broken"
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "data" / "pools"))
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp"])
    ap.add_argument("--json")
    args = ap.parse_args()

    # Repairs only happen on tasks where a sample failed, which are harder than average, so a
    # fair baseline is a fresh sample's success on the SAME tasks: for every repair attempt we
    # add its task's fresh-sample success rate to the matching (tier, class) bucket.
    repair = defaultdict(lambda: [0, 0, 0.0])  # (tier, class) -> [n, repair ok, sum of fresh rate]
    for b in args.benchmarks:
        for tier_dir in sorted((Path(args.pools) / b).glob("t*")):
            tier = tier_dir.name
            for f in tier_dir.glob("*.json"):
                p = json.loads(f.read_text())
                ss = p["samples"]
                task_fresh = sum(c["hidden_pass"] for c in ss) / max(len(ss), 1)
                for s, chain in p["repairs"].items():
                    prev = ss[int(s)]
                    for c in chain:  # each repair round, keyed by the failure it was shown
                        r = repair[(tier, error_class(prev["error_type"]))]
                        r[0] += 1
                        r[1] += c["hidden_pass"]
                        r[2] += task_fresh
                        prev = c

    rows = []
    print(
        f"{'tier':5s} {'failure shown':14s} {'n':>6s} {'repair ok':>9s} {'fresh ok (same tasks)':>22s} {'better':>8s}"
    )
    for (tier, cls), (n, ok_h, fresh_sum) in sorted(repair.items()):
        rep_rate, fresh_rate = ok_h / max(n, 1), fresh_sum / max(n, 1)
        better = "repair" if rep_rate > fresh_rate else "resample"
        rows.append({"tier": tier, "failure": cls, "n_repairs": n, "repair_success": rep_rate,
                     "fresh_success": fresh_rate, "better": better})  # fmt: skip
        print(f"{tier:5s} {cls:14s} {n:6d} {rep_rate:9.3f} {fresh_rate:22.3f} {better:>8s}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
