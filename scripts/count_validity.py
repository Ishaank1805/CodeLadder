"""Count evaluation runs and calibrations under results/ and report the worst violation rate
(share of splits whose true risk exceeds alpha) over all of them.

    python scripts/count_validity.py [results]
"""

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    runs = calibrations = 0
    worst, where = 0.0, None
    for p in sorted(root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        rows = [
            c
            for k in ("calibrated", "calibrated_grouped")
            for c in s.get(k, [])
            if isinstance(c, dict) and "violation_rate" in c
        ]
        rows += [r for fam in (s.get("mean") or {}).values() for r in fam]  # router runs
        if not rows:
            continue
        runs += 1
        for c in rows:
            v = c.get("violation_rate")
            if v is None or v != v:
                continue
            calibrations += 1
            if v > worst:
                worst, where = v, p.parent.name
    print(f"{runs} runs, {calibrations} calibrations, worst violation rate {worst:.2f} ({where})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
