"""Achievable frontier versus what calibration certifies.

For each evaluate run, uses the full replay (all tasks and sample orders) as a stand-in for
the true risk of every threshold setting, and reports the best solved rate any setting
reaches under several risk caps. Comparing this with the calibrated results shows how much
coverage the finite-sample guarantee costs.

    python scripts/analysis_frontier.py results/runs/<benchmarks>/<run> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CAPS = (0.03, 0.05, 0.07, 0.085, 0.10, 0.20)


def main() -> int:
    for arg in sys.argv[1:]:
        run = Path(arg)
        j = json.loads((run / "summary.json").read_text())
        r = np.load(run / "replay.npz")
        risk = r["loss"].mean(axis=(1, 2))
        solved = r["correct"].mean(axis=(1, 2))
        cost = r["cost"].mean(axis=(1, 2))
        print(f"== {run.name}")
        for cap in CAPS:
            ok = np.where(risk <= cap)[0]
            if len(ok) == 0:
                print(f"   true risk <= {cap:.3f}: none")
                continue
            b = ok[np.argmax(solved[ok])]
            print(f"   true risk <= {cap:.3f}: best solved {solved[b]:.3f}  cost {cost[b]:7.0f}  "
                  f"risk {risk[b]:.3f}  setting {j['thresholds'][b]}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
