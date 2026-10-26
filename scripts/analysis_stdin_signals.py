"""X2: how well do confidence signals separate hidden-correct from hidden-wrong visible-passing
candidates on LiveCodeBench stdin problems? AUROC per tier for the visible-pass share, agreement
on mutated public inputs, agreement on stress inputs, and the adopted rule (stress timeout =
failed check), with and without self-confidence. Also: how often a stress timeout is a true
failure.

    python scripts/analysis_stdin_signals.py --tiers t1 t2 t3 t4 d1 --json results/x2_stdin_signals.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeladder.tasks import load_tasks  # noqa: E402
from codeladder.types import TierPool  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # rank-based (Mann-Whitney) with ties averaged
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    rpos = sum(ranks[k] for k, (_, y) in enumerate(allv) if y)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


SCORES = [
    "visible",
    "probes",
    "probes_stress",
    "stress",
    "visible_timeout_fail",
    "rule",
    "rule_x_selfconf",
]


def tier_rows(pools_root: Path, tier: str, modes: dict[str, str]) -> list[dict]:
    rows = []
    for f in sorted((pools_root / "livecodebench" / tier).glob("*.json")):
        p = TierPool.from_dict(json.loads(f.read_text()))
        if modes.get(p.task_id) != "stdin":
            continue
        good = [c for c in p.samples if c.passes_visible]
        if not good:
            continue
        n = len(p.samples)
        ss = [c.extra.get("stress_sig", "") for c in good]
        c_pr = Counter(c.signature for c in good)
        c_ps = Counter(c.signature + s for c, s in zip(good, ss, strict=True))
        c_s = Counter(ss)
        for c, s in zip(good, ss, strict=True):
            ok_checks = 0.0 if '"timeout"' in s else 1.0
            rule = c_s[s] / n * ok_checks
            rows.append({
                "ok": bool(c.hidden_pass), "has_stress": bool(s), "timeout": not ok_checks,
                "visible": len(good) / n, "probes": c_pr[c.signature] / n,
                "probes_stress": c_ps[c.signature + s] / n, "stress": c_s[s] / n,
                "visible_timeout_fail": len(good) / n * ok_checks, "rule": rule,
                "rule_x_selfconf": rule * float(c.extra.get("self_conf") or 0.0),
            })  # fmt: skip
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "data" / "pools"))
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    modes = {t.task_id: t.mode for t in load_tasks("livecodebench")}
    out = {}
    for tier in args.tiers:
        rows = tier_rows(Path(args.pools), tier, modes)
        if not rows:
            continue
        to = [r for r in rows if r["timeout"]]
        out[tier] = {
            "candidates": len(rows),
            "correct": sum(r["ok"] for r in rows),
            "with_stress_inputs": sum(r["has_stress"] for r in rows),
            "stress_timeouts": len(to),
            "stress_timeouts_hidden_wrong": sum(not r["ok"] for r in to),
            "auroc": {k: auroc([r[k] for r in rows if r["ok"]], [r[k] for r in rows if not r["ok"]]) for k in SCORES},
        }  # fmt: skip
        a = out[tier]["auroc"]
        print(f"{tier}: {len(rows)} candidates, timeouts {len(to)} ({out[tier]['stress_timeouts_hidden_wrong']} wrong); "
              + "  ".join(f"{k} {v:.3f}" for k, v in a.items()))  # fmt: skip
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
