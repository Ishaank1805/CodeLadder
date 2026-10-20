"""Recompute per-candidate energy from the GPU energy logs, counting only busy server time.

``attach-energy`` divides all GPU energy in a generation run's window by the tokens generated
in it. GPUs also draw power while the client runs tests and the server waits (one slow task can
keep four GPUs idle for hours), which inflates joules per token. This script splits each run's
energy into busy energy (seconds when vLLM reported running requests) and total energy, and
writes per candidate:

    energy_j               tokens_out x busy joules per token (used by the energy cost metric)
    energy_total_j         tokens_out x total joules per token
    joules_per_token, joules_per_token_total

Runs come from data/pools/runs.jsonl; the energy log and vLLM log of each run are found by tier
and time in the logs directory. Run only when no job writes the pools being updated.

    python scripts/recompute_energy.py --logs ~/codeladder_logs [--tiers t4] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeladder.energy import busy_energy_between, busy_times, energy_between  # noqa: E402


def cands(p: dict) -> list[dict]:
    return p["samples"] + [c for ch in p["repairs"].values() for c in ch]


def log_span(path: Path) -> tuple[float, float]:
    with open(path) as f:
        rows = list(csv.reader(f))
    return float(rows[1][0]), float(rows[-1][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "data" / "pools"))
    ap.add_argument("--logs", default=str(Path.home() / "codeladder_logs"))
    ap.add_argument("--tiers", nargs="*", help="only these tiers (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logs = Path(args.logs)
    spans = {p: log_span(p) for p in logs.glob("energy-*.csv")}
    runs = [json.loads(x) for x in (Path(args.pools) / "runs.jsonl").read_text().splitlines()]
    runs = [
        r
        for r in runs
        if r["benchmark"] != "toy"
        and r["pools"] > 0
        and (not args.tiers or r["tier"] in args.tiers)
    ]
    busy_cache: dict[Path, list[float]] = {}
    print(
        f"{'tier':5s} {'benchmark':14s} {'tokens':>10s} {'J/tok busy':>10s} {'J/tok total':>11s} {'busy share':>10s}"
    )
    for r in runs:
        tier, bench = r["tier"], r["benchmark"]
        logs_for_run = [
            p for p, (a, b) in spans.items()
            if p.name.startswith(f"energy-{tier}-") and a <= r["start"] + 60 and b >= r["end"] - 60
        ]  # fmt: skip
        if len(logs_for_run) != 1:
            print(f"{tier} {bench}: {len(logs_for_run)} energy logs cover the run, skipped")
            continue
        elog = logs_for_run[0]
        vlog = logs / elog.name.replace("energy-", "vllm-").replace(".csv", ".log")
        if not vlog.exists():
            print(f"{tier} {bench}: no vLLM log for {elog.name}, kept as is")
            continue
        if vlog not in busy_cache:
            busy_cache[vlog] = busy_times(vlog, time.localtime(spans[elog][0]).tm_year)
        files = sorted((Path(args.pools) / bench / tier).glob("*.json"))
        pools = {f: json.loads(f.read_text()) for f in files}

        def in_run(c: dict, r: dict = r) -> bool:
            return r["start"] <= (c.get("extra", {}).get("started_at") or 0) <= r["end"]

        tokens = sum(c["tokens_out"] for p in pools.values() for c in cands(p) if in_run(c))
        if not tokens:
            continue
        total = energy_between(elog, r["start"], r["end"])
        busy = busy_energy_between(elog, r["start"], r["end"], busy_cache[vlog])
        jt, jb = total / tokens, busy / tokens
        print(
            f"{tier:5s} {bench:14s} {tokens:10d} {jb:10.3f} {jt:11.3f} {busy / max(total, 1e-9):10.0%}"
        )
        if args.dry_run:
            continue
        for f, p in pools.items():
            changed = False
            for c in cands(p):
                if in_run(c):
                    c["extra"].update(
                        energy_j=c["tokens_out"] * jb,
                        energy_total_j=c["tokens_out"] * jt,
                        joules_per_token=jb,
                        joules_per_token_total=jt,
                    )
                    changed = True
            if changed:
                f.write_text(json.dumps(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
