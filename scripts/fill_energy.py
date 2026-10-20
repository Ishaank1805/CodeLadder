"""Fill in energy for candidates generated before per-run energy attribution existed.

For every tier, candidates without ``energy_j`` get tokens_out x (median measured joules per
output token of that tier's other candidates), and are marked ``energy_estimated``. Run it
only when no generation or annotation job is writing the same pools.

    python scripts/fill_energy.py --benchmarks humaneval mbpp
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cands(p: dict) -> list[dict]:
    return p["samples"] + [c for ch in p["repairs"].values() for c in ch]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "data" / "pools"))
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp"])
    ap.add_argument("--tiers", nargs="*", help="only these tiers (default: every t* tier)")
    args = ap.parse_args()

    tiers = args.tiers or sorted(
        {d.name for b in args.benchmarks for d in (Path(args.pools) / b).glob("t*")}
    )
    for tier in tiers:
        files = [f for b in args.benchmarks for f in (Path(args.pools) / b / tier).glob("*.json")]
        pools = {f: json.loads(f.read_text()) for f in files}
        jpt = [c["extra"]["joules_per_token"] for p in pools.values() for c in cands(p)
               if c.get("extra", {}).get("joules_per_token")]  # fmt: skip
        if not jpt:
            print(f"{tier}: no measured energy yet, skipped")
            continue
        rate = st.median(jpt)
        filled = 0
        for f, p in pools.items():
            changed = False
            for c in cands(p):
                if "energy_j" not in c.setdefault("extra", {}):
                    c["extra"]["energy_j"] = c["tokens_out"] * rate
                    c["extra"]["joules_per_token"] = rate
                    c["extra"]["energy_estimated"] = True
                    filled += 1
                    changed = True
            if changed:
                f.write_text(json.dumps(p))
        print(
            f"{tier}: filled {filled} candidates at {rate:.3f} J per output token (median of measured)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
