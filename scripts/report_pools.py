"""Sanity report over cached candidate pools.

For every benchmark and tier: how many tasks have pools, sample-level pass@1, share of
tasks with at least one correct sample, the silent-failure rate (passes visible tests but
fails hidden ones), repair success, and token / latency / energy statistics.

    python scripts/report_pools.py                       # all benchmarks and tiers found
    python scripts/report_pools.py --benchmarks humaneval mbpp --json results/pool_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarise(files: list[Path]) -> dict:
    n_tasks = len(files)
    samples = repairs = 0
    s_correct = s_vis = s_vis_wrong = 0
    any_correct = 0
    r_fixed_visible = r_fixed_hidden = 0
    tokens, latency, energy, jpt = [], [], [], []
    for f in files:
        p = json.loads(f.read_text())
        ss = p["samples"]
        rs = [c for ch in p["repairs"].values() for c in ch]
        samples += len(ss)
        repairs += len(rs)
        s_correct += sum(c["hidden_pass"] for c in ss)
        vis = [c for c in ss if c["visible_pass_rate"] >= 1.0]
        s_vis += len(vis)
        s_vis_wrong += sum(not c["hidden_pass"] for c in vis)
        any_correct += any(c["hidden_pass"] for c in ss)
        r_fixed_visible += sum(c["visible_pass_rate"] >= 1.0 for c in rs)
        r_fixed_hidden += sum(c["hidden_pass"] for c in rs)
        for c in ss + rs:
            tokens.append(c["tokens_out"])
            latency.append(c["latency_s"])
            if "energy_j" in c.get("extra", {}):
                energy.append(c["extra"]["energy_j"])
                jpt.append(c["extra"].get("joules_per_token", 0.0))
    return {
        "tasks": n_tasks,
        "samples": samples,
        "repairs": repairs,
        "pass@1": s_correct / max(samples, 1),
        "any_correct@n": any_correct / max(n_tasks, 1),
        "visible_pass_rate": s_vis / max(samples, 1),
        "silent_failure_rate": s_vis_wrong / max(s_vis, 1),
        "repair_fixes_visible": r_fixed_visible / max(repairs, 1),
        "repair_fixes_hidden": r_fixed_hidden / max(repairs, 1),
        "median_tokens_out": st.median(tokens) if tokens else 0,
        "median_latency_s": st.median(latency) if latency else 0,
        "energy_coverage": len(energy) / max(len(tokens), 1),
        "joules_per_token": st.median([j for j in jpt if j]) if any(jpt) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(ROOT / "data" / "pools"))
    ap.add_argument("--benchmarks", nargs="*")
    ap.add_argument("--json", help="also write the report as JSON")
    args = ap.parse_args()
    root = Path(args.pools)
    benches = args.benchmarks or sorted(d.name for d in root.iterdir() if d.is_dir())
    report = {}
    header = f"{'benchmark':14s} {'tier':5s} {'tasks':>5s} {'pass@1':>7s} {'any@n':>6s} {'vis-pass':>8s} {'silent':>7s} {'rep-fix':>7s} {'tok':>5s} {'lat s':>6s} {'J/tok':>6s}"
    print(header)
    for b in benches:
        for tier_dir in sorted((root / b).iterdir()) if (root / b).exists() else []:
            files = sorted(tier_dir.glob("*.json"))
            if not files:
                continue
            r = summarise(files)
            report[f"{b}/{tier_dir.name}"] = r
            jpt = f"{r['joules_per_token']:.2f}" if r["joules_per_token"] else "-"
            print(
                f"{b:14s} {tier_dir.name:5s} {r['tasks']:5d} {r['pass@1']:7.3f} {r['any_correct@n']:6.3f} "
                f"{r['visible_pass_rate']:8.3f} {r['silent_failure_rate']:7.3f} {r['repair_fixes_hidden']:7.3f} "
                f"{r['median_tokens_out']:5.0f} {r['median_latency_s']:6.2f} {jpt:>6s}"
            )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
