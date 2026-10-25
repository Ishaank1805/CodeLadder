"""Print result tables from results/ (the numbers quoted in docs/RESULTS.md).

    python scripts/show_results.py runs <result dir>...        calibrated runs (evaluate)
    python scripts/show_results.py runs --baselines <dir>...   ... plus each model alone
    python scripts/show_results.py route <dir>...              X9 router runs (route)
    python scripts/show_results.py actions <dir>...            X4 action-policy runs (actions)
    python scripts/show_results.py ratings <bench>/<tier>...   AUROC of self-confidence ratings

Each calibrated cell is "solved share / mean cost (violation rate)". Directories may be given
with or without the results/ prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(d: str) -> dict | None:
    p = Path(d)
    if not p.exists():
        p = ROOT / "results" / d
    f = p / "summary.json"
    if not f.exists():
        print(f"missing {d}")
        return None
    return json.loads(f.read_text())


def _cell(c: dict) -> str:
    return (
        f"a={c['alpha']:g}: {c['mean_solved'] * 100:5.1f}% / {c['mean_test_cost'] / 1000:6.1f}k"
        f" (viol {c['violation_rate'] * 100:.0f}%)"
    )


def runs(dirs: list[str], baselines: bool) -> None:
    for d in dirs:
        s = _load(d)
        if not s:
            continue
        print(
            f"{Path(d).name} ({s['n_tasks']} tasks): "
            + "  ".join(_cell(c) for c in s["calibrated"])
        )
        for c in s.get("calibrated_grouped", []):
            print("    per group", _cell(c))
        if "transfer" in s:
            t = s["transfer"]
            print(
                f"    transfer {t['calibrate_on']} -> {t['test_on']}: "
                + "  ".join(
                    f"a={r['alpha']:g}: target risk {r['target_risk']:.3f}, "
                    f"viol {r['violation_rate']:.2f}, solved {r['mean_solved'] * 100:.1f}%"
                    for r in t["results"]
                )
            )
        for a in sorted({r["alpha"] for r in s.get("learning_curve", [])}):
            lc = [r for r in s["learning_curve"] if r["alpha"] == a]
            print(f"    learning curve a={a:g}: " + "  ".join(
                f"n={r['n_cal']}: {r['mean_solved'] * 100:.1f}%" for r in lc
            ))  # fmt: skip
        if baselines:
            for k, b in s["baselines"].items():
                if k.startswith("calibrated-only"):
                    print(f"    {k:24s}", "  ".join(_cell(c) for c in b))
                elif isinstance(b, dict) and "solved" in b:
                    extra = f", risk {b['risk']:.3f}" if "risk" in b else ""
                    print(
                        f"    {k:24s} solved {b['solved'] * 100:.1f}%, cost {b['cost'] / 1000:.1f}k{extra}"
                    )


def route(dirs: list[str]) -> None:
    for d in dirs:
        s = _load(d)
        if not s:
            continue
        print(f"{Path(d).name} ({s['n_tasks']} tasks, router trained on {s['router_frac']:.0%}, "
              f"{s['router_seeds']} seeds)")  # fmt: skip
        for seed in s["seeds"]:
            q = seed["router_quality"]
            print(f"    seed {seed['seed']} AUROC", " ".join(
                f"{t}={v['auroc_pass_majority']:.3f}" for t, v in q.items()
            ))  # fmt: skip
        for fam, rows in s["mean"].items():
            print(f"  {fam:14s}", "  ".join(_cell(r) for r in rows))
        for fam in ("router", "oracle_router"):
            for r in s["seeds"][0]["families"][fam]:
                if "routed_share" in r:
                    share = {k: round(v, 2) for k, v in r["routed_share"].items()}
                    print(f"    {fam} a={r['alpha']:g} routed share {share}")


def actions(dirs: list[str]) -> None:
    for d in dirs:
        s = _load(d)
        if not s:
            continue
        print(f"{Path(d).name} ({s['n_tasks']} tasks, family {s['family']})")
        for v, rows in s["mean"].items():
            print(f"  {v:17s}", "  ".join(_cell(r) for r in rows))
        sd = s["seeds"][0]
        for name, t in sd["tables"].items():
            print(f"    {name}: {t['contexts']} contexts; repair in {t['repair_contexts']}")
        st = sd["learned_hidden_stats"]
        coarse = [(k, v) for k, v in st.items() if k.count("|") == 1 and v["n"] >= 30]
        for k, v in sorted(coarse, key=lambda kv: -kv[1]["n"]):
            print(f"      {k:16s} n={v['n']:5d}  repair {v['repair'] * 100:5.1f}%  "
                  f"fresh, same problems {v['fresh'] * 100:5.1f}%")  # fmt: skip


def ratings(specs: list[str]) -> None:
    import numpy as np
    from scipy.stats import rankdata

    def auroc(score, label):
        y = np.asarray(label, bool)
        if y.all() or not y.any():
            return float("nan")
        r = rankdata(score)
        return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

    for spec in specs:
        bench, tier = spec.split("/")
        conf, old, y, missing = [], [], [], 0
        for f in sorted((ROOT / "data" / "pools" / bench / tier).glob("*.json")):
            d = json.loads(f.read_text())
            for c in d["samples"] + [c for ch in d["repairs"].values() for c in ch]:
                if c["visible_pass_rate"] < 1:
                    continue
                e = c.get("extra", {})
                missing += e.get("self_conf") is None
                conf.append(e.get("self_conf") or 0.0)
                old.append(e.get("self_conf_v1", e.get("self_conf")) or 0.0)
                y.append(c["hidden_pass"])
        if not y:
            print(f"{spec}: no visible-passing candidates")
            continue
        line = (
            f"{spec}: {len(y)} visible-passing, hidden-correct {np.mean(y):.2f}, "
            f"unrated {missing / len(y):.0%}, rating AUROC {auroc(conf, y):.3f}"
        )
        if any(a != b for a, b in zip(conf, old, strict=True)):
            line += f" (before prefill {auroc(old, y):.3f})"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs")
    r.add_argument("--baselines", action="store_true")
    r.add_argument("dirs", nargs="+")
    for name in ("route", "actions"):
        sub.add_parser(name).add_argument("dirs", nargs="+")
    sub.add_parser("ratings").add_argument("specs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "runs":
        runs(args.dirs, args.baselines)
    elif args.cmd == "route":
        route(args.dirs)
    elif args.cmd == "actions":
        actions(args.dirs)
    else:
        ratings(args.specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
