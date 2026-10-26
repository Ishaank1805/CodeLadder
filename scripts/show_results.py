"""Print result tables from results/ (the numbers quoted in docs/RESULTS.md).

    python scripts/show_results.py runs <result dir>...        calibrated runs (evaluate)
    python scripts/show_results.py runs --baselines <dir>...   ... plus each model alone
    python scripts/show_results.py route <dir>...              X9 router runs (route)
    python scripts/show_results.py actions <dir>...            X4 action-policy runs (actions)
    python scripts/show_results.py ratings <bench>/<tier>...   AUROC of self-ratings and P(True)
        [--json results/analysis/signals_auroc.json]           (also saved for the X12 figure)
    python scripts/show_results.py ci results/ci/*.json        X14 bootstrap comparisons
    python scripts/show_results.py economics <dir>...          X13 price sweeps
    python scripts/show_results.py guaranteed <dir>...         X16 guaranteed baselines

Each calibrated cell is "solved share / mean cost (violation rate)". Directories may be given
with or without the results/ prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from codeladder.results import find_result  # noqa: E402


def _load(d: str) -> dict | None:
    f = find_result(ROOT / "results", d) / "summary.json"
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


def ratings(specs: list[str], out: str | None = None) -> None:
    import numpy as np
    from scipy.stats import rankdata

    def auroc(score, label):
        y = np.asarray(label, bool)
        if y.all() or not y.any():
            return float("nan")
        r = rankdata(score)
        return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

    saved: dict[str, dict] = {}
    for spec in specs:
        bench, tier = spec.split("/")
        conf, old, pt, y, missing = [], [], [], [], 0
        for f in sorted((ROOT / "data" / "pools" / bench / tier).glob("*.json")):
            d = json.loads(f.read_text())
            for c in d["samples"] + [c for ch in d["repairs"].values() for c in ch]:
                if c["visible_pass_rate"] < 1:
                    continue
                e = c.get("extra", {})
                missing += e.get("self_conf") is None
                conf.append(e.get("self_conf") or 0.0)
                old.append(e.get("self_conf_v1", e.get("self_conf")) or 0.0)
                pt.append(e.get("p_true"))
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
        if any(v is not None for v in pt):
            have = [v is not None for v in pt]
            line += (
                f"; P(True) AUROC {auroc([v or 0.0 for v in pt], y):.3f}"
                f" ({sum(have) / len(have):.0%} with a value)"
            )
        print(line)
        have = [v is not None for v in pt]
        saved.setdefault(bench, {})[tier] = {
            "n": len(y),
            "hidden_correct": float(np.mean(y)),
            "unrated": missing / len(y),
            "rating_auroc": auroc(conf, y),
            "ptrue_auroc": auroc([v or 0.0 for v in pt], y) if any(have) else None,
            "ptrue_share": sum(have) / len(have),
        }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(saved, indent=1))
        print(f"wrote {out}")


def guaranteed(dirs: list[str]) -> None:
    for d in dirs:
        s = _load(d)
        if not s:
            continue
        print(f"{Path(d).name} ({s['n_tasks']} tasks)")
        for name, rows in s["methods"].items():
            cells = [
                _cell(r) + f" prec {r['precision'] * 100:.0f}%" if r.get("certified_splits")
                else f"a={r['alpha']:g}: none certified"
                for r in rows
            ]  # fmt: skip
            print(f"  {name:12s}", "  ".join(cells))


def economics(dirs: list[str]) -> None:
    for d in dirs:
        s = _load(d)
        if not s:
            continue
        print(f"{Path(d).name} ({s['n_tasks']} tasks)")
        for b in s["break_even"]:
            print(
                f"    a={b['alpha']:g}: cheap share {b['cheap_share']:.2f}, break-even ratio predicted"
                f" {b['predicted']:.1f} / measured {b['measured']}, cost vs top alone at the"
                f" largest ratio {b['asymptote']:.2f} (solved {b['solved_diff_at_max'] * 100:+.1f})"
            )
        for base in ("top", "selection"):
            for a in sorted({r["alpha"] for r in s["rows"]}):
                rows = [r for r in s["rows"] if r["alpha"] == a and r[base].get("certified_splits")]
                print(f"      vs {base:9s} a={a:g}: " + " ".join(
                    f"{r['ratio']:g}:{r[base]['cost_ratio']:.2f}" for r in rows
                ))  # fmt: skip


def ci(files: list[str]) -> None:
    for f in files:
        d = json.loads(Path(f).read_text())
        print(
            f"{Path(f).stem}: {d['a']}  minus  {d['b']} ({d['n_tasks']} tasks, {d['n_boot']} resamples)"
        )
        for r in d["alphas"]:
            lo, hi = r["solved_diff_ci"] or [float("nan")] * 2
            clo, chi = r["cost_ratio_ci"] or [float("nan")] * 2
            print(
                f"    a={r['alpha']:g}: solved {r['solved_a'] * 100:5.1f}% vs {r['solved_b'] * 100:5.1f}%"
                f" (diff CI {lo * 100:+.1f} to {hi * 100:+.1f}), cost ratio"
                f" {r['cost_a'] / r['cost_b']:.2f} (CI {clo:.2f} to {chi:.2f})"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs")
    r.add_argument("--baselines", action="store_true")
    r.add_argument("dirs", nargs="+")
    for name in ("route", "actions"):
        sub.add_parser(name).add_argument("dirs", nargs="+")
    rt = sub.add_parser("ratings")
    rt.add_argument("specs", nargs="+")
    rt.add_argument("--json", help="also save the AUROCs to this file")
    sub.add_parser("ci").add_argument("files", nargs="+")
    sub.add_parser("economics").add_argument("dirs", nargs="+")
    sub.add_parser("guaranteed").add_argument("dirs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "runs":
        runs(args.dirs, args.baselines)
    elif args.cmd == "route":
        route(args.dirs)
    elif args.cmd == "actions":
        actions(args.dirs)
    elif args.cmd == "ci":
        ci(args.files)
    elif args.cmd == "economics":
        economics(args.dirs)
    elif args.cmd == "guaranteed":
        guaranteed(args.dirs)
    else:
        ratings(args.specs, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
