"""Report tables from results/, as Markdown and LaTeX (booktabs).

    python scripts/report_tables.py results

Writes results/tables/<name>.md and <name>.tex; a table whose inputs are missing is skipped,
and benchmarks are picked up when their result directories exist (BigCodeBench included).
  main        CodeLadder against single models, guaranteed and published baselines (E1, X16)
  signals     AUROC of the verbalised rating and of P(True) per tier (X1, X12)
  economics   cheap share and ladder cost against the top model alone (X13)
  ci          paired bootstrap intervals (X14)
  oracle      certificate validity under weaker test oracles (X17)
Cells of the main table are "solved % / cost (k tokens) / splits above α (%)".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BENCH = {
    "humaneval+mbpp": "HumanEval+/MBPP+",
    "livecodebench": "LiveCodeBench",
    "bigcodebench": "BigCodeBench",
}
TIERS = {"t1": "1.5B", "t2": "7B", "t3": "14B", "t4": "32B", "d1": "DS 1.3B", "d2": "DS 6.7B",
         "d3": "DS 33B", "api": "120B API"}  # fmt: skip
ALPHAS = (0.05, 0.1, 0.2)
P = "resample-agreement_selfconf"
QWEN = "t1-t2-t3-t4"


def _load(root: Path, name: str) -> dict | None:
    p = root / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _at(rows: list[dict], alpha: float) -> dict | None:
    return next((r for r in rows if abs(r["alpha"] - alpha) < 1e-9), None)


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _write(out: Path, name: str, caption: str, header: list[str], rows: list[list[str]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    md = [f"{caption}\n", "| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")

    def tex(s: str) -> str:
        s = s.replace("\\", "\\textbackslash{}")
        for a, b in (("%", "\\%"), ("&", "\\&"), ("_", "\\_"), ("#", "\\#")):
            s = s.replace(a, b)
        return s.replace("α", "$\\alpha$").replace("δ", "$\\delta$").replace("−", "$-$")

    cols = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        f"\\caption{{{tex(caption)}}}",
        f"\\label{{tab:{name}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(tex(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        if r[0].startswith("**"):  # section row
            lines.append(
                f"\\multicolumn{{{len(header)}}}{{l}}{{\\textit{{{tex(r[0].strip('*'))}}}}} \\\\"
            )
        else:
            lines.append(" & ".join(tex(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/{name}.md and .tex")


def _cell(r: dict | None, solved="mean_solved", cost="mean_test_cost", viol="violation_rate",
          certified="certified_splits") -> str:  # fmt: skip
    if not r or not r.get(certified) or not _ok(r.get(solved)):
        return "infeasible"
    return f"{100 * r[solved]:.1f} / {r[cost] / 1000:.1f}k / {100 * r[viol]:.0f}"


def main_table(root: Path, out: Path) -> None:
    rows: list[list[str]] = []
    for bench, title in BENCH.items():
        sel = _load(root, f"{bench}-{QWEN}-{P}-singles-k64")
        if not sel:
            continue
        rows.append([f"**{title}, Qwen2.5-Coder 1.5B to 32B**", "", "", ""])
        g = _load(root, f"guaranteed-{bench}-{QWEN}-agreement_selfconf-k64")
        if g:
            best = []
            for a in ALPHAS:
                c = [(k, r) for k, v in g["methods"].items() if k.startswith("single-")
                     if (r := _at(v, a)) and r.get("certified_splits")]  # fmt: skip
                best.append(max(c, key=lambda kr: kr[1]["mean_solved"]) if c else None)
            rows.append(["best single model, calibrated abstention"]
                        + [f"{_cell(b[1])} ({TIERS[b[0][7:]]})" if b else "infeasible"
                           for b in best])  # fmt: skip
        rows.append(
            ["CodeLadder: model selection"] + [_cell(_at(sel["calibrated"], a)) for a in ALPHAS]
        )
        for fam, name in (("subsets", "sub-ladders"), ("two_level", "two-level ladder")):
            s = _load(root, f"{bench}-{QWEN}-{P}-{fam}-k64-lb1")
            if s:
                rows.append([f"CodeLadder: {name} (lean)"]
                            + [_cell(_at(s["calibrated"], a)) for a in ALPHAS])  # fmt: skip
        if g:
            for k, name in (("toe", "Trust or Escalate"), ("toe-lean", "Trust or Escalate (lean)")):
                rows.append([name] + [_cell(_at(g["methods"][k], a)) for a in ALPHAS])
        bp = root / f"baselines-{bench}-{QWEN}.json"
        if bp.exists():
            b = json.loads(bp.read_text())
            for k, name in (("mcc", "Model Cascading for Code (tuned)"),
                            ("bse", "Bayesian self-escalation (tuned)")):  # fmt: skip
                rows.append([name] + [_cell(_at(b[k]["calibrated_tuned"], a), certified="tuned_splits")
                                      for a in ALPHAS])  # fmt: skip
    if rows:
        _write(out, "main", "Solved % / mean cost (k tokens) / share of splits with true risk above "
               "α (%), δ = 10%, 100 calibration/test splits. Infeasible: no setting certified.",
               ["Method", "α = 0.05", "α = 0.10", "α = 0.20"], rows)  # fmt: skip


def signals(root: Path, out: Path) -> None:
    d: dict = {}
    for p in sorted(root.glob("signals_auroc*.json")):
        for b, v in json.loads(p.read_text()).items():
            d.setdefault(b, {}).update(v)
    if not d:
        return
    names = {"humaneval": "HumanEval+", "mbpp": "MBPP+", "livecodebench": "LiveCodeBench",
             "bigcodebench": "BigCodeBench"}  # fmt: skip
    tiers = [t for t in TIERS if any(t in d.get(b, {}) for b in names)]
    rows = []
    for b, title in names.items():
        if b not in d:
            continue
        for key, lab in (("rating_auroc", "rating"), ("ptrue_auroc", "P(True)")):
            vals = [d[b].get(t, {}).get(key) for t in tiers]
            rows.append([f"{title}, {lab}"] + [f"{v:.2f}" if _ok(v) else "pending" for v in vals])
    _write(out, "signals", "AUROC of each confidence signal for separating hidden-test-correct "
           "from wrong visible-passing programs (0.5 is chance).",
           ["Benchmark, signal"] + [TIERS[t] for t in tiers], rows)  # fmt: skip


def economics(root: Path, out: Path) -> None:
    rows = []
    for d in sorted(root.glob("economics-*-agreement_selfconf-k64-lb1")):
        bench = d.name.split("-")[1]
        if bench not in BENCH:
            continue
        s = json.loads((d / "summary.json").read_text())
        cheap, top = s["tiers"]
        for be in s["break_even"]:
            a = be["alpha"]
            ratios = {r["ratio"]: r for r in s["rows"] if abs(r["alpha"] - a) < 1e-9}
            vs = [ratios.get(x, {}).get("top", {}).get("cost_ratio") for x in (1, 4, 1024)]
            sel = ratios.get(1024, {}).get("selection", {}).get("cost_ratio")
            rows.append([f"{BENCH[bench]}, {TIERS[cheap]} to {TIERS[top]}", f"{a:g}",
                         f"{be['cheap_share']:.2f}"]
                        + [f"{v:.2f}" if _ok(v) else "n/a" for v in vs]
                        + [f"{sel:.2f}" if _ok(sel) else "n/a"])  # fmt: skip
    if rows:
        _write(out, "economics", "Two-rung ladders: share of tasks the cheap rung certifies, and "
               "cost of the cheapest matched-coverage ladder relative to the top model alone at "
               "price ratios 1, 4 and 1024 (and to model selection at 1024).",
               ["Ladder", "α", "cheap share", "vs top 1x", "vs top 4x", "vs top 1024x",
                "vs selection 1024x"], rows)  # fmt: skip


CI = [
    ("ladders-vs-selection-evalplus", "sub-ladders (lean) − selection, HumanEval+/MBPP+"),
    ("api-ladders-vs-selection-evalplus", "API sub-ladders (lean) − selection, HumanEval+/MBPP+"),
    ("ladders-vs-selection-lcb", "sub-ladders (lean) − selection, LiveCodeBench"),
    ("twolevel-vs-selection-lcb", "two-level ladder − selection, LiveCodeBench"),
    ("ladders-vs-selection-bcb", "sub-ladders (lean) − selection, BigCodeBench"),
    ("all7-vs-qwen-selection-evalplus", "7 models − 4 Qwen, selection, HumanEval+/MBPP+"),
    ("all7-vs-qwen-selection-lcb", "7 models − 4 Qwen, selection, LiveCodeBench"),
    ("all7-vs-qwen-selection-bcb", "7 models − 4 Qwen, selection, BigCodeBench"),
    ("ptrue-vs-selfconf-selection-bcb", "P(True) − rating, selection, BigCodeBench"),
    ("pareto-vs-paths-lcb", "Pareto Testing − Bonferroni paths, LiveCodeBench"),
    ("lean-vs-full-budget-lcb", "lean − full lower rungs, LiveCodeBench"),
    ("selfconf-vs-agreement-32b", "32B: agreement × rating − agreement, HumanEval+/MBPP+"),
]


def ci(root: Path, out: Path) -> None:
    rows = []
    for name, lab in CI:
        p = root / "ci" / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cells = []
        for a in ALPHAS:
            r = _at(d["alphas"], a)
            if not r or not r.get("solved_diff_ci"):
                cells.append("n/a")
                continue
            lo, hi = r["solved_diff_ci"]
            c = f"{100 * (r['solved_a'] - r['solved_b']):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            if a == ALPHAS[-1] and r.get("cost_ratio_ci") and r["cost_b"]:
                clo, chi = r["cost_ratio_ci"]
                c += f"; cost {r['cost_a'] / r['cost_b']:.2f} [{clo:.2f}, {chi:.2f}]"
            cells.append(c)
        rows.append([lab] + cells)
    if rows:
        _write(out, "ci", "Paired task-level bootstrap (200 resamples, 20 splits each): solved "
               "share A − B in points with 95% intervals; cost ratio A / B at α = 0.20.",
               ["Comparison (A − B)", "α = 0.05", "α = 0.10", "α = 0.20"], rows)  # fmt: skip


def oracle(root: Path, out: Path) -> None:
    settings = [
        ("Qwen 32B alone", "oracle-humaneval+mbpp-t4-agreement_selfconf-shared-k64"),
        ("Qwen 1.5B alone", "oracle-humaneval+mbpp-t1-agreement_selfconf-shared-k64"),
        (
            "model selection, 4 tiers",
            f"oracle-humaneval+mbpp-{QWEN}-agreement_selfconf-singles-k64",
        ),
        ("sub-ladders, 4 tiers", f"oracle-humaneval+mbpp-{QWEN}-agreement_selfconf-subsets-k64"),
    ]
    rows, fr = [], None
    for lab, n in settings:
        s = _load(root, n)
        if not s:
            continue
        for a in (0.05, 0.1):
            rs = {r["fraction"]: r for r in s["rows"] if abs(r["alpha"] - a) < 1e-9}
            fr = sorted(rs)
            rows.append([f"{lab}, α = {a:g}"] + [
                f"{rs[f]['claimed_risk_weak']:.3f} / {rs[f]['true_risk']:.3f} / "
                f"{100 * rs[f]['violation_rate_true']:.0f}%" for f in fr])  # fmt: skip
    if rows:
        head = ["original tests" if f == 0 else ("all tests" if f == 1 else f"{f:.0%} of tests")
                for f in fr]  # fmt: skip
        _write(out, "oracle", "HumanEval+/MBPP+: calibrated on labels from a fraction of the "
               "hidden tests. Claimed risk / true risk (all hidden tests) / splits with true "
               "risk above α.", ["Setting"] + head, rows)  # fmt: skip


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = root / "tables"
    for f in (main_table, signals, economics, ci, oracle):
        f(root, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
