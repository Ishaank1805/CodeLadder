"""Report figures from evaluate / baselines / analysis outputs in results/.

    python scripts/figures/report_figures.py results

Writes results/figures/fig_<name>.{pdf,png}; a figure whose inputs are missing is skipped.
  validity   realised test risk of every calibrated run against its target alpha
  baselines  share of splits whose true risk exceeds alpha: CodeLadder against published baselines
  shift      LiveCodeBench risk per release-date window: calibrated once versus sliding
  repair     repair versus a fresh sample after a wrong answer, per tier
  energy     solved share against joules per task (alpha = 0.10)
  stdin      AUROC of stdin signals per tier (X2)
  conditional  per-difficulty risk of pooled calibration (X7)
  router     learned and oracle up-front routers against model selection and sub-ladders (X9)
  economics  ladder cost against the top model alone across price ratios, and the cheap-share law (X13)
  guaranteed CodeLadder against guaranteed baselines: solved share and cost per benchmark (X16)
  ci         paired bootstrap intervals for the headline comparisons (X14)
  oracle     certificate validity when labels come from a weaker test oracle (X17)
  ptrue      AUROC of the verbalised rating against P(True), per tier and benchmark (X12)

Benchmarks are picked up when their result directories exist (BigCodeBench included).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import AQUA, BLUE, INK, INK2, ORANGE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from codeladder.results import find_result, iter_runs  # noqa: E402

YELLOW = "#eda100"  # categorical slot 4 of the reference palette (validated with slots 1 to 3)
TIERS = {
    "t1": "1.5B",
    "t2": "7B",
    "t3": "14B",
    "t4": "32B",
    "d1": "DS 1.3B",
    "d2": "DS 6.7B",
    "d3": "DS 33B",
}
P = "resample-agreement_selfconf"
BENCH = {
    "humaneval+mbpp": "HumanEval+/MBPP+",
    "livecodebench": "LiveCodeBench",
    "bigcodebench": "BigCodeBench",
}
QWEN = "t1-t2-t3-t4"


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out}/fig_{name}.pdf")


def _load(root: Path, name: str) -> dict | None:
    p = find_result(root, name) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def validity(root: Path, out: Path) -> None:
    pts = []
    for d in iter_runs(root):
        s = json.loads((d / "summary.json").read_text())
        for c in s.get("calibrated", []):
            if c.get("certified_splits") and np.isfinite(c.get("mean_test_risk", np.nan)):
                pts.append((c["alpha"], c["mean_test_risk"], c["violation_rate"]))
    if not pts:
        return
    a = np.array(pts)
    alphas = sorted(set(a[:, 0]))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    rng = np.random.default_rng(0)
    labels = []
    for i, al in enumerate(alphas):
        r = a[a[:, 0] == al]
        ax.scatter(
            i + rng.uniform(-0.2, 0.2, len(r)),
            r[:, 1],
            s=9,
            color=BLUE,
            alpha=0.5,
            linewidths=0,
            zorder=3,
        )
        ax.plot([i - 0.34, i + 0.34], [al, al], color=INK, lw=1.4, ls="--", zorder=4)
        labels.append(f"α = {al:g}\n{len(r)} runs\nmax {r[:, 2].max():.0%} > α")
    ax.set_xticks(range(len(alphas)), labels, fontsize=7.5)
    ax.set_xlim(-0.6, len(alphas) - 0.4)
    ax.set_ylim(0, max(alphas) * 1.2)
    ax.set_ylabel("mean realised risk (test halves)")
    ax.set_title("Every calibrated run stays below its target α (dashed)")
    fig.tight_layout()
    _save(fig, out, "validity")


def baselines(root: Path, out: Path) -> None:
    rows = []  # (label, share of splits with true risk above alpha, solved, ours?)
    for tiers, label in (("t1-t2-t3-t4", "4 tiers"), ("t2-t3", "7B to 14B")):
        for fam, name in (
            ("singles", "model selection"),
            ("subsets", "sub-ladders"),
            ("two_level", "two-level ladder"),
        ):
            s = _load(root, f"humaneval+mbpp-{tiers}-{P}-{fam}-k64") or _load(
                root, f"humaneval+mbpp-{tiers}-{P}-{fam}-k4"
            )
            if s:
                c = next(c for c in s["calibrated"] if abs(c["alpha"] - 0.2) < 1e-9)
                rows.append(
                    (f"CodeLadder {name}, {label}", c["violation_rate"], c["mean_solved"], True)
                )
        bp = find_result(root, f"baselines-humaneval+mbpp-{tiers}.json")
        if bp.exists():
            b = json.loads(bp.read_text())
            for k, name in (
                ("mcc", "Model Cascading for Code"),
                ("bse", "Bayesian self-escalation"),
            ):
                r = next(r for r in b[k]["calibrated_tuned"] if abs(r["alpha"] - 0.2) < 1e-9)
                if r["tuned_splits"]:
                    rows.append((f"{name}, {label}", r["violation_rate"], r["mean_solved"], False))
    if not rows:
        return
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 0.9))
    y = np.arange(len(rows))
    ax.barh(
        y,
        [max(r[1], 0.004) for r in rows],
        height=0.6,
        color=[BLUE if r[3] else ORANGE for r in rows],
        zorder=3,
    )
    for yi, r in zip(y, rows, strict=True):
        ax.text(
            max(r[1], 0.1) + 0.015,
            yi,
            f"{r[1]:.0%} of splits (solves {r[2]:.0%})",
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.axvline(0.1, color=INK, lw=1, ls="--", zorder=4)
    ax.text(0.105, len(rows) - 0.4, "δ = 10% allowed", fontsize=7, color=INK2, va="bottom")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("share of calibration/test splits whose true risk exceeds α")
    ax.set_title(
        "HumanEval+/MBPP+ at α = 0.20: CodeLadder (blue) and tuned published baselines (orange)",
        fontsize=8.5,
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "baselines")


def shift(root: Path, out: Path) -> None:
    p = find_result(root, f"livecodebench-t1-t2-t3-t4-{P}-shared-shift") / "shift.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())
    fig, axes = plt.subplots(1, len(res), figsize=(7.0, 2.4), sharey=False)
    for ax, r in zip(axes, res, strict=True):
        w = np.arange(1, len(r["windows"]) + 1)
        ax.plot(
            w,
            [x["fixed"]["risk"] for x in r["windows"]],
            color=BLUE,
            lw=2,
            marker="o",
            ms=5,
            mec="white",
            label="calibrated once (oldest window)",
        )
        ax.plot(
            w,
            [x["sliding"]["risk"] for x in r["windows"]],
            color=ORANGE,
            lw=2,
            marker="o",
            ms=5,
            mec="white",
            label="recalibrated on previous window",
        )
        ax.axhline(r["alpha"], color=INK, lw=1, ls="--")
        ax.set_title(f"α = {r['alpha']:g}")
        ax.set_xticks(w)
        ax.set_ylim(0, max(r["alpha"] * 1.4, max(x["sliding"]["risk"] for x in r["windows"]) * 1.1))
    axes[0].set_ylabel("accepted-but-wrong rate")
    fig.supxlabel("later release-date window (LiveCodeBench)", fontsize=8.5, color=INK2, y=0.02)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    _save(fig, out, "shift")


def repair(root: Path, out: Path) -> None:
    panels = [
        (find_result(root, f"e4_repair_{k}.json"), t)
        for k, t in (("evalplus", "HumanEval+/MBPP+"), ("lcb", "LiveCodeBench"))
    ]
    panels = [(p, t) for p, t in panels if p.exists()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.4))
    axes = np.atleast_1d(axes)
    for ax, (p, title) in zip(axes, panels, strict=True):
        d = json.loads(p.read_text())
        rows = d if isinstance(d, list) else d.get("rows", [])
        tiers = [t for t in ("t1", "t2", "t3", "t4") if any(r.get("tier") == t for r in rows)]
        get = {(r["tier"], r["failure"]): r for r in rows if "failure" in r}
        x = np.arange(len(tiers))
        rep = [get.get((t, "wrong_answer"), {}).get("repair_success", np.nan) for t in tiers]
        fre = [get.get((t, "wrong_answer"), {}).get("fresh_success", np.nan) for t in tiers]
        ax.bar(x - 0.17, rep, width=0.32, color=BLUE, label="repair round", zorder=3)
        ax.bar(x + 0.17, fre, width=0.32, color=ORANGE, label="fresh sample", zorder=3)
        ax.set_xticks(x, [TIERS[t] for t in tiers])
        ax.set_title(f"{title}: after a wrong answer")
        ax.set_ylabel("hidden-test success")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "repair")


def energy(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    drawn = False
    for ax, bench, title in zip(
        axes,
        ("humaneval+mbpp", "livecodebench"),
        ("HumanEval+/MBPP+", "LiveCodeBench"),
        strict=True,
    ):
        pts = []
        for t in ("t1", "t2", "t3", "t4"):
            s = _load(root, f"{bench}-{t}-{P}-shared-k64-energy")
            if s:
                c = next(c for c in s["calibrated"] if abs(c["alpha"] - 0.1) < 1e-9)
                pts.append(
                    (c["mean_test_cost"] / 1000, c["mean_solved"], TIERS[t] + " alone", BLUE)
                )
        for fam, name, col in (
            ("singles", "selection", ORANGE),
            ("subsets", "sub-ladder (lean)", AQUA),
            ("two_level", "two-level (lean)", YELLOW),
        ):
            s = _load(
                root,
                f"{bench}-t1-t2-t3-t4-{P}-{fam}-k64{'' if fam == 'singles' else '-lb1'}-energy",
            )
            if s:
                c = next(c for c in s["calibrated"] if abs(c["alpha"] - 0.1) < 1e-9)
                pts.append((c["mean_test_cost"] / 1000, c["mean_solved"], name, col))
        for x, y, lab, col in pts:
            single = col == BLUE  # single models carry direct labels; ladder families the legend
            ax.scatter(x, y, s=40, color=col, edgecolors="white", linewidths=1.2, zorder=3,
                       label=None if single else lab)  # fmt: skip
            if single:
                ax.annotate(
                    lab,
                    (x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.5,
                    color=INK2,
                )
        ax.set_title(f"{title}, α = 0.10")
        ax.set_xlabel("kJ per task")
        ax.margins(x=0.12, y=0.14)
        drawn |= bool(pts)
    axes[0].set_ylabel("share of tasks solved")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"4 tiers, {x}" for x in lab], loc="lower center", ncol=3, fontsize=7,
               frameon=False)  # fmt: skip
    if drawn:
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        _save(fig, out, "energy")
    else:
        plt.close(fig)


def stdin(root: Path, out: Path) -> None:
    p = find_result(root, "x2_stdin_signals.json")
    if not p.exists():
        return
    d = json.loads(p.read_text())
    tiers = [t for t in TIERS if t in d]
    series = [
        ("visible", "visible-test share", BLUE),
        ("probes", "mutated public inputs", ORANGE),
        ("rule", "stress inputs, timeout = fail", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = np.arange(len(tiers))
    wdt = 0.26
    for k, (key, lab, col) in enumerate(series):
        vals = [d[t]["auroc"][key] for t in tiers]
        ax.bar(x + (k - 1) * wdt, vals, width=wdt - 0.03, color=col, label=lab, zorder=3)
        if key == "rule":
            for xi, v in zip(x, vals, strict=True):
                ax.text(
                    xi + wdt,
                    v + 0.005,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK,
                )
    ax.set_xticks(x, [TIERS[t] + ("*" if t in ("t1", "t2") else "") for t in tiers])
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("AUROC (stdin problems)")
    ax.set_title(
        "Separating correct from wrong visible-passing programs (* rule chosen on these tiers)"
    )
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    _save(fig, out, "stdin")


def conditional(root: Path, out: Path) -> None:
    """X7: per-difficulty risk of one pooled calibration against per-difficulty calibration."""
    s = _load(root, f"livecodebench-t4-{P}-shared-k64-diff")
    if not s or "conditional_risk" not in s:
        return
    groups = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, len(s["conditional_risk"]), figsize=(7.0, 2.4))
    for ax, cr in zip(axes, s["conditional_risk"], strict=True):
        a = cr["alpha"]
        vals = [cr["groups"].get(f"livecodebench:{g}", {}).get("mean_risk", np.nan) for g in groups]
        x = np.arange(len(groups))
        ax.bar(x, vals, width=0.55, color=BLUE, zorder=3)
        top = max(a * 1.9, np.nanmax(vals) * 1.6)
        for xi, v, g in zip(x, vals, groups, strict=True):
            exc = cr["groups"].get(f"livecodebench:{g}", {}).get("exceed_rate", np.nan)
            ax.text(xi, top * 0.97, f"risk {v:.3f}\n> α in {exc:.0%}", ha="center", va="top",
                    fontsize=6.5, color=INK)  # fmt: skip
        ax.axhline(a, color=INK, lw=1, ls="--")
        ax.set_xticks(x, groups)
        ax.set_title(f"α = {a:g}")
        ax.set_ylim(0, top)
    axes[0].set_ylabel("true risk within group")
    worst = max(
        (g["violation_rate"] for g in s.get("calibrated_grouped", [])), default=float("nan")
    )
    fig.suptitle(
        "Qwen 32B calibrated once on all of LiveCodeBench: true risk by difficulty (dashed: α)\n"
        f"calibrating each difficulty separately: at most {worst:.0%} of splits above α",
        fontsize=8.5,
    )
    fig.tight_layout()
    _save(fig, out, "conditional")


def router(root: Path, out: Path) -> None:
    """X9: solved share (top) and cost relative to model selection (bottom) per family."""
    runs = [
        ("HumanEval+/MBPP+\nQwen 1.5B to 32B", "route-humaneval+mbpp-t1-t2-t3-t4"),
        ("LiveCodeBench\nQwen 1.5B to 32B", "route-livecodebench-t1-t2-t3-t4"),
        ("HumanEval+/MBPP+\nQwen + 120B API", "route-humaneval+mbpp-t1-t2-t3-t4-api"),
        ("BigCodeBench\nQwen 1.5B to 32B", "route-bigcodebench-t1-t2-t3-t4"),
    ]
    runs = [(t, s) for t, n in runs if (s := _load(root, f"{n}-agreement_selfconf-k64"))]
    if not runs:
        return
    fams = [
        ("singles", "model selection", BLUE),
        ("subsets", "sub-ladders", AQUA),
        ("router", "learned router", ORANGE),
        ("oracle_router", "oracle router (reads labels)", YELLOW),
    ]
    fig, axes = plt.subplots(2, len(runs), figsize=(7.0, 3.8), sharex=True, squeeze=False)
    w = 0.2
    for col, (title, s) in enumerate(runs):
        base = {r["alpha"]: r["mean_test_cost"] for r in s["mean"]["singles"]}
        alphas = [r["alpha"] for r in s["mean"]["singles"]]
        x = np.arange(len(alphas))
        for k, (key, label, color) in enumerate(fams):
            rows = {r["alpha"]: r for r in s["mean"].get(key, [])}
            if not rows:
                continue
            xs = x + (k - 1.5) * w
            solved = [rows[a]["mean_solved"] * 100 for a in alphas]
            rel = [rows[a]["mean_test_cost"] / base[a] for a in alphas]
            axes[0, col].bar(xs, solved, w * 0.9, color=color, label=label, zorder=3)
            axes[1, col].bar(xs, rel, w * 0.9, color=color, zorder=3)
        axes[1, col].axhline(1.0, color=INK, lw=0.8, ls="--")
        axes[0, col].set_title(title, fontsize=8)
        axes[1, col].set_xticks(x, [f"α={a:g}" for a in alphas])
    axes[0, 0].set_ylabel("solved (%)")
    axes[1, 0].set_ylabel("cost / model selection")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "router")


def economics(root: Path, out: Path) -> None:
    """X13: cost of the cheapest matched-coverage ladder relative to the top model alone."""
    real = {
        "t1-t4": 21.3,
        "t2-t4": 4.6,
        "t3-t4": 2.3,
        "t1-t3": 9.3,
        "d1-t4": 24.6,
        "t1-api": 80.0,
        "t3-api": 8.6,
    }  # fmt: skip  (real price ratios of the pairs, from tokens and model size)
    label = {"t1": "1.5B", "t2": "7B", "t3": "14B", "t4": "32B", "d1": "DS 1.3B", "api": "120B"}
    runs = []
    for d in iter_runs(root, "economics-*-agreement_selfconf-k64-lb1"):
        bench = d.name.split("-")[1]
        if bench in BENCH:
            runs.append((bench, "-".join(json.loads((d / "summary.json").read_text())["tiers"]),
                         json.loads((d / "summary.json").read_text())))  # fmt: skip
    benches = [b for b in BENCH if any(r[0] == b for r in runs)]
    if not benches:
        return
    colors = [BLUE, ORANGE, AQUA, YELLOW, INK2, "#7a5195", "#ef5675"]
    marks = {b: (m, c) for b, m, c in zip(BENCH, "os^", (BLUE, ORANGE, AQUA), strict=True)}
    fig, axes = plt.subplots(1, len(benches) + 1, figsize=(2.35 * (len(benches) + 1), 2.6))
    for ax, bench in zip(axes[:-1], benches, strict=True):
        k = 0
        for b, pair, s in runs:
            if b != bench:
                continue
            rows = [r for r in s["rows"] if r["alpha"] == 0.2 and r["top"].get("certified_splits")]
            if not rows:
                continue
            x = [r["ratio"] for r in rows]
            y = [r["top"]["cost_ratio"] for r in rows]
            c = colors[k % len(colors)]
            a, t = pair.split("-")
            ax.plot(x, y, color=c, lw=1.5, label=f"{label[a]} to {label[t]}")
            if pair in real:
                ax.plot(
                    [real[pair]], [np.interp(np.log(real[pair]), np.log(x), y)], "o", color=c, ms=5
                )
            k += 1
        ax.axhline(1.0, color=INK, lw=0.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.45)  # headroom above 1 for the legend
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(f"{BENCH[bench]}, α = 0.2", fontsize=8)
        ax.set_xlabel("price ratio top / cheap")
        # crowded panels put the legend in the headroom; others low, where curves stay near 1
        crowded = k > 5
        ax.legend(
            fontsize=5.5,
            frameon=False,
            loc="upper right" if crowded else "lower left",
            ncol=2 if crowded else 1,
            columnspacing=0.8,
            handlelength=1.2,
        )
    axes[0].set_ylabel("ladder cost / top model alone")
    ax = axes[-1]
    for bench in benches:
        mk, col = marks[bench]
        pts = [(be["cheap_share"], be["asymptote"]) for b, _p, s in runs if b == bench
               for be in s["break_even"]]  # fmt: skip
        ax.plot(*zip(*pts, strict=True), mk, color=col, ms=4, ls="none", label=BENCH[bench])
    ax.plot([0, 1], [1, 0], color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("share answered by the cheap model")
    ax.set_ylabel("cost / top alone at high prices")
    ax.set_title("saving is bounded by the cheap share", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "economics")


def _at(rows: list[dict], alpha: float) -> dict | None:
    return next((r for r in rows if abs(r["alpha"] - alpha) < 1e-9), None)


def guaranteed(root: Path, out: Path) -> None:
    """X16: CodeLadder against per-model abstention and Trust or Escalate, same splits."""
    alphas = (0.05, 0.1, 0.2)
    panels = []
    for bench in BENCH:
        g = _load(root, f"guaranteed-{bench}-{QWEN}-agreement_selfconf-k64")
        sel = _load(root, f"{bench}-{QWEN}-{P}-singles-k64")
        sub = _load(root, f"{bench}-{QWEN}-{P}-subsets-k64-lb1")
        if not (g and sel and sub):
            continue
        singles = [v for k, v in g["methods"].items() if k.startswith("single-")]
        best = []
        for a in alphas:
            cands = [r for rows in singles if (r := _at(rows, a)) and r.get("certified_splits")]
            best.append(max(cands, key=lambda r: r["mean_solved"]) if cands else None)
        methods = [
            ("CodeLadder: model selection", BLUE, [_at(sel["calibrated"], a) for a in alphas]),
            ("CodeLadder: sub-ladders (lean)", AQUA, [_at(sub["calibrated"], a) for a in alphas]),
            ("best single model with abstention", YELLOW, best),
            (
                "Trust or Escalate (lean)",
                ORANGE,
                [_at(g["methods"]["toe-lean"], a) for a in alphas],
            ),
        ]
        panels.append((BENCH[bench], methods))
    if not panels:
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(2.4 * len(panels) + 0.4, 3.8), sharex=True,
                             squeeze=False)  # fmt: skip
    w = 0.2
    x = np.arange(len(alphas))
    for col, (title, methods) in enumerate(panels):
        base = [r["mean_test_cost"] if r else np.nan for r in methods[0][2]]
        for k, (label, color, rows) in enumerate(methods):
            xs = x + (k - 1.5) * w
            solved = [r["mean_solved"] * 100 if r else np.nan for r in rows]
            rel = [
                r["mean_test_cost"] / b if r else np.nan for r, b in zip(rows, base, strict=True)
            ]
            axes[0, col].bar(xs, solved, w * 0.9, color=color, label=label, zorder=3)
            axes[1, col].bar(xs, rel, w * 0.9, color=color, zorder=3)
        axes[1, col].axhline(1.0, color=INK, lw=0.8, ls="--")
        axes[0, col].set_title(title, fontsize=8)
        axes[1, col].set_xticks(x, [f"α={a:g}" for a in alphas])
    axes[0, 0].set_ylabel("solved (%)")
    axes[1, 0].set_ylabel("cost / model selection")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Qwen2.5-Coder 1.5B to 32B, same 100 calibration/test splits", fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, out, "guaranteed")


CI_LABELS = [
    ("ladders-vs-selection-evalplus", "sub-ladders (lean) − selection, HE/MBPP+"),
    ("api-ladders-vs-selection-evalplus", "API sub-ladders (lean) − selection, HE/MBPP+"),
    ("ladders-vs-selection-lcb", "sub-ladders (lean) − selection, LCB"),
    ("twolevel-vs-selection-lcb", "two-level ladder − selection, LCB"),
    ("ladders-vs-selection-bcb", "sub-ladders (lean) − selection, BCB"),
    ("all7-vs-qwen-selection-evalplus", "7 models − 4 Qwen, selection, HE/MBPP+"),
    ("all7-vs-qwen-selection-lcb", "7 models − 4 Qwen, selection, LCB"),
    ("all7-vs-qwen-selection-bcb", "7 models − 4 Qwen, selection, BCB"),
    ("ptrue-vs-selfconf-selection-bcb", "P(True) − rating, selection, BCB"),
    ("pareto-vs-paths-lcb", "Pareto Testing − Bonferroni paths, LCB"),
    ("lean-vs-full-budget-lcb", "lean − full lower rungs, LCB"),
    ("selfconf-vs-agreement-32b", "32B: + self-confidence − agreement only"),
]


def ci(root: Path, out: Path) -> None:
    """X14: 95% bootstrap intervals of the solved-share difference and the cost ratio."""
    rows = [(lab, json.loads(p.read_text())) for name, lab in CI_LABELS
            if (p := root / "ci" / f"{name}.json").exists()]  # fmt: skip
    if not rows:
        return
    alphas = [a["alpha"] for a in rows[0][1]["alphas"]]
    fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(7.0, 0.28 * len(rows) + 1.1),
                             sharey=True)  # fmt: skip
    y = np.arange(len(rows))[::-1]
    for ax, a in zip(axes[:-1], alphas, strict=True):
        for yi, (_lab, d) in zip(y, rows, strict=True):
            r = _at(d["alphas"], a)
            if not r or not r.get("solved_diff_ci"):
                continue
            lo, hi = (100 * v for v in r["solved_diff_ci"])
            pt = 100 * (r["solved_a"] - r["solved_b"])
            col = INK2 if lo <= 0 <= hi else (BLUE if lo > 0 else ORANGE)
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(pt, yi, "o", color=col, ms=4)
        ax.axvline(0, color=INK, lw=0.8, ls="--")
        ax.set_title(f"α = {a:g}", fontsize=8)
        ax.set_xlabel("solved, A − B (points)", fontsize=7)
    ax = axes[-1]
    a = alphas[-1]
    for yi, (_lab, d) in zip(y, rows, strict=True):
        r = _at(d["alphas"], a)
        if r and r.get("cost_ratio_ci") and r["cost_b"]:
            lo, hi = r["cost_ratio_ci"]
            col = INK2 if lo <= 1 <= hi else (BLUE if hi < 1 else ORANGE)
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(r["cost_a"] / r["cost_b"], yi, "o", color=col, ms=4)
    ax.axvline(1, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_title(f"α = {a:g}", fontsize=8)
    ax.set_xlabel("cost ratio A / B", fontsize=7)
    axes[0].set_yticks(y, [lab for lab, _ in rows], fontsize=6.5)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("95% bootstrap intervals (blue: A significantly better, orange: worse, grey: "
                 "no significant difference)", fontsize=7.5)  # fmt: skip
    fig.tight_layout()
    _save(fig, out, "ci")


def oracle(root: Path, out: Path) -> None:
    """X17: labels from a fraction of the hidden tests; true risk measured on all of them."""
    settings = [
        ("Qwen 32B alone", "oracle-humaneval+mbpp-t4-agreement_selfconf-shared-k64", BLUE),
        ("Qwen 1.5B alone", "oracle-humaneval+mbpp-t1-agreement_selfconf-shared-k64", AQUA),
        ("model selection, 4 tiers",
         f"oracle-humaneval+mbpp-{QWEN}-agreement_selfconf-singles-k64", YELLOW),
        ("sub-ladders, 4 tiers",
         f"oracle-humaneval+mbpp-{QWEN}-agreement_selfconf-subsets-k64", ORANGE),
    ]  # fmt: skip
    settings = [(lab, s, c) for lab, n, c in settings if (s := _load(root, n))]
    if not settings:
        return
    alpha = 0.1
    fr = sorted({r["fraction"] for r in settings[0][1]["rows"]})
    names = ["original\ntests" if f == 0 else ("all" if f == 1 else f"{f:.0%}") for f in fr]
    x = np.arange(len(fr))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for lab, s, col in settings:
        rows = {r["fraction"]: r for r in s["rows"] if abs(r["alpha"] - alpha) < 1e-9}
        axes[0].plot(x, [rows[f]["violation_rate_true"] for f in fr], color=col, lw=1.8,
                     marker="o", ms=4, mec="white", label=lab)  # fmt: skip
        if lab.startswith("Qwen 32B"):
            axes[1].plot(x, [rows[f]["true_risk"] for f in fr], color=col, lw=1.8, marker="o",
                         ms=4, mec="white", label="true risk (all hidden tests)")  # fmt: skip
            axes[1].plot(x, [rows[f]["claimed_risk_weak"] for f in fr], color=col, lw=1.4,
                         ls=":", marker="o", ms=4, mfc="white", label="risk seen on the weak labels")  # fmt: skip
    axes[0].axhline(0.1, color=INK, lw=0.8, ls="--")
    axes[0].text(len(fr) - 1, 0.115, "δ = 10%", ha="right", fontsize=6.5, color=INK2)
    axes[0].set_ylabel("splits with true risk > α")
    axes[0].set_title(f"guarantee at α = {alpha:g}", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    axes[1].axhline(alpha, color=INK, lw=0.8, ls="--")
    axes[1].set_ylabel("risk of accepted programs")
    axes[1].set_title(f"Qwen 32B alone, α = {alpha:g}", fontsize=8)
    axes[1].set_ylim(0, 0.14)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xticks(x, names, fontsize=7)
        ax.set_xlabel("hidden tests used for calibration labels", fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, "oracle")


def ptrue(root: Path, out: Path) -> None:
    """X12: how well each confidence signal separates correct from wrong visible-passing code."""
    d: dict = {}
    for p in sorted(
        [*(root / "analysis").glob("signals_auroc*.json"), *root.glob("signals_auroc*.json")]
    ):  # one file per batch (ptrue.sh, bigcodebench.sh)
        for b, v in json.loads(p.read_text()).items():
            d.setdefault(b, {}).update(v)
    if not d:
        return
    names = {"humaneval": "HumanEval+", "mbpp": "MBPP+", "livecodebench": "LiveCodeBench",
             "bigcodebench": "BigCodeBench"}  # fmt: skip
    benches = [b for b in names if b in d]
    fig, axes = plt.subplots(1, len(benches), figsize=(7.0 * max(1, len(benches) / 3), 2.5),
                             sharey=True, squeeze=False)  # fmt: skip
    for ax, b in zip(axes[0], benches, strict=True):
        tiers = [t for t in TIERS if t in d[b]]
        x = np.arange(len(tiers))
        for k, (key, lab, col) in enumerate(
            (("rating_auroc", "verbalised rating (0 to 100)", BLUE),
             ("ptrue_auroc", "P(True)", ORANGE))
        ):  # fmt: skip
            vals = [d[b][t].get(key) for t in tiers]
            ax.bar(x + (k - 0.5) * 0.38, [np.nan if v is None else v for v in vals], 0.34,
                   color=col, label=lab, zorder=3)  # fmt: skip
            for xi, v in zip(x, vals, strict=True):
                if v is None and key == "ptrue_auroc":
                    ax.text(xi + 0.19, 0.505, "pending", rotation=90, fontsize=5.5, color=INK2,
                            ha="center", va="bottom")  # fmt: skip
        ax.axhline(0.5, color=INK, lw=0.8)
        ax.set_xticks(x, [TIERS[t].replace("DS ", "DS\n") for t in tiers], fontsize=6)
        ax.set_title(names[b], fontsize=8)
        ax.set_ylim(0.45, 0.92)
    axes[0, 0].set_ylabel("AUROC (visible-passing code)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, out, "ptrue")


def main() -> int:
    root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results"
    )
    out = root / "figures"
    figures = (validity, baselines, shift, repair, energy, stdin, conditional, router, economics,
               guaranteed, ci, oracle, ptrue)  # fmt: skip
    for f in figures:
        f(root, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
