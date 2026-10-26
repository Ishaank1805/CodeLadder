"""Report figures from an evaluate run (results/<run>/summary.json + replay.npz).

    python scripts/figures/plot_results.py results/humaneval+mbpp-t1-t2-resample-agreement

Writes into the run folder:
  fig_guarantee.{pdf,png}   realised test risk over calibration/test splits per target alpha,
                            with the y = alpha line, plus coverage per alpha
  fig_cost_risk.{pdf,png}   risk versus cost of every (uncalibrated) threshold, with the
                            calibrated operating points and baselines marked
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (light mode): fixed slot order, recessive ink for text and grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def guarantee_figure(summary: dict, out: Path) -> None:
    cal = summary["calibrated"]
    alphas = [c["alpha"] for c in cal]
    risks = [[s["test_risk"] for s in c["splits"] if s.get("chosen") is not None] for c in cal]
    cover = [c["mean_coverage"] for c in cal]
    x = np.arange(len(alphas))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    rng = np.random.default_rng(0)
    for i, r in enumerate(risks):
        jitter = rng.uniform(-0.16, 0.16, size=len(r))
        ax1.scatter(x[i] + jitter, r, s=9, color=BLUE, alpha=0.45, linewidths=0, zorder=3)
        ax1.plot([x[i] - 0.3, x[i] + 0.3], [alphas[i]] * 2, color=INK, lw=1.4, ls="--", zorder=4)
        ax1.text(x[i], 0.97, f"true risk > α\nin {cal[i]['violation_rate']:.0%} of splits",
                 transform=ax1.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=INK2)  # fmt: skip
    ax1.set_xticks(x, [f"α = {a:g}" for a in alphas])
    ax1.set_xlim(-0.5, len(alphas) - 0.5)
    ax1.set_ylabel("accepted-but-wrong rate\n(each dot: one test split)")
    ax1.set_title("Realised risk stays below the target")
    ax1.set_ylim(0, 1.45 * max(alphas))

    ax2.bar(x, cover, width=0.5, color=BLUE, zorder=3)
    for xi, c in zip(x, cover, strict=True):
        ax2.text(xi, c + 0.02, f"{c:.0%}", ha="center", fontsize=8, color=INK)
    ax2.set_xticks(x, [f"{a:g}" for a in alphas])
    ax2.set_xlabel("target α")
    ax2.set_ylabel("share of tasks answered")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Coverage")
    fig.suptitle(
        f"{summary['n_tasks']} tasks, tiers {'+'.join(summary['tiers'])}, "
        f"{len(risks[0]) if risks else 0} calibration/test splits, δ = {cal[0]['delta']:g}",
        fontsize=8,
        color=INK2,
        y=1.02,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_guarantee.{ext}")
    plt.close(fig)


def cost_risk_figure(summary: dict, rep: dict, out: Path) -> None:
    loss, cost = rep["loss"], rep["cost"]
    risk_curve = loss.mean(axis=(1, 2))
    cost_curve = cost.mean(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    order = np.argsort(cost_curve)
    ax.plot(
        cost_curve[order],
        risk_curve[order],
        color=ORANGE,
        lw=2,
        zorder=3,
        label="fixed thresholds, uncalibrated",
    )
    for c in summary["calibrated"]:
        ax.scatter(c["mean_test_cost"], c["mean_test_risk"], s=60, color=BLUE, zorder=5,
                   edgecolors="white", linewidths=1.5)  # fmt: skip
        ax.annotate(f"calibrated, α = {c['alpha']:g}", (c["mean_test_cost"], c["mean_test_risk"]),
                    xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=7.5, color=INK)  # fmt: skip
        ax.axhline(c["alpha"], color=MUTED, lw=0.8, ls=":", zorder=1)
    for name, b in summary["baselines"].items():
        if name.startswith("only-"):
            ax.scatter(b["cost"], b["risk"], s=40, marker="s", color=AQUA, zorder=4,
                       edgecolors="white", linewidths=1.2)  # fmt: skip
            ax.annotate(name.replace("only-", "") + " alone\n(uncalibrated)", (b["cost"], b["risk"]),
                        xytext=(0, -24), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)  # fmt: skip
    ax.set_xlabel("mean cost per task (weighted output tokens)")
    ax.set_ylabel("accepted-but-wrong rate")
    ax.set_title("Cost against risk")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_cost_risk.{ext}")
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    run = Path(sys.argv[1])
    summary = json.loads((run / "summary.json").read_text())
    rep = dict(np.load(run / "replay.npz"))
    guarantee_figure(summary, run)
    cost_risk_figure(summary, rep, run)
    print(f"wrote figures to {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
