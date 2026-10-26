"""System diagram of CodeLadder for the reports and the demo video.

    python scripts/figures/diagram.py [results]

Writes results/figures/fig_system.{pdf,png}: top, how a certified ladder answers one task
(rungs from the cheapest model up; at each rung draw candidates, run the visible tests and
probes, score, then accept, resample / repair, escalate or abstain); bottom, how the thresholds
are certified offline (replay every setting on labelled calibration tasks, Learn-then-Test,
choose the cheapest certified setting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import AQUA, BLUE, INK, INK2, ORANGE  # noqa: E402

YELLOW = "#eda100"
PALE = {BLUE: "#e7f0fb", AQUA: "#e3f6ef", ORANGE: "#fdece4", YELLOW: "#fdf3dc", INK2: "#f1f0ee"}


def box(ax, x, y, w, h, text, color=INK2, bold=False, size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=1.1, edgecolor=color, facecolor=PALE.get(color, "white")))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)  # fmt: skip


def arrow(ax, a, b, color=INK2, text=None, rad=0.0, tx=0.0, ty=0.012, ha="center", size=6.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))  # fmt: skip
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, ha=ha, va="bottom",
                fontsize=size, color=color)  # fmt: skip


def polyline_arrow(ax, pts, color=INK2):
    """Orthogonal connector: straight segments, arrowhead on the last one."""
    xs, ys = zip(*pts, strict=True)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=1.0, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color)


def main() -> int:
    root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results"
    )
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- online: answering one task
    ax.text(
        0.02,
        0.965,
        "Answering a task with a certified ladder",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    h = 0.13
    rows = [(0.80, "rung 1: cheapest model", "λ₁"), (0.50, "rung k: next model up", "λₖ")]
    box(ax, 0.02, 0.80 - h / 2, 0.13, h, "Task:\nprompt and\nvisible tests", INK2)
    arrow(ax, (0.15, 0.80), (0.19, 0.80))
    for y, model, lam in rows:
        box(ax, 0.19, y - h / 2, 0.17, h, f"{model}\ndraws a candidate\n(sample or repair)", BLUE)
        box(
            ax,
            0.40,
            y - h / 2,
            0.18,
            h,
            "checks: visible tests,\nprobe and stress inputs\n(sandboxed)",
            AQUA,
        )
        box(
            ax,
            0.62,
            y - h / 2,
            0.15,
            h,
            f"score = agreement\nx confidence,\ncompared with {lam}",
            YELLOW,
        )
        arrow(ax, (0.36, y), (0.40, y))
        arrow(ax, (0.58, y), (0.62, y))
        box(
            ax,
            0.82,
            y - 0.045,
            0.16,
            0.09,
            "score ≥ " + lam + ":\naccept the code",
            AQUA,
            bold=True,
            size=6.8,
        )
        arrow(ax, (0.77, y), (0.82, y), AQUA)
        # below the threshold with draws left: back to the same rung
        arrow(ax, (0.695, y + h / 2), (0.275, y + h / 2), ORANGE,
              "below " + lam + ", draws left: resample or repair", rad=0.3, ty=0.022, size=6.4)  # fmt: skip
    # budget spent at this rung: escalate to the next model
    ym = 0.70
    polyline_arrow(
        ax, [(0.695, 0.80 - h / 2), (0.695, ym), (0.275, ym), (0.275, 0.50 + h / 2 + 0.002)]
    )
    ax.text(0.705, ym - 0.002, "draws spent:\nescalate", fontsize=6.4, color=INK2, va="center")
    # last rung spent: abstain
    box(ax, 0.82, 0.325, 0.16, 0.08, "last rung spent:\nabstain", ORANGE, bold=True, size=6.8)
    polyline_arrow(ax, [(0.695, 0.50 - h / 2), (0.695, 0.365), (0.82, 0.365)], ORANGE)
    ax.text(0.02, 0.385, "Guarantee: with probability at least 1 − δ,\nat most α of tasks end with\naccepted but wrong code.", fontsize=7.0, color=BLUE, va="center",
            fontstyle="italic")  # fmt: skip

    # ---- offline: certifying the thresholds
    ax.plot([0.02, 0.98], [0.305, 0.305], color="#d9d7d2", lw=0.8, ls="--")
    ax.text(
        0.02,
        0.265,
        "Certifying the thresholds (offline, once per ladder and α)",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    steps = [
        ("Candidate pools\nfor calibration tasks,\nlabelled by hidden tests", INK2),
        ("Replay the ladder\nfor every setting λ:\nrisk and cost per task", BLUE),
        ("Learn-then-Test: keep\nsettings with risk ≤ α\nwith prob. ≥ 1 − δ", YELLOW),
        ("Choose the cheapest\ncertified setting\n(cost + κ x abstention)", AQUA),
    ]
    w, gap, x0 = 0.205, 0.035, 0.02
    for i, (text, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, 0.03, w, 0.17, text, color)
        if i:
            arrow(ax, (x - gap, 0.115), (x, 0.115))
    xl = x0 + 3 * (w + gap) + w / 2
    polyline_arrow(
        ax, [(xl, 0.20), (xl, 0.285), (0.645, 0.285), (0.645, 0.50 - h / 2 - 0.002)], AQUA
    )
    ax.text(
        0.632,
        0.385,
        "certified\nthresholds\nλ₁ ... λₖ",
        fontsize=6.6,
        color=AQUA,
        va="center",
        ha="right",
    )
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_system.{ext}", dpi=200)
    print(f"wrote {out}/fig_system.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
