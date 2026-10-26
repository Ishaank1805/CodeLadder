"""Cost versus solved share of the calibrated families across abstention prices.

    python scripts/figures/plot_frontier.py results humaneval+mbpp-t1-t2-t3-t4-resample-agreement_selfconf

Reads results/<prefix>-<family>-k<κ>/summary.json for the families singles, subsets and
two_level and every κ found, and writes fig_frontier-<prefix>.{pdf,png} into results/: one panel per
target α, mean cost per task (log scale) against the share of tasks solved, one line per
family through its κ points (higher κ: abstaining costs more, so more tasks are answered).
Every point is a certified operating point.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import AQUA, BLUE, INK2, ORANGE  # noqa: E402  (shared style and palette)

FAMILIES = [
    ("singles", "single model (calibrated selection)", BLUE),
    ("subsets", "sub-ladders", ORANGE),
    ("two_level", "full ladder (two-level)", AQUA),
]


def load(root: Path, prefix: str, family: str) -> list[tuple[float, dict]]:
    out = []
    for d in root.glob(f"{prefix}-{family}-k*"):
        m = re.fullmatch(rf"{re.escape(prefix)}-{family}-k([0-9.]+)", d.name)
        if m and (d / "summary.json").exists():
            out.append((float(m.group(1)), json.loads((d / "summary.json").read_text())))
    return sorted(out, key=lambda x: x[0])


def main() -> int:
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    runs = {f: load(root, prefix, f) for f, _, _ in FAMILIES}
    alphas = next([c["alpha"] for c in s["calibrated"]] for rs in runs.values() for _, s in rs[:1])
    fig, axes = plt.subplots(1, len(alphas), figsize=(7.2, 2.6), sharey=True)
    for ax, a in zip(axes, alphas, strict=True):
        for fam, label, color in FAMILIES:
            pts = []
            for k, s in runs[fam]:
                c = next(c for c in s["calibrated"] if abs(c["alpha"] - a) < 1e-9)
                if c["certified_splits"]:
                    pts.append((c["mean_test_cost"], c["mean_solved"], k))
            if not pts:
                continue
            pts.sort()
            xs, ys, _ = zip(*pts, strict=True)
            ax.plot(
                xs, ys, color=color, lw=2, marker="o", ms=5.5, mec="white", mew=1.2, label=label
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"α = {a:g}")
    axes[0].set_ylabel("share of tasks solved\n(accepted and correct)")
    axes[0].set_ylim(0, 1)
    fig.supxlabel(
        "mean cost per task (weighted tokens, log scale)", fontsize=8.5, color=INK2, y=0.02
    )
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(root / f"fig_frontier-{prefix}.{ext}")
    print(f"wrote {root}/fig_frontier-{prefix}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
