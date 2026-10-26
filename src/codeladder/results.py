"""Layout of the results folder: where each kind of result is written and how it is found.

    results/
      runs/<benchmarks>/<run>/   evaluate runs (summary.json, replay.npz), by benchmark set,
                                 e.g. runs/humaneval+mbpp/humaneval+mbpp-t1-t2-t3-t4-...
      router/ actions/ economics/ guaranteed/ oracle/   X9, X4, X13, X16, X17 runs
      baselines/                 replays of the published cascades (baselines-*.json)
      analysis/                  analysis outputs (signals, repair, silent failures, preflight)
      ci/ policies/ figures/ tables/
      archive/                   results no current batch produces (never read by the tools)

A result is named as before (the name encodes benchmarks, tiers, score, family and tag);
``result_path`` turns a name into its place, and ``find_result`` accepts a name, a
``results/<name>`` path written before the layout existed, or any existing path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

PREFIX_GROUPS = (
    ("route-", "router"),
    ("actions-", "actions"),
    ("economics-", "economics"),
    ("guaranteed-", "guaranteed"),
    ("oracle-", "oracle"),
    ("baselines-", "baselines"),
)
SKIP = ("archive",)  # folders the tools never read


def group_of(name: str) -> Path:
    """The folder (relative to results/) a result of this name belongs in."""
    for prefix, group in PREFIX_GROUPS:
        if name.startswith(prefix):
            return Path(group)
    if name.endswith(".json"):  # analysis outputs (x2_stdin_signals.json, e4_repair_*.json, ...)
        return Path("analysis")
    return Path("runs") / name.split("-", 1)[0]  # evaluate runs, by benchmark set


def result_path(root: str | Path, name: str) -> Path:
    return Path(root) / group_of(name) / name


def find_result(root: str | Path, ref: str | Path) -> Path:
    """An existing result for a name or path; the place it would have if it does not exist."""
    p = Path(ref)
    if p.exists():
        return p
    for cand in (result_path(root, p.name), Path(root) / p.name):
        if cand.exists():
            return cand
    return result_path(root, p.name)


def iter_runs(root: str | Path, pattern: str = "*") -> Iterator[Path]:
    """Every result folder with a summary.json whose name matches ``pattern``, outside archive."""
    root = Path(root)
    for f in sorted(root.rglob("summary.json")):
        d = f.parent
        if d.match(pattern) and not any(part in SKIP for part in d.relative_to(root).parts):
            yield d
