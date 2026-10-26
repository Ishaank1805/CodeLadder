"""Move a flat results/ folder into the grouped layout (see src/codeladder/results.py).

    python scripts/organise_results.py [results] [--dry-run]

Run folders and result files are moved by name: evaluate runs to runs/<benchmarks>/, route-,
actions-, economics-, guaranteed- and oracle- runs to their own folders, baselines-*.json to
baselines/, other JSON files and preflight/ to analysis/, loose figures to figures/, and
stale_ptrue32/ into archive/. Safe to rerun; anything it does not recognise stays in place.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from codeladder.results import PREFIX_GROUPS, result_path  # noqa: E402

KEEP = {
    "runs",
    "analysis",
    "baselines",
    "ci",
    "policies",
    "figures",
    "tables",
    "archive",
    ".gitkeep",
}
KEEP |= {g for _, g in PREFIX_GROUPS}


def target(root: Path, p: Path) -> Path | None:
    name = p.name
    if name in KEEP:
        return None
    if name == "preflight":
        return root / "analysis" / "preflight"
    if name.startswith("stale_"):
        return root / "archive" / name
    if p.is_file() and name.startswith("fig_"):
        return root / "figures" / name
    if p.is_file() and name.endswith(".json"):
        return result_path(root, name)  # baselines/ or analysis/
    if p.is_dir() and any((p / f).exists() for f in ("summary.json", "shift.json")):
        return result_path(root, name)
    return None


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    root = Path(args[0]) if args else ROOT / "results"
    moved, left = 0, []
    for p in sorted(root.iterdir()):
        dst = target(root, p)
        if dst is None:
            if p.name not in KEEP:
                left.append(p.name)
            continue
        if dst.exists():
            left.append(f"{p.name} (target exists)")
            continue
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
        moved += 1
    print(f"{'would move' if dry else 'moved'} {moved} items in {root}")
    for name in left:
        print(f"left in place: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
