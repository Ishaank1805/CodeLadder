"""The experiment batches in scripts/experiments parse and list their commands (DRY=1)."""

import os
import subprocess
from pathlib import Path

import pytest

BATCHES = sorted(
    p
    for p in (Path(__file__).resolve().parent.parent / "scripts" / "experiments").glob("*.sh")
    if p.name not in ("common.sh", "all.sh")
)


def test_batches_exist():
    assert len(BATCHES) >= 10


@pytest.mark.parametrize("batch", BATCHES, ids=lambda p: p.stem)
def test_batch_dry_run_lists_commands(batch):
    out = subprocess.run(
        ["bash", str(batch)],
        env={**os.environ, "DRY": "1", "CL": "codeladder", "PY": "python3"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[-1] == "ALLDONE"
    cmds = [ln for ln in lines if ln.startswith("== ")]
    assert cmds and not any("FAILED" in ln for ln in lines)
