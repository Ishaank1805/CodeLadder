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


@pytest.mark.parametrize("batch", BATCHES, ids=lambda p: p.stem)
def test_batch_commands_parse(batch):
    """Every codeladder command a batch would run is accepted by the real argument parser."""
    import shlex

    from codeladder.cli import build_parser

    out = subprocess.run(
        ["bash", str(batch)],
        env={**os.environ, "DRY": "1", "CL": "codeladder", "PY": "python3"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    parser = build_parser()
    for ln in out.stdout.splitlines():
        if not ln.startswith("== "):
            continue
        argv = shlex.split(ln[3:])
        if (
            argv
            and argv[0] in ("scripts/show_results.py", "scripts/count_validity.py")
            or (argv and argv[0].endswith(".py"))
        ):
            continue  # helper scripts, not the codeladder command
        try:
            parser.parse_args(argv)
        except SystemExit as e:
            raise AssertionError(f"{batch.name}: rejected by the parser: {ln}") from e


def test_every_command_has_help():
    from codeladder.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "cmd")
    missing = [a.dest for a in sub._choices_actions if not a.help]
    assert not missing and set(sub.choices) == {a.dest for a in sub._choices_actions}, missing
