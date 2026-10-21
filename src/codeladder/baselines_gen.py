"""Extra generation passes needed by two published baselines.

Model Cascading for Code (arXiv 2405.15842): every model also writes k test cases of l assert
lines; solutions are ranked by how many generated test lines they pass and how many solutions
agree on them. ``make_selftests`` stores those tests per (task, tier).

Bayesian self-escalation (arXiv 2608.24087, real-model protocol): a greedy generation with
per-token log-probability, entropy and top-2 margin; a logistic-regression belief on their
running means decides, token by token, whether to abort and escalate. ``make_traces`` stores
the greedy program and its per-token statistics per (task, tier).

Both are generation only; execution and scoring happen offline (``baselines.py``).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .prompts import extract_code, initial_messages
from .types import Task

log = logging.getLogger(__name__)


def selftest_messages(task: Task, lines: int) -> list[dict[str, str]]:
    call = (
        f"`Solution().{task.entry_point}(...)`" if task.mode == "method" else f"`{task.entry_point}(...)`"
    )  # fmt: skip
    return [
        {"role": "system", "content": "You are an expert Python programmer and tester."},
        {
            "role": "user",
            "content": (
                f"{task.prompt.strip()}\n\nDo not implement the solution. Write {lines} test "
                f"lines, each a single Python assert statement that calls {call} on a new input "
                "and checks the exact expected result. Answer with the asserts only, one per "
                "line, in one ```python code block."
            ),
        },
    ]


def assert_lines(text: str, task: Task) -> list[str]:
    """Valid test lines: single-line asserts that parse and call the entry point."""
    code = extract_code(text, dataclasses.replace(task, mode="stdin"))  # no body grafting
    out = []
    for ln in code.splitlines():
        ln = ln.strip()
        if not ln.startswith("assert") or task.entry_point not in ln:
            continue
        try:
            ast.parse(ln)
        except SyntaxError:
            continue
        out.append(ln)
    return out


def _path(root: str | Path, kind: str, task: Task, tier: str) -> Path:
    return Path(root) / kind / task.benchmark / tier / f"{task.task_id.replace('/', '__')}.json"


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _run_all(todo: list[Task], work, workers: int) -> dict[str, int]:
    counts = {"ok": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, t): t for t in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
                counts["ok"] += 1
            except Exception:
                log.exception("failed on %s", futures[fut].task_id)
                counts["error"] += 1
    return counts


def make_selftests(
    tasks: list[Task],
    client,
    root: str | Path,
    tier: str,
    k: int = 8,
    lines: int = 4,
    max_tokens: int = 512,
    workers: int = 8,
) -> dict[str, int]:
    """k sampled test cases (temperature 0.8, as in the reference baseline) of up to ``lines`` asserts each."""
    todo = [
        t for t in tasks if t.mode != "stdin" and not _path(root, "selftests", t, tier).exists()
    ]

    def work(task: Task) -> None:
        tests = []
        for i in range(k):
            comp = client.complete(
                selftest_messages(task, lines), temperature=0.8, top_p=0.95, max_tokens=max_tokens, seed=i
            )  # fmt: skip
            tests.append(
                {
                    "lines": assert_lines(comp.text, task)[:lines],
                    "tokens_in": comp.tokens_in,
                    "tokens_out": comp.tokens_out,
                }
            )
        _write(_path(root, "selftests", task, tier), {"task_id": task.task_id, "tests": tests})

    return _run_all(todo, work, workers)


def make_traces(
    tasks: list[Task],
    client,
    root: str | Path,
    tier: str,
    max_tokens: int = 1024,
    top_logprobs: int = 5,
    workers: int = 8,
) -> dict[str, int]:
    """One greedy generation per task with per-token statistics."""
    todo = [t for t in tasks if not _path(root, "traces", t, tier).exists()]

    def work(task: Task) -> None:
        comp = client.complete(
            initial_messages(task), temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=0,
            top_logprobs=top_logprobs,
        )  # fmt: skip
        _write(
            _path(root, "traces", task, tier),
            {
                "task_id": task.task_id,
                "code": extract_code(comp.text, task),
                "tokens_in": comp.tokens_in,
                "tokens_out": comp.tokens_out,
                "token_stats": comp.token_stats or [],
            },
        )

    return _run_all(todo, work, workers)
