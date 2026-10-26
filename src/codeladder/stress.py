"""Stress inputs for whole-program (stdin) tasks.

Most stdin silent failures are correct-but-too-slow programs (RESULTS.md, 19 Oct): probes
mutated from the small public inputs cannot expose them. Here a fixed model writes, once per
problem, a generator program that prints one large valid input from the statement and its
constraints. Candidates are then also run on these inputs with the problem's own time limit,
and the outcome (output or timeout) joins their behaviour signature for agreement.

Label-free: generators see only the statement and the public examples, never hidden tests.
Generated inputs are kept only if their first line has the same shape (token count and kinds)
as the public inputs' first line, a cheap check that the format was followed.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .executor import _limits, run_isolated
from .llm import ChatClient
from .prompts import extract_code
from .types import Task

log = logging.getLogger(__name__)

MAX_CHARS = 200_000  # asked for: big enough for quadratic programs to time out
ACCEPT_CHARS = 1_000_000  # accepted (stored gzipped); larger outputs get one request to shrink


def stress_messages(task: Task) -> list[dict[str, str]]:
    statement = task.prompt.split("\n\nWrite a complete Python program")[0]
    return [
        {"role": "system", "content": "You are an expert competitive programmer and tester."},
        {
            "role": "user",
            "content": (
                f"{statement}\n\n"
                "Write a Python 3 program that prints ONE valid test input for this problem to "
                "standard output, in exactly the input format above. Make it a stress test: as "
                f"large as the constraints allow, but at most {MAX_CHARS:,} characters in total "
                "(scale the sizes down to fit), with random values that satisfy every "
                "constraint. Use random.seed(1). Print nothing else. Answer with the program "
                "only, in one ```python code block."
            ),
        },
    ]


def run_generator(code: str, timeout_s: float = 10.0, memory_mb: int = 2048) -> str | None:
    """Run a generator program in the sandbox limits and return its stdout, or None."""
    out, _ = run_generator_why(code, timeout_s, memory_mb)
    return out


def run_generator_why(
    code: str, timeout_s: float = 10.0, memory_mb: int = 2048
) -> tuple[str | None, str]:
    """Like ``run_generator``, with the reason when there is no usable output."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="codeladder-gen-") as tmp:
        try:
            proc = run_isolated(
                [sys.executable, "-s", "-B", "-c", code],
                stdin=subprocess.DEVNULL,
                timeout=timeout_s,
                cwd=tmp,
                env=env,
                preexec_fn=_limits(memory_mb, int(timeout_s) + 1),
            )
        except subprocess.TimeoutExpired:
            return None, f"it did not finish within {timeout_s:g} seconds"
    out = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        if proc.returncode < 0 or len(proc.stdout) >= 10 * 1024 * 1024:
            return None, "it was killed (too much output or memory)"
        err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return None, f"it crashed: {err[-1] if err else 'no message'}"
    if not out.strip():
        return None, "it printed nothing"
    if len(out) > ACCEPT_CHARS:
        return None, f"it printed {len(out):,} characters"
    return out, ""


def _head_shape(text: str) -> tuple[str, ...]:
    first = text.strip().split("\n", 1)[0].split()
    return tuple("i" if t.lstrip("-").isdigit() else "s" for t in first)


def plausible(text: str, visible_inputs: list[str]) -> bool:
    """First line has the public inputs' token count, with a number wherever every public
    input has one (a digit-only string such as "101" may stand for a string token)."""
    shapes = [_head_shape(x) for x in visible_inputs if isinstance(x, str)]
    if not shapes or len({len(sh) for sh in shapes}) != 1:
        return False
    head = _head_shape(text)
    if len(head) != len(shapes[0]):
        return False
    return all(h == "i" for k, h in enumerate(head) if all(sh[k] == "i" for sh in shapes))


def stress_path(root: str | Path, task: Task) -> Path:
    return Path(root) / f"{task.task_id.replace('/', '__')}.json.gz"


def make_stress_inputs(
    tasks: list[Task],
    client: ChatClient,
    root: str | Path,
    model: str,
    n_generators: int = 3,
    keep: int = 2,
    max_tokens: int = 1024,
    workers: int = 8,
) -> dict[str, int]:
    """Write up to ``keep`` stress inputs per stdin task (resumable: existing files are kept).
    Returns counts of tasks by outcome."""
    from .tasks import prepare_task, release_tests

    Path(root).mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if t.mode == "stdin" and not stress_path(root, t).exists()]

    def work(task: Task) -> str:
        prepare_task(task)
        visible = list(task.visible_inputs)
        release_tests(task)  # only the statement is needed
        inputs, tokens = [], 0
        for k in range(n_generators):
            msgs = stress_messages(task)
            comp = client.complete(msgs, temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=k)
            tokens += comp.tokens_in + comp.tokens_out
            code = extract_code(comp.text, task)
            out, why = run_generator_why(code)
            if out is None and why:
                # One request to fix it, with the observed problem (usually: far too large).
                fix = msgs + [
                    {"role": "assistant", "content": comp.text},
                    {
                        "role": "user",
                        "content": f"The program is not usable: {why}. Fix it so it prints one "
                        f"valid input of at most {MAX_CHARS:,} characters in total. Answer with "
                        "the program only, in one ```python code block.",
                    },
                ]
                comp = client.complete(
                    fix, temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=100 + k
                )
                tokens += comp.tokens_in + comp.tokens_out
                out, why = run_generator_why(extract_code(comp.text, task))
            if out is not None and plausible(out, visible) and out not in inputs:
                inputs.append(out)
            if len(inputs) >= keep:
                break
        record = {"task_id": task.task_id, "model": model, "inputs": inputs, "tokens": tokens}
        tmp = stress_path(root, task).with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(record).encode()))
        os.replace(tmp, stress_path(root, task))
        return "ok" if inputs else "none"

    counts: dict[str, int] = {"ok": 0, "none": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, t): t for t in todo}
        for fut in as_completed(futures):
            try:
                counts[fut.result()] += 1
            except Exception:
                log.exception("stress generation failed for %s", futures[fut].task_id)
                counts["error"] += 1
    return counts


def load_stress_inputs(root: str | Path, task: Task) -> list[str]:
    p = stress_path(root, task)
    if not p.exists():
        return []
    rec: dict[str, Any] = json.loads(gzip.decompress(p.read_bytes()))
    return rec.get("inputs", [])
