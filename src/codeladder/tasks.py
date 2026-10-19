"""Benchmark loaders.

Every loader returns a list of :class:`Task`. Visible tests are what the harness
may use; hidden tests define correctness and are only used for labels/evaluation.

- ``toy``: three tiny built-in tasks for smoke tests (no downloads).
- ``humaneval`` / ``mbpp``: HumanEval+ and MBPP+ from EvalPlus. Base inputs are
  visible, plus inputs are hidden. Requires ``pip install evalplus``.
- ``livecodebench``: TODO (see docs/PLAN.md). Needs a stdin-style runner for
  AtCoder/Codeforces problems in addition to the function-call runner.

Note: a few EvalPlus tasks use custom output checkers (for example HumanEval/32);
our generic comparison may disagree with the official evaluator on those. Final
numbers should be cross-checked with ``evalplus.evaluate`` on accepted outputs.
"""

from __future__ import annotations

import time
from typing import Any

from .executor import run_program
from .oracles import oracle_for
from .types import Task

_TOY = [
    {
        "task_id": "toy/add",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        "entry_point": "add",
        "reference": "def add(a, b):\n    return a + b\n",
        "visible": [[1, 2], [0, 0]],
        "hidden": [[-5, 5], [10**6, 1], [7, -9]],
    },
    {
        "task_id": "toy/remove_duplicates",
        "prompt": (
            "from typing import List\n\n"
            "def remove_duplicates(numbers: List[int]) -> List[int]:\n"
            '    """From a list of integers, remove all elements that occur more than once.\n'
            "    Keep order of elements left the same as in the input.\n"
            "    >>> remove_duplicates([1, 2, 3, 2, 4])\n"
            '    [1, 3, 4]\n    """\n'
        ),
        "entry_point": "remove_duplicates",
        "reference": (
            "def remove_duplicates(numbers):\n"
            "    from collections import Counter\n"
            "    c = Counter(numbers)\n"
            "    return [n for n in numbers if c[n] == 1]\n"
        ),
        "visible": [[[1, 2, 3, 2, 4]]],
        "hidden": [[[]], [[1, 1, 1]], [[5, 6, 5, 7, 8, 8]]],
    },
    {
        "task_id": "toy/is_prime",
        "prompt": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
        "entry_point": "is_prime",
        "reference": (
            "def is_prime(n):\n"
            "    if n < 2:\n        return False\n"
            "    i = 2\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n            return False\n"
            "        i += 1\n"
            "    return True\n"
        ),
        "visible": [[2], [4], [17]],
        "hidden": [[1], [0], [97], [100], [7919]],
    },
]


def load_toy() -> list[Task]:
    return [
        Task(
            task_id=t["task_id"],
            benchmark="toy",
            prompt=t["prompt"],
            entry_point=t["entry_point"],
            reference=t["reference"],
            visible_inputs=t["visible"],
            hidden_inputs=t["hidden"],
        )
        for t in _TOY
    ]


def _from_evalplus(problems: dict[str, dict[str, Any]], benchmark: str) -> list[Task]:
    tasks = []
    for p in problems.values():
        canonical = p["canonical_solution"]
        entry = p["entry_point"]
        reference = canonical if f"def {entry}" in canonical else p["prompt"] + canonical
        tasks.append(
            Task(
                task_id=p["task_id"],
                benchmark=benchmark,
                prompt=p["prompt"],
                entry_point=entry,
                reference=reference,
                visible_inputs=[list(x) for x in p["base_input"]],
                hidden_inputs=[list(x) for x in p["plus_input"]],
                atol=float(p.get("atol") or 0.0),
                oracle=oracle_for(benchmark, entry),
            )
        )
    return tasks


def load_humaneval_plus() -> list[Task]:
    from evalplus.data import get_human_eval_plus

    return _from_evalplus(get_human_eval_plus(), "humaneval")


def load_mbpp_plus() -> list[Task]:
    from evalplus.data import get_mbpp_plus

    return _from_evalplus(get_mbpp_plus(), "mbpp")


def load_livecodebench() -> list[Task]:
    raise NotImplementedError(
        "LiveCodeBench adapter not written yet: see docs/PLAN.md (week 3, stdin runner)."
    )


LOADERS = {
    "toy": load_toy,
    "humaneval": load_humaneval_plus,
    "mbpp": load_mbpp_plus,
    "livecodebench": load_livecodebench,
}


def load_tasks(benchmark: str, limit: int | None = None) -> list[Task]:
    if benchmark not in LOADERS:
        raise ValueError(f"unknown benchmark {benchmark!r}; choose from {sorted(LOADERS)}")
    tasks = LOADERS[benchmark]()
    return tasks[:limit] if limit else tasks


def attach_expected_outputs(
    task: Task, per_call_timeout: float = 5.0, memory_mb: int = 4096
) -> Task:
    """Run the reference solution to get expected outputs.

    Inputs on which the reference itself fails are dropped, so visible/hidden
    tests stay aligned with their expected outputs.
    """
    for split in ("visible", "hidden"):
        inputs = getattr(task, f"{split}_inputs")
        t0 = time.time()
        run = run_program(
            task.reference, task.entry_point, inputs, per_call_timeout, memory_mb, total_timeout=900
        )
        if split == "hidden":
            task.ref_seconds = time.time() - t0
        if run.fatal:
            raise RuntimeError(
                f"reference failed to load for {task.task_id}: {run.fatal} {run.detail}"
            )
        kept_inputs, expected, times = [], [], []
        for args, res in zip(inputs, run.results, strict=True):
            if res["status"] == "ok":
                kept_inputs.append(args)
                expected.append(res["out"])
                times.append(float(res.get("dt", 0.0)))
        setattr(task, f"{split}_inputs", kept_inputs)
        setattr(task, f"{split}_expected", expected)
        setattr(task, f"{split}_ref_times", times)
    return task
