"""Prompt templates and code extraction."""

from __future__ import annotations

import re
from typing import Any

from .types import Task

SYSTEM = (
    "You are an expert Python programmer. Reply with one complete, self-contained Python "
    "solution in a single ```python code block. Include any imports and helper functions. "
    "Do not include tests, examples or explanations."
)


def initial_messages(task: Task) -> list[dict[str, str]]:
    user = (
        f"Complete the following Python function `{task.entry_point}`.\n\n"
        f"```python\n{task.prompt.rstrip()}\n```"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _show(canon: Any) -> str:
    """Render a canonical output (see executor._canon) back into readable Python-ish text."""
    tag = canon[0]
    if tag == "n":
        return "None"
    if tag in ("b", "s"):
        return repr(canon[1])
    if tag == "i":
        return canon[1]
    if tag == "f":
        return str(canon[1])
    if tag == "l":
        return "[" + ", ".join(_show(x) for x in canon[1]) + "]"
    if tag == "t":
        inner = ", ".join(_show(x) for x in canon[1])
        return "(" + inner + ("," if len(canon[1]) == 1 else "") + ")"
    if tag == "S":
        return "{" + ", ".join(_show(x) for x in canon[1]) + "}" if canon[1] else "set()"
    if tag == "d":
        return "{" + ", ".join(f"{_show(k)}: {_show(v)}" for k, v in canon[1]) + "}"
    return str(canon[1])


def _args(args: list[str]) -> str:
    return ", ".join(args)  # each argument was stored as repr() by scoring.visible_report


def feedback_text(task: Task, feedback: dict[str, Any] | None) -> str:
    if not feedback:
        return "The previous solution may be incorrect."
    kind = feedback.get("kind")
    call = f"{task.entry_point}({_args(feedback['input'])})" if "input" in feedback else None
    if kind == "wrong_answer":
        return (
            f"For {call} the expected result is {_show(feedback['expected'])} "
            f"but your code returned {_show(feedback['got'])}."
        )
    if kind == "timeout":
        return f"{call} did not finish within the time limit."
    if kind == "exception":
        return f"{call} raised {feedback.get('error')}: {feedback.get('detail', '')}"
    return f"The code failed before running any test ({kind}): {feedback.get('detail', '')}"


def repair_messages(task: Task, code: str, feedback: dict[str, Any] | None) -> list[dict[str, str]]:
    msgs = initial_messages(task)
    msgs.append({"role": "assistant", "content": f"```python\n{code.rstrip()}\n```"})
    msgs.append(
        {
            "role": "user",
            "content": (
                "Your solution fails a visible test.\n"
                f"{feedback_text(task, feedback)}\n"
                "Fix the code and reply with the full corrected solution in one ```python block."
            ),
        }
    )
    return msgs


_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str, task: Task) -> str:
    """Pull the solution out of a model reply. Falls back to the raw text."""
    blocks = _FENCE.findall(text or "")
    code = max(blocks, key=len) if blocks else (text or "")
    code = code.strip("\n")
    if f"def {task.entry_point}" not in code:
        # The model returned only a function body: graft it onto the prompt's signature.
        body = "\n".join(("    " + ln if ln.strip() else ln) for ln in code.splitlines())
        code = task.prompt.rstrip("\n") + "\n" + body
    return code + "\n"
