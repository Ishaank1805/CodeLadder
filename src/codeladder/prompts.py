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
    if task.mode == "function":
        user = (
            f"Complete the following Python function `{task.entry_point}`.\n\n"
            f"```python\n{task.prompt.rstrip()}\n```"
        )
    else:  # LiveCodeBench: the prompt is the full problem statement and instructions
        user = task.prompt
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
    if task.mode == "stdin":
        return _stdin_feedback(feedback)
    fn = f"Solution().{task.entry_point}" if task.mode == "method" else task.entry_point
    call = f"{fn}({_args(feedback['input'])})" if "input" in feedback else None
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


def _clip(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else text[:n] + f"... [{len(text) - n} more characters]"


def _stdin_feedback(feedback: dict[str, Any]) -> str:
    kind = feedback.get("kind")
    shown = f"Input:\n{_clip(feedback['input'][0])}\n" if feedback.get("input") else ""
    if kind == "wrong_answer":
        exp, got = feedback["expected"], feedback["got"]
        exp_s = _clip(exp[1]) if exp[0] == "s" else "(very long output)"
        got_s = _clip(got[1]) if got[0] == "s" else "(very long output)"
        return f"{shown}Expected output:\n{exp_s}\nYour program printed:\n{got_s}"
    if kind == "timeout":
        return f"{shown}Your program did not finish within the time limit."
    if kind == "exception":
        return f"{shown}Your program raised {feedback.get('error')}: {feedback.get('detail', '')}"
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
    if task.mode == "function" and f"def {task.entry_point}" not in code:
        # The model returned only a function body: graft it onto the prompt's signature.
        body = "\n".join(("    " + ln if ln.strip() else ln) for ln in code.splitlines())
        code = task.prompt.rstrip("\n") + "\n" + body
    return code + "\n"


def confidence_messages(task: Task, code: str) -> list[dict[str, str]]:
    """Ask the model how likely a candidate is to be fully correct (verbalised confidence)."""
    return initial_messages(task) + [
        {"role": "assistant", "content": f"```python\n{code.rstrip()}\n```"},
        {
            "role": "user",
            "content": (
                "How confident are you that this solution is correct on all valid inputs, "
                "including edge cases? Reply with a single integer from 0 to 100 and nothing else."
            ),
        },
    ]


_NUMBER = re.compile(r"\d{1,3}")


def parse_confidence(text: str) -> float | None:
    m = _NUMBER.search(text or "")
    if not m:
        return None
    return min(int(m.group()), 100) / 100.0
