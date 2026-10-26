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
    if task.mode in ("function", "bcb"):
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
    # Clipped so a huge failing input cannot overflow the context window (8k tokens for the
    # local tiers); below the limit the text is unchanged.
    return _clip(_feedback_text(task, feedback), 8000)


def _feedback_text(task: Task, feedback: dict[str, Any] | None) -> str:
    if not feedback:
        return "The previous solution may be incorrect."
    kind = feedback.get("kind")
    if task.mode == "stdin":
        return _stdin_feedback(feedback)
    if task.mode == "bcb":
        if kind == "test_failure":
            return (
                f"It fails this test:\n```python\n{feedback.get('test', '')}\n```\n"
                f"{_clip(feedback.get('detail', ''))}"
            )
        return (
            f"The code failed before running any test ({kind}): {_clip(feedback.get('detail', ''))}"
        )
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
    if task.mode in ("function", "bcb") and f"def {task.entry_point}" not in code:
        # The model returned only a function body: graft it onto the prompt's signature.
        body = "\n".join(("    " + ln if ln.strip() else ln) for ln in code.splitlines())
        code = task.prompt.rstrip("\n") + "\n" + body
    elif task.mode == "bcb":
        # The prompt's imports belong to the task (BigCodeBench prepends them too).
        header = task.prompt.split(f"def {task.entry_point}")[0].strip("\n")
        missing = [ln for ln in header.splitlines() if ln.strip() and ln not in code]
        if missing:
            code = "\n".join(missing) + "\n" + code
    return code + "\n"


CONF_PREFILL = "Confidence (0-100): "


def confidence_messages(task: Task, code: str, prefill: bool = False) -> list[dict[str, str]]:
    """Ask the model how likely a candidate is to be fully correct (verbalised confidence).
    With ``prefill`` the answer is started for the model ("Confidence (0-100): "), for models
    that otherwise decline to give a number (DeepSeek-Coder: "as an AI, I cannot execute
    code"); the server must continue that assistant message (see ``continue_final``)."""
    msgs = initial_messages(task) + [
        {"role": "assistant", "content": f"```python\n{code.rstrip()}\n```"},
        {
            "role": "user",
            "content": (
                "How confident are you that this solution is correct on all valid inputs, "
                "including edge cases? Reply with a single integer from 0 to 100 and nothing else."
            ),
        },
    ]
    if prefill:
        msgs.append({"role": "assistant", "content": CONF_PREFILL})
    return msgs


_NUMBER = re.compile(r"\d{1,3}")


_MARKED = re.compile(r"(\d{1,3})\s*(?:%|/\s*100|percent)", re.IGNORECASE)
_NAMED = re.compile(r"confiden\w*\D{0,40}?(\d{1,3})", re.IGNORECASE)


_LEADING = re.compile(r"\s*(\d{1,3})(?!\d)")


def parse_confidence(text: str, prefilled: bool = False) -> float | None:
    """A 0 to 100 rating from the reply. A bare number (what most models answer) is read as
    is; in a sentence, a number marked as a percentage or named as the confidence wins over
    other numbers (e.g. "Python 3"); otherwise the last number from 0 to 100. A reply that
    continues the prefilled "Confidence (0-100): " must start with the number."""
    if prefilled:
        m = _LEADING.match(text or "")
        return min(int(m.group(1)), 100) / 100.0 if m else None
    text = (text or "").strip()
    if text.isdigit():
        return min(int(text), 100) / 100.0
    for pat in (_MARKED, _NAMED):
        m = pat.search(text)
        if m and int(m.group(1)) <= 100:
            return int(m.group(1)) / 100.0
    nums = [int(x) for x in _NUMBER.findall(text) if int(x) <= 100]
    return nums[-1] / 100.0 if nums else None


PTRUE_PREFILL = "Answer:"


def ptrue_messages(task: Task, code: str) -> list[dict[str, str]]:
    """P(True) self-evaluation (Kadavath et al., 2022): ask whether the candidate is correct
    and read the probability of "Yes" against "No" from the first answer token. The answer is
    started for the model ("Answer:"), so the next token is the verdict for every model."""
    return initial_messages(task) + [
        {"role": "assistant", "content": f"```python\n{code.rstrip()}\n```"},
        {
            "role": "user",
            "content": (
                "Is this solution correct on all valid inputs, including edge cases? "
                "Answer Yes or No."
            ),
        },
        {"role": "assistant", "content": PTRUE_PREFILL},
    ]


def p_true(first_top: dict[str, float] | None) -> float | None:
    """P(Yes) / (P(Yes) + P(No)) from the first token's top alternatives (any case or
    leading space); None when neither appears."""
    import math

    if not first_top:
        return None
    yes = sum(math.exp(lp) for t, lp in first_top.items() if t.strip().lower() == "yes")
    no = sum(math.exp(lp) for t, lp in first_top.items() if t.strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None
