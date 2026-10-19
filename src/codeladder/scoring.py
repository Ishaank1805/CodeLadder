"""Turning execution results into confidence signals.

phi(candidate) is 0 if the candidate fails any visible test. Otherwise it is the
share of candidates drawn so far at this tier that pass the visible tests AND behave
identically on the probe inputs (CodeT / MBR-exec style agreement).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from .executor import RunResult
from .oracles import check
from .types import Candidate


def _input_repr(args: Any) -> list[str]:
    """Failing input for repair prompts: argument reprs, or the raw stdin text."""
    if isinstance(args, str):
        return [args[:4000]]
    return [repr(a) for a in args]


def visible_report(
    run: RunResult,
    inputs: list[list[Any]],
    expected: list[Any],
    atol: float,
    oracle: str | None = None,
) -> tuple[float, str, dict[str, Any] | None]:
    """Return (pass rate, error type, first failure) for a run on the visible tests."""
    if run.fatal:
        return 0.0, run.fatal, {"kind": run.fatal, "detail": run.detail}
    if not expected:
        return 1.0, "pass", None
    passed = 0
    first_failure: dict[str, Any] | None = None
    error_type = "pass"
    for args, res, exp in zip(inputs, run.results, expected, strict=True):
        if res["status"] == "ok" and check(oracle, args, res["out"], exp, atol):
            passed += 1
            continue
        if first_failure is None:
            if res["status"] == "ok":
                error_type = "wrong_answer"
                first_failure = {
                    "kind": "wrong_answer",
                    "input": _input_repr(args),
                    "expected": exp,
                    "got": res["out"],
                }
            elif res["status"] == "timeout":
                error_type = "timeout"
                first_failure = {"kind": "timeout", "input": _input_repr(args)}
            else:
                error_type = f"exception:{res.get('error', 'Exception')}"
                first_failure = {
                    "kind": "exception",
                    "input": _input_repr(args),
                    "error": res.get("error"),
                    "detail": res.get("detail", ""),
                }
    return passed / len(expected), error_type, first_failure


def behaviour_signature(run: RunResult) -> str:
    """A string that is equal for two programs iff they behave the same on the probes."""
    if run.fatal:
        return f"fatal:{run.fatal}"
    parts = []
    for r in run.results:
        if r["status"] == "ok":
            parts.append(r["out"])
        elif r["status"] == "timeout":
            parts.append(["timeout"])
        else:
            parts.append(["exc", r.get("error")])
    return json.dumps(parts, sort_keys=True)


SCORES = ("agreement", "logprob", "selfconf", "visible")


def phi_scores(cands: list[Candidate], score: str = "agreement") -> list[float]:
    """Confidence for every candidate drawn so far at one tier. Candidates that fail a visible
    test always score 0.

    - agreement: share of drawn candidates that pass the visible tests and behave identically
      on the probe inputs (CodeT / MBR-exec style).
    - logprob: the model's own confidence, exp(mean token log-probability).
    - selfconf: the model's verbalised confidence (0 to 1) when asked about the candidate.
    - visible: 1 for any candidate passing the visible tests (no further evidence).
    """
    if not cands:
        return []
    if score == "agreement":
        clusters = Counter(c.signature for c in cands if c.passes_visible)
        n = len(cands)
        return [clusters[c.signature] / n if c.passes_visible else 0.0 for c in cands]
    if score == "logprob":
        return [
            math.exp(c.mean_logprob) if c.passes_visible and c.mean_logprob is not None else 0.0
            for c in cands
        ]
    if score == "selfconf":
        return [float(c.extra.get("self_conf") or 0.0) if c.passes_visible else 0.0 for c in cands]
    if score == "visible":
        return [1.0 if c.passes_visible else 0.0 for c in cands]
    raise ValueError(f"unknown score {score!r}; choose from {SCORES}")
