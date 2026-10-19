"""Turning execution results into confidence signals.

phi(candidate) is 0 if the candidate fails any visible test. Otherwise it is the
share of candidates drawn so far at this tier that pass the visible tests AND behave
identically on the probe inputs (CodeT / MBR-exec style agreement).
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .executor import RunResult
from .oracles import check
from .types import Candidate


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
                    "input": [repr(a) for a in args],
                    "expected": exp,
                    "got": res["out"],
                }
            elif res["status"] == "timeout":
                error_type = "timeout"
                first_failure = {"kind": "timeout", "input": [repr(a) for a in args]}
            else:
                error_type = f"exception:{res.get('error', 'Exception')}"
                first_failure = {
                    "kind": "exception",
                    "input": [repr(a) for a in args],
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


def phi_scores(cands: list[Candidate]) -> list[float]:
    """Agreement-based confidence for every candidate drawn so far at one tier."""
    if not cands:
        return []
    clusters = Counter(c.signature for c in cands if c.passes_visible)
    n = len(cands)
    return [clusters[c.signature] / n if c.passes_visible else 0.0 for c in cands]
