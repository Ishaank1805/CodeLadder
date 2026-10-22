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


SCORES = ("agreement", "logprob", "selfconf", "agreement_selfconf", "visible")


def checks_ok(c: Candidate) -> bool:
    """Passes the visible tests and, where the task has model-written stress inputs, finishes
    on all of them within the time limit (a stress timeout counts as a failed check: 80 to 90%
    of such candidates are hidden-wrong, RESULTS.md 19 Oct)."""
    return c.passes_visible and '"timeout"' not in c.extra.get("stress_sig", "")


def agreement_key(c: Candidate) -> str:
    """Behaviour used for agreement clusters: outputs on the stress inputs for stdin tasks that
    have them, nothing for other stdin tasks (mutated stdin probes split correct programs), and
    outputs on the argument probes otherwise."""
    if "stress_sig" in c.extra:
        return c.extra["stress_sig"]
    if str(c.extra.get("probes", "")).startswith("stdin"):
        return ""
    return c.signature


def phi_scores(cands: list[Candidate], score: str = "agreement") -> list[float]:
    """Confidence for every candidate drawn so far at one tier. Candidates that fail a check
    (``checks_ok``) always score 0.

    - agreement: share of drawn candidates that pass the checks and behave identically
      (``agreement_key``; CodeT / MBR-exec style).
    - logprob: the model's own confidence, exp(mean token log-probability).
    - selfconf: the model's verbalised confidence (0 to 1) when asked about the candidate.
    - agreement_selfconf: product of the two.
    - visible: 1 for any candidate passing the visible tests (no further evidence).
    """
    if not cands:
        return []
    if score == "agreement":
        ok = [checks_ok(c) for c in cands]
        key = [agreement_key(c) for c in cands]
        clusters = Counter(k for k, o in zip(key, ok, strict=True) if o)
        n = len(cands)
        return [clusters[k] / n if o else 0.0 for k, o in zip(key, ok, strict=True)]
    if score == "logprob":
        return [
            math.exp(c.mean_logprob) if checks_ok(c) and c.mean_logprob is not None else 0.0
            for c in cands
        ]
    if score == "agreement_selfconf":
        # Behavioural agreement times the model's own rating: both must be high to accept.
        agree = phi_scores(cands, "agreement")
        conf = phi_scores(cands, "selfconf")
        return [a * c for a, c in zip(agree, conf, strict=True)]
    if score == "selfconf":
        return [float(c.extra.get("self_conf") or 0.0) if checks_ok(c) else 0.0 for c in cands]
    if score == "visible":
        return [1.0 if checks_ok(c) else 0.0 for c in cands]
    raise ValueError(f"unknown score {score!r}; choose from {SCORES}")
