"""Sandboxed execution of candidate programs.

Each call runs the candidate in a fresh Python subprocess with CPU, memory and
file-size limits and a per-call timeout. Outputs are converted to a canonical
JSON form inside the child, so the parent never unpickles anything produced by
generated code.

This is defence in depth, not a security boundary: on Ada, run generation inside
a Singularity container and never as a privileged user.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

_MARKER = "@@CODELADDER_RESULT@@"

# Runs inside the child process. Kept as a string so the child imports nothing from us.
_RUNNER = r"""
import copy, hashlib, io, json, math, pickle, signal, sys, time

_MARKER = "@@CODELADDER_RESULT@@"
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)  # allow huge ints (HumanEval/83, /139)
_real_stdout = sys.stdout
sys.stdout = io.StringIO()  # swallow candidate prints

def _canon(x, depth=0):
    if depth > 40:
        return ["r", "<too-deep>"]
    if x is None:
        return ["n"]
    if isinstance(x, bool):
        return ["b", x]
    if isinstance(x, int):
        return ["i", str(x)]
    if isinstance(x, float):
        if math.isnan(x):
            return ["f", "nan"]
        if math.isinf(x):
            return ["f", "inf" if x > 0 else "-inf"]
        return ["f", x]
    if isinstance(x, complex):
        return ["c", [x.real, x.imag]]
    if isinstance(x, str):
        return ["s", x]
    if isinstance(x, (bytes, bytearray)):
        return ["y", list(x)]
    if isinstance(x, tuple):
        return ["t", [_canon(v, depth + 1) for v in x]]
    if isinstance(x, list):
        return ["l", [_canon(v, depth + 1) for v in x]]
    if isinstance(x, (set, frozenset)):
        items = [_canon(v, depth + 1) for v in x]
        return ["S", sorted(items, key=lambda c: json.dumps(c, sort_keys=True))]
    if isinstance(x, dict):
        items = [[_canon(k, depth + 1), _canon(v, depth + 1)] for k, v in x.items()]
        return ["d", sorted(items, key=lambda kv: json.dumps(kv[0], sort_keys=True))]
    try:
        return ["r", repr(x)[:1000]]
    except Exception:
        return ["r", "<unrepr>"]

def _numkey(c):
    # Python treats 3 == 3.0 == True-ish ints; normalise numbers before hashing.
    tag, val = c[0], c[1]
    if tag == "b":
        return format(float(val), ".10g")
    if tag == "i":
        return format(float(val), ".10g") if len(val) < 16 else val
    if isinstance(val, str):
        return val  # nan / inf
    return format(val, ".10g")

def _norm(c):
    tag = c[0]
    if tag in ("b", "i", "f"):
        return ["#", _numkey(c)]
    if tag in ("l", "t", "S"):
        return [tag, [_norm(x) for x in c[1]]]
    if tag == "d":
        return [tag, [[_norm(k), _norm(v)] for k, v in c[1]]]
    return c

def _maybe_digest(c, big):
    s = json.dumps(c)
    if len(s) <= big:
        return c
    # Huge outputs (HumanEval/130 returns ~1e6 numbers) travel as a fingerprint.
    return ["H", hashlib.sha256(json.dumps(_norm(c)).encode()).hexdigest()]

class _Timeout(BaseException):
    pass

def _on_alarm(signum, frame):
    raise _Timeout()

def _emit(obj):
    _real_stdout.write("\n" + _MARKER + json.dumps(obj))
    _real_stdout.flush()

def _run_stdin(compiled, inputs, limits, big):
    # Whole-program mode (AtCoder/Codeforces): fresh globals per test, stdin fed from the
    # test input, stdout captured as the output. exit()/sys.exit() ends the program normally.
    results = []
    for text, limit in zip(inputs, limits):
        out_buf = io.StringIO()
        sys.stdin = io.TextIOWrapper(io.BytesIO(text.encode()), encoding="utf-8")
        sys.stdout = out_buf
        status, err, detail = "ok", None, ""
        t0 = time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, limit)
            exec(compiled, {"__name__": "__main__"})
            signal.setitimer(signal.ITIMER_REAL, 0)
        except _Timeout:
            status = "timeout"
        except SystemExit:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            status, err, detail = "exception", type(e).__name__, str(e)[:200]
        dt = time.perf_counter() - t0
        sys.stdout = io.StringIO()
        if status == "ok":
            # Whitespace is normalised as LiveCodeBench's grader ignores it (strip each line).
            norm = "\n".join(line.strip() for line in out_buf.getvalue().strip().split("\n"))
            results.append({"status": "ok", "out": _maybe_digest(["s", norm], big), "dt": dt})
        elif status == "timeout":
            results.append({"status": "timeout"})
        else:
            results.append({"status": "exception", "error": err, "detail": detail})
    return results

def _main():
    payload = pickle.loads(sys.stdin.buffer.read())
    code = payload["code"]
    entry = payload["entry_point"]
    inputs = payload["inputs"]
    per_call = float(payload["per_call_timeout"])
    limits = payload.get("timeouts") or [per_call] * len(inputs)
    big = int(payload.get("big_output_bytes") or 1_000_000)
    mode = payload.get("mode") or "function"
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        _emit({"fatal": "syntax_error", "detail": str(e)[:300]})
        return
    signal.signal(signal.SIGALRM, _on_alarm)
    if mode == "stdin":
        _emit({"results": _run_stdin(compiled, inputs, limits, big)})
        return
    ns = {"__name__": "__candidate__"}
    try:
        signal.setitimer(signal.ITIMER_REAL, per_call * 2)
        exec(compiled, ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except _Timeout:
        _emit({"fatal": "timeout", "detail": "module level"})
        return
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _emit({"fatal": "load_error", "detail": (type(e).__name__ + ": " + str(e))[:300]})
        return
    if mode == "method":  # LeetCode style: a fresh Solution() per call
        cls = ns.get("Solution")
        if cls is None or not callable(getattr(cls, entry, None)):
            _emit({"fatal": "missing_entry_point", "detail": "Solution." + str(entry)})
            return
        make_fn = lambda: getattr(cls(), entry)
    else:
        fn = ns.get(entry)
        if not callable(fn):
            _emit({"fatal": "missing_entry_point", "detail": entry})
            return
        make_fn = lambda: fn
    results = []
    for args, limit in zip(inputs, limits):
        try:
            signal.setitimer(signal.ITIMER_REAL, limit)
            t0 = time.perf_counter()
            out = make_fn()(*copy.deepcopy(args))
            dt = time.perf_counter() - t0
            signal.setitimer(signal.ITIMER_REAL, 0)
            results.append({"status": "ok", "out": _maybe_digest(_canon(out), big), "dt": dt})
        except _Timeout:
            results.append({"status": "timeout"})
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            results.append({"status": "exception", "error": type(e).__name__, "detail": str(e)[:200]})
    _emit({"results": results})

_main()
"""


@dataclass
class RunResult:
    """Outcome of running one program on a list of inputs."""

    fatal: str | None = None  # syntax_error | timeout | load_error | missing_entry_point | crashed
    detail: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)


def _limits(memory_mb: int, cpu_s: int):
    def apply() -> None:
        import resource

        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        os.setsid()

    return apply


def run_isolated(cmd: list[str], timeout: float, **kw: Any) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, capture_output=True, ...)`` for a child that calls ``os.setsid``
    (see ``_limits``): on timeout the child's whole process group is killed, so processes the
    program started do not outlive it, and ``TimeoutExpired`` is raised as by ``run``."""
    data = kw.pop("input", None)
    if data is not None:
        kw["stdin"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    try:
        out, err = proc.communicate(data, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_program(
    code: str,
    entry_point: str,
    inputs: list[list[Any]],
    per_call_timeout: float = 2.0,
    memory_mb: int = 2048,
    total_timeout: float | None = None,
    timeouts: list[float] | None = None,
    mode: str = "function",
) -> RunResult:
    """Run ``entry_point`` from ``code`` on each argument list in ``inputs``.

    ``mode``: "function" calls ``entry_point(*args)``; "method" calls
    ``Solution().entry_point(*args)`` (LeetCode); "stdin" runs the whole program with each
    input string on stdin and returns its stdout (AtCoder/Codeforces).
    ``timeouts`` optionally gives a per-input time limit (EvalPlus style); otherwise every
    call gets ``per_call_timeout``. ``total_timeout`` caps the whole run.
    """
    if timeouts is not None and len(timeouts) != len(inputs):
        raise ValueError("timeouts must have one entry per input")
    if total_timeout is None:
        per_input = sum(timeouts) if timeouts else per_call_timeout * len(inputs)
        total_timeout = per_input + 2 * per_call_timeout + 5.0
    # Inputs travel parent -> child with pickle: they are our own benchmark data and can
    # hold complex numbers, sets, tuples. Outputs come back as canonical JSON only.
    payload = pickle.dumps(
        {
            "code": code,
            "entry_point": entry_point,
            "inputs": inputs,
            "per_call_timeout": per_call_timeout,
            "timeouts": timeouts,
            "mode": mode,
            "big_output_bytes": 1_000_000,
        }
    )
    # Fixed hash seed so set/dict iteration order is identical across runs (do not use
    # python -I here: isolated mode ignores PYTHONHASHSEED).
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="codeladder-") as tmp:
        try:
            proc = run_isolated(
                [sys.executable, "-s", "-B", "-c", _RUNNER],
                input=payload,
                timeout=total_timeout,
                cwd=tmp,
                env=env,
                preexec_fn=_limits(memory_mb, int(total_timeout) + 1),
            )
        except subprocess.TimeoutExpired:
            return RunResult(fatal="timeout", detail="total timeout")
    stdout = proc.stdout.decode("utf-8", errors="replace")
    idx = stdout.rfind(_MARKER)
    if idx < 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        return RunResult(fatal="crashed", detail=stderr[-300:])
    try:
        obj = json.loads(stdout[idx + len(_MARKER) :])
    except json.JSONDecodeError:
        return RunResult(fatal="crashed", detail="unparseable child output")
    if "fatal" in obj:
        return RunResult(fatal=obj["fatal"], detail=obj.get("detail", ""))
    return RunResult(results=obj["results"])


def canon(x: Any) -> list[Any]:
    """Canonical form of a Python value, as the sandbox encodes results (for JSON-like values:
    the sandbox also encodes complex numbers, bytes and sets, which callers here never pass)."""
    if x is None:
        return ["n"]
    if isinstance(x, bool):
        return ["b", x]
    if isinstance(x, int):
        return ["i", str(x)]
    if isinstance(x, float):
        if math.isnan(x):
            return ["f", "nan"]
        if math.isinf(x):
            return ["f", "inf" if x > 0 else "-inf"]
        return ["f", x]
    if isinstance(x, str):
        return ["s", x]
    if isinstance(x, tuple):
        return ["t", [canon(v) for v in x]]
    if isinstance(x, list):
        return ["l", [canon(v) for v in x]]
    if isinstance(x, dict):
        items = [[canon(k), canon(v)] for k, v in x.items()]
        return ["d", sorted(items, key=lambda kv: json.dumps(kv[0], sort_keys=True))]
    return ["r", repr(x)[:1000]]


def normalise_stdout(text: str) -> str:
    """Strip every line and surrounding blank lines, as the sandbox does for stdin programs."""
    return "\n".join(line.strip() for line in text.strip().split("\n"))


def digest_if_big(c: list[Any], big: int = 1_000_000) -> list[Any]:
    """Same fingerprinting the sandbox applies to outputs larger than ``big`` bytes."""
    if len(json.dumps(c)) <= big:
        return c
    import hashlib

    return ["H", hashlib.sha256(json.dumps(_norm_parent(c)).encode()).hexdigest()]


def _norm_parent(c: list[Any]) -> list[Any]:
    tag = c[0]
    if tag in ("b", "i", "f"):
        val = c[1]
        if tag == "b":
            key = format(float(val), ".10g")
        elif tag == "i":
            key = format(float(val), ".10g") if len(val) < 16 else val
        else:
            key = val if isinstance(val, str) else format(val, ".10g")
        return ["#", key]
    if tag in ("l", "t", "S"):
        return [tag, [_norm_parent(x) for x in c[1]]]
    if tag == "d":
        return [tag, [[_norm_parent(k), _norm_parent(v)] for k, v in c[1]]]
    return c


_NUMERIC = {"b", "i", "f"}


def _as_float(c: list[Any]) -> float:
    tag, val = c[0], c[1]
    if tag == "b":
        return float(val)
    if tag == "i":
        return float(val)  # parses the decimal string; huge ints become inf, no digit limit
    if isinstance(val, str):  # "nan", "inf", "-inf"
        return float(val)
    return float(val)


def _int_text(c: list[Any]) -> str:
    if c[0] == "b":
        return "1" if c[1] else "0"
    return c[1]


def outputs_match(got: list[Any], exp: list[Any], atol: float = 0.0) -> bool:
    """Compare two canonical outputs, following Python equality plus float tolerance."""
    tg, te = got[0], exp[0]
    if tg == "H" or te == "H":  # fingerprints of huge outputs: exact (normalised) match
        return got == exp
    if tg in _NUMERIC and te in _NUMERIC:
        if tg in ("b", "i") and te in ("b", "i"):
            # Compare canonical decimal strings (no int() call: avoids Python's 4300-digit
            # limit on huge answers). Booleans map to "1"/"0" so True == 1 as in Python.
            return _int_text(got) == _int_text(exp)
        a, b = _as_float(got), _as_float(exp)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        # EvalPlus enforces atol=1e-6 for float outputs when the task sets none.
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=atol if atol > 0 else 1e-6)
    if tg != te:
        return False
    if tg in ("l", "t", "S"):
        a, b = got[1], exp[1]
        return len(a) == len(b) and all(
            outputs_match(x, y, atol) for x, y in zip(a, b, strict=True)
        )
    if tg == "d":
        a, b = got[1], exp[1]
        return len(a) == len(b) and all(
            outputs_match(ka, kb, 0.0) and outputs_match(va, vb, atol)
            for (ka, va), (kb, vb) in zip(a, b, strict=True)
        )
    if tg == "c":
        return all(
            math.isclose(x, y, rel_tol=1e-7, abs_tol=atol)
            for x, y in zip(got[1], exp[1], strict=True)
        )
    return got == exp
