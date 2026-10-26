"""BigCodeBench test runner, executed by the benchmark's own Python (``~/venvs/bcb``).

Standalone (standard library plus whatever the task imports). Reads one JSON payload on
stdin: {"code", "test", "methods", "probes", "load_timeout", "method_timeout", "probe_timeout"}, runs in the
current directory (a fresh temporary directory chosen by the parent), and prints one JSON line:
{"fatal": str|null, "detail": str, "methods": {name: {"status", "detail"}}, "probes": [str]}.

* methods: the named unittest methods of ``TestCases`` (all of them if null), each with its
  own time limit; status is pass, fail (assertion), error (exception) or timeout.
* probes: ``probes()`` from the probe code yields zero-argument callables; each result (or
  exception type) is normalised to a short string for behavioural agreement.
"""

import io
import json
import os
import re
import signal
import sys
import traceback
import unittest


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _norm(x, depth=0):
    """Short, stable text for a probe result (no memory addresses, rounded floats)."""
    if depth > 3:
        return "..."
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            x = np.round(x, 6).tolist() if x.dtype.kind in "fc" else x.tolist()
        elif isinstance(x, np.generic):
            x = x.item()
    except Exception:
        pass
    try:
        import pandas as pd

        if isinstance(x, (pd.DataFrame, pd.Series)):
            return type(x).__name__ + ":" + x.round(6).to_csv()[:4000]
    except Exception:
        pass
    if isinstance(x, float):
        return repr(round(x, 6))
    if isinstance(x, (list, tuple)):
        return type(x).__name__ + "[" + ",".join(_norm(v, depth + 1) for v in list(x)[:200]) + "]"
    if isinstance(x, dict):
        items = sorted(((str(k), _norm(v, depth + 1)) for k, v in list(x.items())[:200]))
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(x, (str, int, bool, type(None), bytes)):
        return repr(x)[:4000]
    return re.sub(r" at 0x[0-9a-fA-F]+", "", type(x).__name__ + ":" + repr(x))[:2000]


def _fence_signals():
    """Generated code may only signal processes of its own session: no os.kill / killpg of the
    model server or other processes of the same user on the node (psutil uses os.kill too).
    Signal 0 (an existence check) stays allowed. Tests that mock these are unaffected."""
    sid = os.getsid(0)
    os.environ["BCB_SID"] = str(sid)
    real_kill, real_killpg = os.kill, os.killpg

    def ours(pid):
        try:
            return os.getsid(pid) == sid
        except OSError:
            return True  # no such process: let the real call report it

    def kill(pid, sig):
        if sig != 0 and (pid == -1 or (pid != 0 and not ours(abs(pid)))):
            raise PermissionError(1, "Operation not permitted")
        return real_kill(pid, sig)

    def killpg(pgid, sig):
        if sig != 0 and pgid != 0 and not ours(pgid):
            raise PermissionError(1, "Operation not permitted")
        return real_killpg(pgid, sig)

    os.kill, os.killpg = kill, killpg


def main():
    _fence_signals()
    payload = json.loads(sys.stdin.read())
    out = {"fatal": None, "detail": "", "methods": {}, "probes": []}
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # the candidate's prints must not corrupt the JSON line
    os.environ.setdefault("MPLBACKEND", "Agg")
    signal.signal(signal.SIGALRM, _alarm)
    # Like BigCodeBench's own harness: solution and tests run as one module named __test__,
    # registered in sys.modules (tests patch f"{__name__}.x"; multiprocessing pickles by module).
    import types

    module = types.ModuleType("__test__")
    module.__dict__.update(
        {
            "__file__": "__test__.py",
            "__package__": None,
            "__doc__": None,
            "sys": sys,
            "os": os,
            "environ": os.environ,
        }  # fmt: skip
    )
    sys.modules["__test__"] = module
    ns = module.__dict__
    try:
        signal.alarm(int(payload.get("load_timeout", 60)))
        full = payload["code"] + "\n" + payload["test"]
        exec(compile(full, "__test__.py", "exec"), ns)
        signal.alarm(0)
    except SyntaxError as e:
        out["fatal"], out["detail"] = "syntax_error", str(e)
    except _Timeout:
        out["fatal"], out["detail"] = "timeout", "loading the solution timed out"
    except BaseException as e:  # noqa: BLE001 (import errors, crashes at import time)
        out["fatal"], out["detail"] = "load_error", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
    if out["fatal"] is None:
        cls = ns.get("TestCases")
        names = payload.get("methods") or (
            list(unittest.TestLoader().getTestCaseNames(cls)) if cls else []
        )
        for name in names:
            res = unittest.TestResult()
            try:
                signal.alarm(int(payload.get("method_timeout", 30)))
                cls(name).run(res)
                signal.alarm(0)
            except _Timeout:
                out["methods"][name] = {"status": "timeout", "detail": "time limit"}
                continue
            except BaseException as e:  # noqa: BLE001
                out["methods"][name] = {"status": "error", "detail": f"{type(e).__name__}: {e}"}
                continue
            finally:
                signal.alarm(0)
            if res.failures:
                out["methods"][name] = {"status": "fail", "detail": res.failures[0][1][-2000:]}
            elif res.errors:
                detail = res.errors[0][1]
                status = "timeout" if "_Timeout" in detail else "error"
                out["methods"][name] = {"status": status, "detail": detail[-2000:]}
            elif res.skipped:
                out["methods"][name] = {"status": "pass", "detail": "skipped"}
            else:
                out["methods"][name] = {"status": "pass", "detail": ""}
        if payload.get("probes"):
            try:
                pns = dict(ns)
                exec(compile(payload["probes"], "probes.py", "exec"), pns)
                thunks = list(pns["probes"]())[:8]
            except BaseException as e:  # noqa: BLE001
                thunks = []
                out["probes"].append("probe-setup:" + type(e).__name__)
            for th in thunks:
                try:
                    import random

                    random.seed(0)
                    try:
                        import numpy as np

                        np.random.seed(0)
                    except Exception:
                        pass
                    signal.alarm(int(payload.get("probe_timeout", 10)))
                    val = _norm(th())
                    signal.alarm(0)
                except _Timeout:
                    val = "timeout"
                except BaseException as e:  # noqa: BLE001
                    val = "raise:" + type(e).__name__
                finally:
                    signal.alarm(0)
                out["probes"].append(val)
    sys.stdout = real_stdout
    print(json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        sys.stdout = sys.__stdout__
        print(
            json.dumps(
                {
                    "fatal": "crashed",
                    "detail": traceback.format_exc()[-2000:],
                    "methods": {},
                    "probes": [],
                }
            )
        )
