"""X18: BigCodeBench (library-heavy function tasks with unit tests).

Tasks come from data/bcb/tasks.jsonl (``scripts/prepare_bcb.py``, release v0.1.4, the
"complete" prompts). Each task's unittest class is split as a developer would have it: the
first test method is the visible test (signals, repair feedback); all methods together are
the hidden test (label, the benchmark's own pass criterion). Programs run in the benchmark's
pinned environment (``~/venvs/bcb``, Python 3.10) through ``bcb_runner.py``: one subprocess
per program, in a fresh temporary directory, with memory limits and per-test time limits.

Agreement between candidates uses label-free probe calls written once per task by a fixed
model from the docstring (``make-bcb-probes``, like the stress inputs for stdin problems):
two candidates agree when the probes return the same normalised results.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .types import Task

BCB_DIR = Path(__file__).resolve().parents[2] / "data" / "bcb"
RUNNER = Path(__file__).resolve().parent / "bcb_runner.py"
_METHOD = re.compile(r"^\s+def (test\w*)\(self", re.MULTILINE)


def bcb_python() -> str:
    return os.environ.get("CODELADDER_BCB_PYTHON") or str(Path.home() / "venvs/bcb/bin/python")


def unit_test_methods(test_code: str) -> list[str]:
    """Test method names in source order (the first is the visible test)."""
    seen: list[str] = []
    for name in _METHOD.findall(test_code):
        if name not in seen:
            seen.append(name)
    return seen


def method_source(test_code: str, name: str) -> str:
    """Source of one test method (shown in repair feedback: it is the visible test)."""
    lines = test_code.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"\s+def {name}\(self", ln)), None)
    if start is None:
        return ""
    indent = len(lines[start]) - len(lines[start].lstrip())
    out = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            break
        out.append(ln)
    return "\n".join(out)


def load_bigcodebench(root: Path = BCB_DIR) -> list[Task]:
    path = root / "tasks.jsonl"
    excluded = set()
    if (root / "excluded.json").exists():  # reference fails in our environment
        excluded = set(json.loads((root / "excluded.json").read_text()))
    probes = {}
    if (root / "probes.json").exists():
        probes = json.loads((root / "probes.json").read_text())
    hard = set()
    if (root / "hard_ids.json").exists():
        hard = set(json.loads((root / "hard_ids.json").read_text()))
    tasks = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["task_id"] in excluded:
            continue
        methods = unit_test_methods(r["test"])
        if not methods:
            continue
        tasks.append(
            Task(
                task_id=r["task_id"],
                benchmark="bigcodebench",
                prompt=r["complete_prompt"],
                entry_point=r["entry_point"],
                reference=r["complete_prompt"] + r["canonical_solution"],
                visible_inputs=[],
                hidden_inputs=[],
                visible_expected=[],
                hidden_expected=[],
                mode="bcb",
                difficulty="hard" if r["task_id"] in hard else "full",
                tests=r["test"],
                probe_code=probes.get(r["task_id"]),
            )
        )
    return tasks


def _limits(as_bytes: int, data_bytes: int):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (data_bytes, data_bytes))

    return apply


def run_bcb(
    code: str,
    test_code: str,
    methods: list[str] | None = None,
    probes: str | None = None,
    exec_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a program against (some of) the unit tests and the probes, in the benchmark's
    environment. Returns the runner's JSON (fatal, methods, probes)."""
    cfg = exec_cfg or {}
    method_timeout = float(cfg.get("bcb_method_timeout_s", 60))
    load_timeout = float(cfg.get("bcb_load_timeout_s", 120))
    n_methods = len(methods) if methods else len(unit_test_methods(test_code))
    total = load_timeout + method_timeout * n_methods + (80 if probes else 0) + 30
    payload = json.dumps(
        {
            "code": code,
            "test": test_code,
            "methods": methods,
            "probes": probes,
            "method_timeout": method_timeout,
            "load_timeout": load_timeout,
            "probe_timeout": float(cfg.get("bcb_probe_timeout_s", 10)),
        }
    )
    gb = 1024**3
    work = tempfile.mkdtemp(prefix="bcb-")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": work,
        "TMPDIR": work,
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    proc = subprocess.Popen(
        [bcb_python(), str(RUNNER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=work,
        env=env,
        preexec_fn=_limits(
            int(cfg.get("bcb_as_gb", 32)) * gb, int(cfg.get("bcb_data_gb", 16)) * gb
        ),
    )
    try:
        out, _ = proc.communicate(payload.encode(), timeout=total)
        lines = [ln for ln in out.decode(errors="replace").splitlines() if ln.startswith("{")]
        res = json.loads(lines[-1]) if lines else {"fatal": "crashed", "detail": "no output"}
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        res = {"fatal": "timeout", "detail": "total time limit"}
    except (json.JSONDecodeError, ValueError) as e:
        res = {"fatal": "crashed", "detail": str(e)}
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        shutil.rmtree(work, ignore_errors=True)
    res.setdefault("methods", {})
    res.setdefault("probes", [])
    return res


_ERROR = {"syntax_error": "syntax_error", "load_error": "load_error", "timeout": "timeout",
          "crashed": "crashed"}  # fmt: skip


def evaluate_bcb(task: Task, code: str, exec_cfg: dict[str, Any]) -> dict[str, Any]:
    """Candidate fields for a BigCodeBench program: visible = first test method, hidden = all
    methods, signature = probe results (visible-passing programs only)."""
    methods = unit_test_methods(task.tests or "")
    res = run_bcb(code, task.tests or "", None, task.probe_code, exec_cfg)
    if res.get("fatal"):
        kind = _ERROR.get(res["fatal"], "crashed")
        return {
            "visible_pass_rate": 0.0,
            "error_type": kind,
            "feedback": {"kind": kind, "detail": str(res.get("detail", ""))[-1500:]},
            "signature": "",
            "hidden_pass": False,
        }
    first = res["methods"].get(methods[0], {"status": "error", "detail": "not run"})
    visible = first["status"] == "pass"
    status_map = {"fail": "wrong_answer", "error": "exception:Error", "timeout": "timeout"}
    feedback = None
    if not visible:
        feedback = {
            "kind": "test_failure",
            "test": method_source(task.tests or "", methods[0]),
            "detail": str(first.get("detail", ""))[-1500:],
        }
    hidden = visible and all(res["methods"].get(m, {}).get("status") == "pass" for m in methods)
    return {
        "visible_pass_rate": 1.0 if visible else 0.0,
        "error_type": "pass" if visible else status_map.get(first["status"], "exception:Error"),
        "feedback": feedback,
        "signature": json.dumps(res["probes"]) if visible and res["probes"] else "",
        "hidden_pass": hidden,
    }


PROBE_SYSTEM = (
    "You write test inputs for Python functions. Reply with one ```python code block and "
    "nothing else."
)


def probe_messages(task: Task) -> list[dict[str, str]]:
    """Ask a model for probe calls: inputs only, no expected outputs (label-free)."""
    return [
        {"role": "system", "content": PROBE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the specification of a Python function:\n\n"
                f"```python\n{task.prompt.rstrip()}\n```\n\n"
                f"Write a Python function `probes()` that yields up to 5 zero-argument "
                f"callables (lambdas or nested functions), each calling `{task.entry_point}` "
                "once with different valid arguments that exercise ordinary and edge cases "
                "described in the specification. If a call needs files, directories or data, "
                "create them inside the callable, in the current directory, before the call. "
                "Use only the standard library and the modules the specification imports. "
                "Do not compute or check expected results, do not import the function, and "
                "do not call probes() yourself."
            ),
        },
    ]


def make_bcb_probes(tasks: list[Task], client, out_path: Path = BCB_DIR / "probes.json",
                    workers: int = 8) -> dict[str, int]:  # fmt: skip
    """One probe program per task (resumable). Kept if it compiles and defines probes()."""
    from concurrent.futures import ThreadPoolExecutor

    from .prompts import _FENCE

    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [t for t in tasks if t.task_id not in done]

    def work(task: Task) -> tuple[str, str | None]:
        comp = client.complete(
            probe_messages(task), temperature=0.0, top_p=1.0, max_tokens=1024, seed=0
        )
        blocks = _FENCE.findall(comp.text or "")
        code = max(blocks, key=len) if blocks else ""
        try:
            compile(code, "probes.py", "exec")
        except SyntaxError:
            return task.task_id, None
        return task.task_id, code if "def probes(" in code else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, code in ex.map(work, todo):
            if code:
                done[tid] = code
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(done))
    return {"tasks": len(tasks), "with_probes": sum(t.task_id in done for t in tasks)}


def check_references(tasks: list[Task], exec_cfg: dict[str, Any], workers: int = 8,
                     out_path: Path = BCB_DIR / "excluded.json") -> dict[str, Any]:  # fmt: skip
    """Run every reference solution on all its tests in our environment; tasks whose
    reference does not pass are excluded (as BigCodeBench's own evaluation does)."""
    from concurrent.futures import ThreadPoolExecutor

    def work(task: Task) -> tuple[str, str]:
        res = run_bcb(task.reference, task.tests or "", None, None, exec_cfg)
        if res.get("fatal"):
            return task.task_id, f"{res['fatal']}: {str(res.get('detail', ''))[:200]}"
        bad = {m: r["status"] for m, r in res["methods"].items() if r["status"] != "pass"}
        return task.task_id, json.dumps(bad) if bad else ""

    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, why in ex.map(work, tasks):
            if why:
                failures[tid] = why
    out_path.write_text(json.dumps(sorted(failures)))
    (out_path.parent / "excluded_reasons.json").write_text(json.dumps(failures, indent=1))
    return {"tasks": len(tasks), "excluded": len(failures)}
