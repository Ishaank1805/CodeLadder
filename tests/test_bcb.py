"""X18 BigCodeBench: unit-test runner, candidate fields, prompts (stdlib-only tasks, so the
tests run with the current Python instead of the benchmark environment)."""

import json
import sys

import pytest

from codeladder.bcb import evaluate_bcb, method_source, run_bcb, unit_test_methods
from codeladder.prompts import extract_code, feedback_text
from codeladder.types import Task

TESTS = """import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self):
        self.assertEqual(task_func([3, 1, 2]), 3)
    def test_case_2(self):
        self.assertEqual(task_func([-5, -2]), -2)
    def test_case_3(self):
        with self.assertRaises(ValueError):
            task_func([])
"""
PROMPT = 'import math\n\ndef task_func(xs):\n    """Return the largest value; raise ValueError if empty."""\n'
PROBES = (
    "def probes():\n    yield lambda: task_func([1, 9, 4])\n    yield lambda: task_func([-1, -7])\n"
)
GOOD = "import math\ndef task_func(xs):\n    if not xs:\n        raise ValueError\n    return max(xs)\n"
NEG_BUG = "def task_func(xs):\n    best = 0\n    for x in xs:\n        best = max(best, x)\n    if not xs:\n        raise ValueError\n    return best\n"


@pytest.fixture(autouse=True)
def _python(monkeypatch):
    monkeypatch.setenv("CODELADDER_BCB_PYTHON", sys.executable)


def _task():
    return Task("BigCodeBench/x", "bigcodebench", PROMPT, "task_func", PROMPT + "    return max(xs)\n",
                [], [], visible_expected=[], hidden_expected=[], mode="bcb", tests=TESTS,
                probe_code=PROBES)  # fmt: skip


def test_methods_and_source():
    assert unit_test_methods(TESTS) == ["test_case_1", "test_case_2", "test_case_3"]
    assert "task_func([3, 1, 2])" in method_source(TESTS, "test_case_1")
    assert "test_case_2" not in method_source(TESTS, "test_case_1")


def test_methods_only_from_testcases_and_multiline_signatures():
    tests = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_b(self):\n        pass\n"
        "    def test_a(\n        self, mock_x=None\n    ):\n        pass\n"
        "    def helper(self):\n        pass\n"
        "class OtherCases(unittest.TestCase):\n"
        "    def test_other(self):\n        pass\n"
    )
    assert unit_test_methods(tests) == ["test_b", "test_a"]
    assert "mock_x" in method_source(tests, "test_a")
    assert unit_test_methods("    def test_x(self):\n  oops(") == ["test_x"]  # unparsable


def test_runner_reports_each_method_and_probes():
    res = run_bcb(GOOD, TESTS, None, PROBES)
    assert res["fatal"] is None, res
    assert {m: r["status"] for m, r in res["methods"].items()} == dict.fromkeys(
        unit_test_methods(TESTS), "pass"
    )
    assert res["probes"] == ["9", "-1"]
    res = run_bcb("def task_func(:\n", TESTS)
    assert res["fatal"] == "syntax_error"
    slow = "def task_func(xs):\n    while True:\n        pass\n"
    res = run_bcb(slow, TESTS, ["test_case_1"], exec_cfg={"bcb_method_timeout_s": 1})
    assert res["methods"]["test_case_1"]["status"] == "timeout"


def test_candidate_fields_split_visible_and_hidden():
    task = _task()
    good = evaluate_bcb(task, GOOD, {})
    assert good["visible_pass_rate"] == 1.0 and good["hidden_pass"] and good["signature"], good
    bug = evaluate_bcb(task, NEG_BUG, {})  # passes the visible test, fails a hidden one
    assert bug["visible_pass_rate"] == 1.0 and not bug["hidden_pass"]
    assert bug["signature"] != good["signature"]  # the probes tell them apart
    broken = evaluate_bcb(task, "def task_func(xs):\n    return min(xs)\n", {})
    assert broken["visible_pass_rate"] == 0.0 and broken["feedback"]["kind"] == "test_failure"
    text = feedback_text(task, broken["feedback"])
    assert "task_func([3, 1, 2])" in text and "AssertionError" in text


def test_extract_code_keeps_the_task_imports():
    code = extract_code(
        "```python\ndef task_func(xs):\n    return math.floor(max(xs))\n```", _task()
    )
    assert code.startswith("import math\n")


def test_make_probes_keeps_compiling_programs(tmp_path):
    from codeladder.bcb import make_bcb_probes
    from codeladder.llm import Completion

    class Writer:
        def complete(self, messages, **kw):
            assert "Do not compute or check expected results" in messages[1]["content"]
            if "broken" in messages[1]["content"]:
                return Completion("```python\ndef probes(:\n```", 10, 5, 0.0)
            return Completion(f"```python\n{PROBES}```", 10, 5, 0.0)

    ok, bad = _task(), _task()
    bad.task_id, bad.prompt = "BigCodeBench/y", PROMPT + "    # broken\n"
    out = tmp_path / "probes.json"
    stats = make_bcb_probes([ok, bad], Writer(), out_path=out, workers=2)
    assert stats == {"tasks": 2, "with_probes": 1}
    assert "def probes()" in json.loads(out.read_text())["BigCodeBench/x"]


def test_runner_matches_the_official_harness():
    # tests that patch via __name__, a class attribute named test_*, multiprocessing by module
    code = (
        "from random import randint\nimport multiprocessing\n"
        "def _sq(x):\n    return x * x\n"
        "def task_func(n):\n"
        "    with multiprocessing.get_context('fork').Pool(2) as p:\n"
        "        return [randint(0, 9)] + p.map(_sq, range(n))\n"
    )
    tests = (
        "import unittest\nfrom unittest.mock import patch\n"
        "class TestCases(unittest.TestCase):\n"
        "    test_data_dir = 'mnt/data'\n"
        "    @patch(__name__ + '.randint', return_value=7)\n"
        "    def test_case_1(self, mock_randint):\n"
        "        self.assertEqual(task_func(3), [7, 0, 1, 4])\n"
    )
    res = run_bcb(code, tests)
    assert res["fatal"] is None, res
    assert res["methods"] == {"test_case_1": {"status": "pass", "detail": ""}}, res


def test_generation_stops_when_the_server_is_gone(tmp_path):
    from codeladder.generate import generate_pools

    class Dead:
        calls = 0

        def complete(self, messages, **kw):
            Dead.calls += 1
            raise RuntimeError("Connection error.")

    tasks = []
    for i in range(40):
        t = _task()
        t.task_id = f"BigCodeBench/{i}"
        tasks.append(t)
    cfg = {"paths": {"pools": str(tmp_path)}, "generation": {"n_samples": 2, "temperature": 0.8,
           "top_p": 0.95, "max_tokens": 64, "seed": 0, "repair": {"parents": 0, "depth": 0}},
           "probes": {"n": 0, "seed": 0}, "execution": {}}  # fmt: skip
    written = generate_pools(tasks, "t1", Dead(), cfg, workers=2, max_consecutive_failures=5)
    assert written == [] and Dead.calls < 60  # stopped early instead of 40 tasks x 3 attempts


def test_generated_code_cannot_signal_processes_outside_its_session():
    import signal
    import subprocess

    outside = subprocess.Popen(["sleep", "41.5"])  # stands in for the model server
    code = (
        "import os, signal, subprocess, time\n"
        f"VICTIM = {outside.pid}\n"
        "def task_func():\n"
        "    blocked = 0\n"
        "    try:\n"
        "        os.kill(VICTIM, signal.SIGTERM)\n"
        "    except PermissionError:\n"
        "        blocked += 1\n"
        "    subprocess.run(['pkill', '-f', 'sleep 41.5'])\n"
        "    subprocess.run(['kill', str(VICTIM)], stderr=subprocess.DEVNULL)\n"
        "    subprocess.run(['killall', 'sleep'], stderr=subprocess.DEVNULL)\n"
        "    own = subprocess.Popen(['sleep', '30'])\n"
        "    os.kill(own.pid, signal.SIGTERM)\n"  # its own child: allowed
        "    own.wait(timeout=5)\n"
        "    return blocked\n"
    )
    tests = (
        "import unittest\nclass TestCases(unittest.TestCase):\n"
        "    def test_case_1(self):\n        self.assertEqual(task_func(), 1)\n"
    )
    try:
        res = run_bcb(code, tests)
        assert res["methods"]["test_case_1"]["status"] == "pass", res
        assert outside.poll() is None  # still running
    finally:
        outside.send_signal(signal.SIGKILL)
        outside.wait()
