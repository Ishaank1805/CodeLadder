"""LiveCodeBench support: stdin and method execution modes, oracles, loader, prompts."""

import gzip
import json

from codeladder.executor import run_program
from codeladder.generate import evaluate_code
from codeladder.oracles import check, lcb_stdin_match
from codeladder.prompts import extract_code, initial_messages, repair_messages
from codeladder.tasks import load_livecodebench, prepare_task

EXEC = {"per_call_timeout_s": 2.0, "memory_mb": 1024, "hidden_chunk": 250}

STDIN_OK = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"
STDIN_INPUT_FN = "a, b = map(int, input().split())\nprint(a + b)\nexit()\n"
STDIN_BAD = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a - b)\n"
METHOD_OK = "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen:\n                return [seen[target - x], i]\n            seen[x] = i\n"


def test_stdin_mode_feeds_input_and_captures_output():
    r = run_program(STDIN_OK, None, ["1 2\n", "10 -3\n"], mode="stdin")
    assert [x["out"] for x in r.results] == [["s", "3"], ["s", "7"]]
    r = run_program(STDIN_INPUT_FN, None, ["4 5\n"], mode="stdin")  # input() and exit()
    assert r.results[0]["out"] == ["s", "9"]


def test_stdin_programs_get_fresh_globals_and_buffer():
    code = "import sys\ncount = globals().get('count', 0) + 1\nprint(count, sys.stdin.buffer.read().decode().strip())\n"
    r = run_program(code, None, ["x", "y"], mode="stdin")
    assert [x["out"][1] for x in r.results] == ["1 x", "1 y"]


def test_method_mode_calls_fresh_solution_instance():
    r = run_program(METHOD_OK, "twoSum", [[[2, 7, 11, 15], 9]], mode="method")
    assert r.results[0]["out"] == ["l", [["i", "0"], ["i", "1"]]]
    missing = run_program(METHOD_OK, "threeSum", [[[1], 1]], mode="method")
    assert missing.fatal == "missing_entry_point"


def test_lcb_stdin_grading_rules():
    assert lcb_stdin_match("3\n", "3")
    assert lcb_stdin_match("1 2 \n3", "1 2\n3\n")
    assert lcb_stdin_match("0.50", "0.5")  # exact decimal comparison
    assert not lcb_stdin_match("3", "4")
    assert not lcb_stdin_match("3\n4", "3")


def test_lcb_func_oracle_treats_tuples_as_lists():
    assert check("lcb_func", [], ["t", [["i", "1"]]], ["l", [["i", "1"]]], 0.0)
    assert not check("lcb_func", [], ["l", [["i", "2"]]], ["l", [["i", "1"]]], 0.0)


def _write_lcb(root, mode):
    (root / "tasks").mkdir(parents=True)
    if mode == "stdin":
        tests = {"public": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                 "private": [{"input": "5 5\n", "output": "10\n", "testtype": "stdin"},
                             {"input": "-1 1\n", "output": "0\n", "testtype": "stdin"}]}  # fmt: skip
        entry, starter = None, ""
    else:
        tests = {"public": [{"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "testtype": "functional"}],
                 "private": [{"input": "[3, 2, 4]\n6", "output": "[1, 2]", "testtype": "functional"}]}  # fmt: skip
        entry, starter = "twoSum", "class Solution:\n    def twoSum(self, nums, target):\n        "
    (root / "tasks" / "q1.json.gz").write_bytes(gzip.compress(json.dumps(tests).encode()))
    index = [{"task_id": "lcb/q1", "question": "Solve it.", "starter_code": starter, "mode": mode,
              "entry_point": entry, "release_date": "2025-01-04T00:00:00", "difficulty": "easy",
              "platform": "atcoder", "n_public": 1, "n_private": len(tests["private"]), "file": "q1.json.gz"}]  # fmt: skip
    (root / "index.json").write_text(json.dumps(index))


def test_loader_and_labels_stdin(tmp_path):
    _write_lcb(tmp_path, "stdin")
    task = prepare_task(load_livecodebench(tmp_path)[0])
    assert task.mode == "stdin" and task.oracle == "lcb_stdin" and task.time_limit_s == 6.0
    assert task.release_date.startswith("2025-01-04")
    good = evaluate_code(task, STDIN_OK, [], EXEC)
    bad = evaluate_code(task, STDIN_BAD, [], EXEC)
    assert good["hidden_pass"] and good["visible_pass_rate"] == 1.0
    assert not bad["hidden_pass"] and bad["error_type"] == "wrong_answer"
    repair = repair_messages(task, STDIN_BAD, bad["feedback"])[-1]["content"]
    assert "Expected output:\n3" in repair and "1 2" in repair
    assert "standard input" in initial_messages(task)[1]["content"]


def test_loader_and_labels_method(tmp_path):
    _write_lcb(tmp_path, "method")
    task = prepare_task(load_livecodebench(tmp_path)[0])
    assert task.mode == "method" and task.entry_point == "twoSum"
    assert evaluate_code(task, METHOD_OK, [], EXEC)["hidden_pass"]
    assert "class Solution" in initial_messages(task)[1]["content"]
    code = extract_code("```python\n" + METHOD_OK + "```", task)
    assert code.startswith("class Solution")  # no grafting for non-function modes


def test_repair_feedback_is_clipped_for_huge_inputs():
    from codeladder.prompts import feedback_text
    from codeladder.types import Task

    task = Task(task_id="x", benchmark="livecodebench", prompt="p", entry_point="f", reference=None,
                visible_inputs=[], hidden_inputs=[], mode="method")  # fmt: skip
    fb = {
        "kind": "wrong_answer",
        "input": [repr(list(range(50_000)))],
        "expected": ["i", "1"],
        "got": ["i", "2"],
    }
    text = feedback_text(task, fb)
    assert len(text) < 8100 and "more characters" in text
    small = {"kind": "wrong_answer", "input": ["[1, 2]"], "expected": ["i", "1"], "got": ["i", "2"]}
    assert "more characters" not in feedback_text(task, small)


def test_release_tests_makes_prepare_reload(tmp_path):
    from test_pipeline import toy_tasks

    from codeladder.tasks import prepare_task, release_tests

    task = toy_tasks()[0]  # loader-provided tests are kept
    n = len(task.hidden_inputs)
    release_tests(task)
    assert len(task.hidden_inputs) == n and task.visible_expected is not None

    _write_lcb(tmp_path, "stdin")
    lcb = load_livecodebench(tmp_path)[0]
    prepare_task(lcb)
    n_vis, n_hid = len(lcb.visible_inputs), len(lcb.hidden_inputs)
    release_tests(lcb)
    assert lcb.hidden_inputs == [] and lcb.visible_expected is None
    prepare_task(lcb)  # reloaded from disk instead of being treated as prepared
    assert (len(lcb.visible_inputs), len(lcb.hidden_inputs)) == (n_vis, n_hid) and n_hid > 0
