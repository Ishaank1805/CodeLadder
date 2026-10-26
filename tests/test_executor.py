from codeladder.executor import outputs_match, run_program

ADD = "def add(a, b):\n    return a + b\n"


def test_runs_correct_code():
    r = run_program(ADD, "add", [[1, 2], [3, 4]])
    assert r.fatal is None
    assert [x["out"] for x in r.results] == [["i", "3"], ["i", "7"]]


def test_syntax_error_is_fatal():
    r = run_program("def add(a, b)\n    return a+b\n", "add", [[1, 2]])
    assert r.fatal == "syntax_error"


def test_missing_entry_point():
    r = run_program(ADD, "sub", [[1, 2]])
    assert r.fatal == "missing_entry_point"


def test_exception_and_timeout_per_input():
    code = (
        "def f(x):\n"
        "    if x == 0:\n        raise ValueError('zero')\n"
        "    if x < 0:\n        while True: pass\n"
        "    return x\n"
    )
    r = run_program(code, "f", [[1], [0], [-1], [2]], per_call_timeout=0.5)
    statuses = [x["status"] for x in r.results]
    assert statuses == ["ok", "exception", "timeout", "ok"]
    assert r.results[1]["error"] == "ValueError"


def test_candidate_prints_do_not_corrupt_results():
    code = "def f(x):\n    print('noise', x)\n    return x * 2\n"
    r = run_program(code, "f", [[2]])
    assert r.results[0]["out"] == ["i", "4"]


def test_inputs_are_not_shared_between_calls():
    code = "def f(xs):\n    xs.append(1)\n    return len(xs)\n"
    r = run_program(code, "f", [[[0]], [[0]]])
    assert [x["out"] for x in r.results] == [["i", "2"], ["i", "2"]]


def test_outputs_match_semantics():
    assert outputs_match(["f", 0.1 + 0.2], ["f", 0.3])
    assert outputs_match(["i", "1"], ["f", 1.0])
    assert outputs_match(["b", True], ["i", "1"])  # True == 1 in Python
    assert not outputs_match(["t", [["i", "1"]]], ["l", [["i", "1"]]])  # tuple != list
    assert outputs_match(["f", 1.0], ["f", 1.05], atol=0.1)
    assert not outputs_match(["f", 1.0], ["f", 1.05])
    assert outputs_match(["f", "nan"], ["f", "nan"])


def test_huge_integers_round_trip():
    code = "def f(n):\n    return 10 ** n\n"
    r = run_program(code, "f", [[9000]])
    assert r.results[0]["status"] == "ok"
    big = r.results[0]["out"]
    assert outputs_match(big, big)
    assert not outputs_match(big, ["i", "1" + "0" * 8999])


def test_complex_and_set_inputs():
    code = "def f(z, s):\n    return (z * 2, sorted(s))\n"
    r = run_program(code, "f", [[1 + 2j, {3, 1, 2}]])
    assert r.results[0]["out"] == [
        "t",
        [["c", [2.0, 4.0]], ["l", [["i", "1"], ["i", "2"], ["i", "3"]]]],
    ]


def test_huge_outputs_are_fingerprinted_with_python_number_semantics():
    as_float = "def f(n):\n    return [i / 1 for i in range(n)]\n"
    as_int = "def f(n):\n    return list(range(n))\n"
    off_by_one = "def f(n):\n    return list(range(1, n + 1))\n"
    outs = [
        run_program(c, "f", [[200_000]]).results[0]["out"] for c in (as_float, as_int, off_by_one)
    ]
    assert outs[0][0] == "H"  # too big to ship: fingerprint
    assert outputs_match(outs[0], outs[1])  # 3.0 == 3, as in Python
    assert not outputs_match(outs[0], outs[2])


def test_timeout_kills_the_whole_process_group(tmp_path):
    import subprocess
    import sys
    import time

    from codeladder.executor import _limits, run_isolated

    marker = tmp_path / "pid"
    code = (
        "import subprocess, time\n"
        f"p = subprocess.Popen(['sleep', '30'])\nopen({str(marker)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(30)\n"
    )
    try:
        run_isolated([sys.executable, "-c", code], timeout=2, preexec_fn=_limits(512, 10))
        raise AssertionError("expected a timeout")
    except subprocess.TimeoutExpired:
        pass
    pid = int(marker.read_text())
    time.sleep(0.2)
    alive = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert alive.stdout.strip() in ("", "Z"), "the grandchild outlived the timeout"
