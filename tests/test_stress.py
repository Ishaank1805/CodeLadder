"""Model-written stress inputs for stdin tasks."""

from conftest import cand

from codeladder.generate import stress_signature
from codeladder.llm import Completion
from codeladder.scoring import phi_scores
from codeladder.stress import load_stress_inputs, make_stress_inputs, plausible, run_generator
from codeladder.types import Task

EXEC = {"per_call_timeout_s": 2.0, "memory_mb": 1024, "memory_mb_hidden": 1024}

GEN = "import random\nrandom.seed(1)\nn = 20000\nprint(n)\nprint(*[random.randint(1, 10**9) for _ in range(n)])\n"


def stdin_task(**kw) -> Task:
    base = dict(
        task_id="lcb/x", benchmark="livecodebench", prompt="Sum the N numbers.\n\nInput\n\nN\nA_1 ... A_N",
        entry_point=None, reference=None, visible_inputs=["3\n1 2 3\n"], hidden_inputs=[],
        visible_expected=[["s", "6"]], hidden_expected=[], mode="stdin", time_limit_s=1.0,
    )  # fmt: skip
    base.update(kw)
    return Task(**base)


def test_run_generator_returns_output_or_none():
    out = run_generator(GEN)
    assert out and out.split("\n")[0] == "20000" and len(out.split("\n")[1].split()) == 20000
    assert run_generator("raise SystemExit(1)") is None
    assert run_generator("while True: pass", timeout_s=1) is None


def test_plausible_checks_the_first_line_shape():
    assert plausible("5\n1 2 3 4 5\n", ["3\n1 2 3\n"])
    assert not plausible("5 7\n1 2\n", ["3\n1 2 3\n"])
    assert not plausible("abc\n", ["3\n1 2 3\n"])


class GenClient:
    def __init__(self, code):
        self.code, self.calls = code, 0

    def complete(self, messages, *, temperature, top_p, max_tokens, seed):
        self.calls += 1
        return Completion(
            text=f"```python\n{self.code}```", tokens_in=100, tokens_out=50, latency_s=0.0
        )


def test_make_stress_inputs_is_resumable(tmp_path, monkeypatch):
    import codeladder.tasks

    monkeypatch.setattr(codeladder.tasks, "prepare_task", lambda t: t)
    task, client = stdin_task(), GenClient(GEN)
    counts = make_stress_inputs([task], client, tmp_path, "m", n_generators=3, keep=2, workers=1)
    assert counts["ok"] == 1
    assert len(load_stress_inputs(tmp_path, task)) == 1  # identical outputs are kept once
    assert client.calls == 3
    make_stress_inputs([task], client, tmp_path, "m", workers=1)
    assert client.calls == 3  # nothing left to do


def test_slow_programs_split_from_fast_ones_on_stress_inputs():
    task = stdin_task()
    big = [run_generator(GEN)]
    fast = "n = int(input())\nprint(sum(map(int, input().split())))\n"
    slow = "n = int(input())\na = list(map(int, input().split()))\nt = 0\nfor i in range(n):\n    for j in range(n):\n        t += 1\nprint(sum(a))\n"
    s_fast = stress_signature(task, fast, big, EXEC)
    s_slow = stress_signature(task, slow, big, EXEC)
    assert s_fast != s_slow and "timeout" in s_slow
    a, b, c = (cand("t1", i, visible=True, hidden=True) for i in range(3))
    a.extra["stress_sig"], b.extra["stress_sig"], c.extra["stress_sig"] = s_fast, s_fast, s_slow
    assert phi_scores([a, b, c]) == [2 / 3, 2 / 3, 0.0]  # a stress timeout fails the checks
    d = cand("t1", 3, visible=True, hidden=True, sig="mutated")
    d.extra["probes"] = "stdin-v2"  # stdin task without stress inputs: visible share only
    e = cand("t1", 4, visible=True, hidden=True, sig="other")
    e.extra["probes"] = "stdin-v2"
    assert phi_scores([d, e]) == [1.0, 1.0]


def test_plausible_allows_digit_strings_where_publics_disagree():
    assert plausible("?0?1\n", ["?0?\n2\n", "101\n3\n"])
    assert not plausible("?0? 1\n", ["?0?\n2\n", "101\n3\n"])


def test_too_large_generators_get_one_fix_request(tmp_path, monkeypatch):
    import codeladder.tasks

    monkeypatch.setattr(codeladder.tasks, "prepare_task", lambda t: t)

    class FixingClient(GenClient):
        def complete(self, messages, *, temperature, top_p, max_tokens, seed):
            big = "print(3)\nprint('1 ' * 2_000_000)\n"
            self.code = GEN if len(messages) > 2 else big  # fixed once told it is too large
            return super().complete(
                messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
            )

    client = FixingClient(GEN)
    make_stress_inputs([stdin_task()], client, tmp_path, "m", n_generators=1, keep=1, workers=1)
    assert client.calls == 2 and len(load_stress_inputs(tmp_path, stdin_task())) == 1


def test_assert_lines_keep_only_valid_entry_point_asserts():
    from codeladder.baselines_gen import assert_lines

    task = stdin_task(mode="function", entry_point="add", prompt="def add(a, b):\n    pass\n")
    text = "```python\nassert add(1, 2) == 3\nassert add(1,\nprint(1)\nassert other(1) == 2\nassert add(-1, 1) == 0\n```"
    assert assert_lines(text, task) == ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]


def test_token_stats_entropy_and_margin():
    from types import SimpleNamespace as NS

    from codeladder.llm import _token_stats

    tok = NS(logprob=-0.1, top_logprobs=[NS(logprob=-0.1), NS(logprob=-2.4), NS(logprob=-5.0)])
    lp, ent, margin = _token_stats(tok)
    assert lp == -0.1 and 0 < ent < 0.6 and abs(margin - 2.3) < 1e-9
    assert _token_stats(NS(logprob=-0.5, top_logprobs=[]))[1] == 0.0
