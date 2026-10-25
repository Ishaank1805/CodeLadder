"""X12 P(True) self-evaluation: prompt, client parsing, annotation and scores."""

import math
import sys
import types

from conftest import FakeClient, cand
from test_pipeline import EXEC, GEN, PROBES, SOLUTIONS, toy_tasks

from codeladder import llm
from codeladder.config import TierConfig
from codeladder.generate import annotate_ptrue, generate_pools, load_pools
from codeladder.llm import Completion
from codeladder.prompts import PTRUE_PREFILL, p_true, ptrue_messages
from codeladder.scoring import phi_scores


def test_p_true_normalises_yes_against_no():
    lp = {" Yes": math.log(0.6), "yes": math.log(0.1), " No": math.log(0.1), "Maybe": -1.0}
    assert abs(p_true(lp) - 0.7 / 0.8) < 1e-9
    assert p_true({" Yes": -0.01}) == 1.0
    assert p_true({"Hmm": -0.1}) is None and p_true(None) is None


def test_prompt_prefills_the_answer():
    msgs = ptrue_messages(toy_tasks()[0], "def add(a, b):\n    return a + b\n")
    assert msgs[-1] == {"role": "assistant", "content": PTRUE_PREFILL}
    assert "Answer Yes or No" in msgs[-2]["content"]


def test_client_keeps_first_token_alternatives(monkeypatch):
    seen = {}

    def tok(t, lp):
        return type("T", (), {"token": t, "logprob": lp, "top_logprobs": []})()

    first = type(
        "F",
        (),
        {"token": " Yes", "logprob": -0.2, "top_logprobs": [tok(" Yes", -0.2), tok(" No", -1.7)]},
    )()

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout, max_retries):
            self.chat = type("Ch", (), {"completions": self})()

        def create(self, **kw):
            seen.update(kw)
            msg = type("M", (), {"content": " Yes"})()
            lp = type("L", (), {"content": [first]})()
            resp = type("R", (), {})()
            resp.choices = [type("C", (), {"message": msg, "logprobs": lp})()]
            resp.usage = type("U", (), {"prompt_tokens": 50, "completion_tokens": 1})()
            return resp

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = llm.OpenAIChatClient(TierConfig(name="t1", model="m", params_b=1, base_url="http://x"))
    out = client.complete(
        [{"role": "user", "content": "q"}], temperature=0, top_p=1, max_tokens=1, seed=0,
        top_logprobs=10, continue_final=True,
    )  # fmt: skip
    assert out.first_top == {" Yes": -0.2, " No": -1.7}
    assert seen["top_logprobs"] == 10 and seen["extra_body"]["continue_final_message"] is True


def test_annotate_ptrue_and_scores(tmp_path):
    class Judge:
        def complete(self, messages, **kw):
            assert kw["max_tokens"] == 1 and kw["continue_final"] and kw["top_logprobs"]
            good = "return a + b" in messages[-3]["content"]
            lp = {" Yes": math.log(0.9 if good else 0.2), " No": math.log(0.1 if good else 0.8)}
            return Completion(" Yes", 120, 1, 0.01, first_top=lp)

    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path)},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    by_id = {t.task_id: t for t in tasks}
    assert annotate_ptrue(tmp_path, "toy", "t1", Judge(), by_id) > 0
    assert annotate_ptrue(tmp_path, "toy", "t1", Judge(), by_id) == 0  # resumable
    pool = load_pools(tmp_path, "toy", ["t1"])["toy/add"][0]
    passing = [c for c in pool.samples if c.passes_visible]
    assert passing and all(abs(c.extra["p_true"] - 0.9) < 1e-9 for c in passing)
    assert all(c.extra["p_true_tokens"] == 121 for c in passing)

    a = cand("t1", 0, visible=True, hidden=True, sig="A")
    b = cand("t1", 1, visible=True, hidden=True, sig="A")
    a.extra["p_true"], b.extra["p_true"] = 0.8, None
    assert phi_scores([a, b], "ptrue") == [0.8, 0.0]
    assert phi_scores([a, b], "agreement_ptrue") == [0.8, 0.0]
