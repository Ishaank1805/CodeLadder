"""API key pools: loading from the key file and rotating on rate limits."""

import time

import pytest

from codeladder import config, llm
from codeladder.config import TierConfig


def test_key_pool_from_file_and_env(tmp_path, monkeypatch):
    f = tmp_path / ".codeladder_keys"
    f.write_text("# comment\nGROQ_API_KEY_2='b'\nGROQ_API_KEY_1=a\nOTHER=x\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY_1", raising=False)
    tier = TierConfig(
        name="api", model="m", params_b=1, base_url="http://x", api_key_env="GROQ_API_KEY_*"
    )
    assert tier.api_keys() == ["a", "b"]
    monkeypatch.setenv("GROQ_API_KEY_1", "env")  # the environment wins
    assert tier.api_keys() == ["env", "b"]
    assert TierConfig(name="l", model="m", params_b=1).api_keys() == ["EMPTY"]
    with pytest.raises(RuntimeError):
        TierConfig(name="x", model="m", params_b=1, api_key_env="NOPE_*").api_keys()
    assert config._key_file(str(tmp_path / "missing")) == []


class _Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg, "logprobs": None})()]
        self.usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 2})()


class _RateLimited(Exception):
    status_code = 429

    def __init__(self, secs):
        super().__init__("429")
        self.response = type("R", (), {"headers": {"retry-after": str(secs)}})()


def test_rotates_keys_on_rate_limits(monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout, max_retries):
            assert max_retries == 2  # free APIs keep the SDK's retries
            self.key = api_key
            self.chat = type("Ch", (), {"completions": self})()

        def create(self, **kw):
            calls.append(self.key)
            if self.key == "a":
                raise _RateLimited(3600)  # key a is spent for an hour
            return _Resp("ok")

    import sys
    import types

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    tier = TierConfig(name="api", model="m", params_b=1, base_url="http://x", api_key_env="K_*")
    monkeypatch.setenv("K_1", "a")
    monkeypatch.setenv("K_2", "b")
    client = llm.OpenAIChatClient(tier)
    out = [
        client.complete(
            [{"role": "user", "content": str(i)}], temperature=0, top_p=1, max_tokens=5, seed=i
        ).text
        for i in range(3)
    ]
    assert out == ["ok"] * 3 and calls.count("a") == 1  # a is tried once, then rests
    client._rest_until = [time.time() + 7 * 3600] * 2
    with pytest.raises(llm.QuotaExhausted):
        client.complete(
            [{"role": "user", "content": "z"}], temperature=0, top_p=1, max_tokens=5, seed=9
        )


def test_server_json_failure_retries_without_logprobs(monkeypatch):
    import sys
    import types

    seen = []

    class ServerError(Exception):
        status_code = 500

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout, max_retries):
            assert max_retries == 0  # local server: a 500 reaches the fallback at once
            self.chat = type("Ch", (), {"completions": self})()

        def create(self, **kw):
            seen.append(kw.get("logprobs"))
            if kw.get("logprobs"):
                raise ServerError("Out of range float values are not JSON compliant")
            return _Resp("ok")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    tier = TierConfig(name="t", model="m", params_b=1, base_url="http://x")
    out = llm.OpenAIChatClient(tier).complete(
        [{"role": "user", "content": "x"}], temperature=0, top_p=1, max_tokens=5, seed=1
    )
    assert out.text == "ok" and seen == [True, False] and out.mean_logprob is None
    # Known-bad tasks can skip the failing first attempt.
    seen.clear()
    monkeypatch.setenv("CODELADDER_NO_LOGPROBS", "1")
    out = llm.OpenAIChatClient(tier).complete(
        [{"role": "user", "content": "y"}], temperature=0, top_p=1, max_tokens=5, seed=2
    )
    assert out.text == "ok" and seen == [False]


def test_continue_final_sends_vllm_prefill_flags(monkeypatch):
    import sys
    import types

    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout, max_retries):
            self.chat = type("Ch", (), {"completions": self})()

        def create(self, **kw):
            seen.update(kw)
            return _Resp("88")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    tier = TierConfig(name="d1", model="m", params_b=1, base_url="http://x")
    client = llm.OpenAIChatClient(tier)
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Confidence: "}]
    client.complete(msgs, temperature=0, top_p=1, max_tokens=5, seed=0, continue_final=True)
    assert seen["extra_body"] == {"continue_final_message": True, "add_generation_prompt": False}
    client.complete(msgs, temperature=0, top_p=1, max_tokens=5, seed=0)
    assert seen["extra_body"] is None
