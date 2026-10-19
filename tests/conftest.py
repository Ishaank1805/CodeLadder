"""Shared test helpers: a fake chat client and hand-built candidates."""

from __future__ import annotations

import pytest

from codeladder.llm import Completion
from codeladder.types import Candidate, TierPool


class FakeClient:
    """Returns the reference solution or a buggy one, decided by the request seed.

    ``bug_every`` = k makes every k-th seed buggy; repair requests (seed >= 10000)
    always get the fix, so repair chains are exercised.
    """

    def __init__(self, solutions: dict[str, tuple[str, str]], bug_every: int = 2):
        self.solutions = solutions  # entry_point -> (good_code, bad_code)
        self.bug_every = bug_every
        self.calls = 0

    def complete(self, messages, *, temperature, top_p, max_tokens, seed) -> Completion:
        self.calls += 1
        text = messages[1]["content"]
        entry = next(e for e in self.solutions if f"`{e}`" in text)
        good, bad = self.solutions[entry]
        is_repair = len(messages) > 2
        code = good if is_repair or seed % self.bug_every else bad
        return Completion(text=f"```python\n{code}```", tokens_in=50, tokens_out=40, latency_s=0.01)


def cand(
    tier: str,
    idx: int,
    *,
    visible: bool,
    hidden: bool,
    sig: str = "A",
    depth: int = 0,
    err: str | None = None,
    tokens: int = 10,
) -> Candidate:
    return Candidate(
        cand_id=f"t|{tier}|{idx}|{depth}",
        tier=tier,
        kind="sample" if depth == 0 else "repair",
        sample_idx=idx,
        depth=depth,
        code="",
        visible_pass_rate=1.0 if visible else 0.0,
        error_type=err or ("pass" if visible else "wrong_answer"),
        feedback=None,
        signature=sig,
        hidden_pass=hidden,
        tokens_out=tokens,
    )


def pool(
    tier: str, samples: list[Candidate], repairs: dict[int, list[Candidate]] | None = None
) -> TierPool:
    return TierPool(task_id="t", tier=tier, samples=samples, repairs=repairs or {})


@pytest.fixture
def fake_client_factory():
    return FakeClient
