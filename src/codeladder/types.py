"""Core data types shared across the harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """A function-level coding task.

    The harness only ever sees ``prompt``, ``entry_point`` and the visible tests.
    ``reference`` and the hidden tests are used for labels and evaluation only.
    """

    task_id: str
    benchmark: str
    prompt: str
    entry_point: str
    reference: str
    visible_inputs: list[list[Any]]
    hidden_inputs: list[list[Any]]
    atol: float = 0.0
    release_date: str | None = None
    # Filled by running the reference solution (see tasks.attach_expected_outputs).
    visible_expected: list[Any] | None = None
    hidden_expected: list[Any] | None = None
    ref_seconds: float = 0.0  # reference runtime on all hidden inputs (sets time budgets)
    visible_ref_times: list[float] | None = None  # per-input reference runtimes (seconds)
    hidden_ref_times: list[float] | None = None
    oracle: str | None = None  # special checker name (see oracles.py), None = equality
    mode: str = (
        "function"  # function | method (LeetCode) | stdin (whole program) | bcb (unit tests)
    )
    time_limit_s: float | None = None  # per-test limit when there is no reference runtime
    difficulty: str | None = None
    source_path: str | None = None  # where lazily-loaded tests live (LiveCodeBench)
    tests: str | None = None  # unit-test source (BigCodeBench: first method visible, all hidden)
    probe_code: str | None = None  # model-written probe calls for agreement (BigCodeBench)


@dataclass
class Candidate:
    """One generated program and everything we measured about it.

    ``hidden_pass`` is the ground-truth label. The controller must never read it;
    only the evaluator does.
    """

    cand_id: str
    tier: str
    kind: str  # "sample" or "repair"
    sample_idx: int  # index of the root sample this candidate descends from
    depth: int  # 0 for samples, 1..d for repair rounds
    code: str
    visible_pass_rate: float
    error_type: str  # "pass", "syntax_error", "timeout", "wrong_answer", "exception:<Name>", ...
    feedback: dict[str, Any] | None  # first visible failure, used to build repair prompts
    signature: str  # behaviour on probe inputs, used for agreement clustering
    hidden_pass: bool
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    mean_logprob: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_visible(self) -> bool:
        return self.visible_pass_rate >= 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Candidate:
        return Candidate(**d)


@dataclass
class TierPool:
    """All cached candidates for one (task, tier): samples plus repair chains."""

    task_id: str
    tier: str
    samples: list[Candidate]
    repairs: dict[int, list[Candidate]]  # sample_idx -> chain of repairs (depth 1, 2, ...)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "samples": [c.to_dict() for c in self.samples],
            "repairs": {str(k): [c.to_dict() for c in v] for k, v in self.repairs.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TierPool:
        return TierPool(
            task_id=d["task_id"],
            tier=d["tier"],
            samples=[Candidate.from_dict(c) for c in d["samples"]],
            repairs={int(k): [Candidate.from_dict(c) for c in v] for k, v in d["repairs"].items()},
        )
