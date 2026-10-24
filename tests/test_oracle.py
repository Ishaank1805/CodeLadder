"""X17 oracle sensitivity: per-test outcomes, weak labels, calibration on a weak oracle."""

import copy

from conftest import FakeClient
from test_pipeline import EXEC, GEN, PROBES, toy_tasks

from codeladder.evaluate import make_cost_fn
from codeladder.generate import generate_pools, load_pools
from codeladder.oracle import (
    compute_outcomes,
    hidden_failures,
    load_outcomes,
    run_oracle_shift,
    weak_labels,
)

# add() is right on small inputs but wrong once an argument exceeds 100: visible tests pass,
# some hidden tests fail.
SOLUTIONS = {
    "add": (
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b if a < 100 else 0\n",
    ),
    "remove_duplicates": (
        "def remove_duplicates(numbers):\n    return [n for n in numbers if numbers.count(n) == 1]\n",
        "def remove_duplicates(numbers):\n    return list(dict.fromkeys(numbers))\n",
    ),
    "is_prime": (
        "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n",
        "def is_prime(n):\n    return n % 2 == 1\n",
    ),
}


def test_hidden_failures_lists_every_failing_test():
    task = next(t for t in toy_tasks() if t.task_id == "toy/add")
    assert hidden_failures(task, SOLUTIONS["add"][0], EXEC) == []
    bad = hidden_failures(task, SOLUTIONS["add"][1], EXEC)
    assert bad == [
        i for i, args in enumerate(task.hidden_inputs) if args[0] >= 100 and sum(args) != 0
    ]


def test_weak_oracle_underreports_risk(tmp_path):
    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools")},
        "generation": {**GEN, "repair": {"parents": 0, "depth": 0}},
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 2, "action": "resample", "permutations": 2},
        "evaluation": {
            "alphas": [0.2], "delta": 0.2, "n_splits": 4, "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.5}, "abstain_factor": 4.0,
        },
    }  # fmt: skip
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    by_id = {t.task_id: t for t in tasks}
    stats = compute_outcomes(cfg["paths"]["pools"], tmp_path, "toy", "t1", by_id, EXEC, workers=2)
    assert stats["label_mismatch"] == 0 and stats["candidates"] > 0
    outcomes = load_outcomes(tmp_path, "toy", "t1")
    none = weak_labels(outcomes, 0.0)  # visible tests only: every candidate looks right
    assert all(none.values())
    full = weak_labels(outcomes, 1.0)
    assert not all(full.values())
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1"])
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(20)]
    out = run_oracle_shift(task_pools, ["t1"], {"toy/t1": outcomes}, cfg, make_cost_fn("tokens", {}),
                           fractions=(0.0, 1.0))  # fmt: skip
    weak = next(r for r in out["rows"] if r["fraction"] == 0.0)
    strong = next(r for r in out["rows"] if r["fraction"] == 1.0)
    assert weak["claimed_risk_weak"] == 0.0 and weak["true_risk"] > strong["true_risk"]
