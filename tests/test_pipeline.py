"""End-to-end on the built-in toy tasks with a fake model: generate -> replay -> calibrate."""

import copy

from conftest import FakeClient

from codeladder.evaluate import make_cost_fn, run_experiment
from codeladder.generate import build_pool, generate_pools, load_pools
from codeladder.prompts import extract_code
from codeladder.tasks import attach_expected_outputs, load_toy

SOLUTIONS = {
    "add": ("def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a - b\n"),
    "remove_duplicates": (
        "def remove_duplicates(numbers):\n    return [n for n in numbers if numbers.count(n) == 1]\n",
        "def remove_duplicates(numbers):\n    return list(dict.fromkeys(numbers))\n",
    ),
    "is_prime": (
        "def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n",
        "def is_prime(n):\n    return n % 2 == 1\n",
    ),
}

GEN = {
    "n_samples": 6,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 256,
    "seed": 0,
    "repair": {"parents": 6, "depth": 2},
}
PROBES = {"n": 8, "seed": 0}
EXEC = {"per_call_timeout_s": 2.0, "memory_mb": 1024, "max_workers": 2}


def toy_tasks():
    return [attach_expected_outputs(t) for t in load_toy()]


def test_reference_outputs_attached():
    tasks = toy_tasks()
    assert all(t.visible_expected and t.hidden_expected for t in tasks)


def test_build_pool_labels_and_repairs():
    task = next(t for t in toy_tasks() if t.entry_point == "add")
    p = build_pool(task, "t1", FakeClient(SOLUTIONS, bug_every=2), GEN, PROBES, EXEC)
    good = [c for c in p.samples if c.hidden_pass]
    bad = [c for c in p.samples if not c.passes_visible]
    assert good and bad
    assert all(not c.hidden_pass for c in bad)
    assert p.repairs and all(chain[-1].passes_visible for chain in p.repairs.values())
    assert len({c.signature for c in good}) == 1  # correct programs agree on probes


def test_remove_duplicates_bug_is_caught_by_visible_test():
    task = next(t for t in toy_tasks() if t.entry_point == "remove_duplicates")
    p = build_pool(task, "t1", FakeClient(SOLUTIONS, bug_every=2), GEN, PROBES, EXEC)
    for c in p.samples:
        assert c.passes_visible == c.hidden_pass


def test_extract_code_handles_fences_and_bodies():
    task = next(t for t in load_toy() if t.entry_point == "add")
    assert "return a + b" in extract_code(
        "Sure!\n```python\ndef add(a, b):\n    return a + b\n```", task
    )
    grafted = extract_code("return a + b", task)
    assert grafted.startswith("def add") and "return a + b" in grafted


def test_end_to_end_experiment(tmp_path):
    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 4, "action": "adaptive", "permutations": 2},
        "evaluation": {
            "alphas": [0.2],
            "delta": 0.2,
            "n_splits": 5,
            "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
            "cost_metric": "weighted_tokens",
        },
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    generate_pools(tasks, "t2", FakeClient(SOLUTIONS, bug_every=5), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1", "t2"])
    assert len(pools) == 3
    # Pad to more tasks so calibration splits are non-trivial.
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(10)]
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.5, "t2": 7})
    summary = run_experiment(task_pools, ["t1", "t2"], cfg, cost_fn, tmp_path / "results")
    assert (tmp_path / "results" / "summary.json").exists()
    assert summary["baselines"]["oracle"]["solved"] == 1.0
    assert summary["calibrated"][0]["n_splits"] == 5
