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


def test_confidence_annotation(tmp_path):
    from codeladder.generate import annotate_confidence
    from codeladder.llm import Completion
    from codeladder.prompts import parse_confidence

    assert parse_confidence("85") == 0.85 and parse_confidence("I'd say 100%") == 1.0
    assert parse_confidence("no idea") is None

    class ConfClient:
        def complete(self, messages, **kw):
            return Completion(text="73", tokens_in=100, tokens_out=2, latency_s=0.01)

    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path)},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    n = annotate_confidence(tmp_path, "toy", "t1", ConfClient(), {t.task_id: t for t in tasks})
    assert n > 0
    pool = load_pools(tmp_path, "toy", ["t1"])["toy/add"][0]
    passing = [c for c in pool.samples if c.passes_visible]
    assert passing and all(c.extra["self_conf"] == 0.73 for c in passing)
    assert all("self_conf" not in c.extra for c in pool.samples if not c.passes_visible)
    assert (
        annotate_confidence(tmp_path, "toy", "t1", ConfClient(), {t.task_id: t for t in tasks}) == 0
    )


def test_two_level_family_runs_end_to_end(tmp_path):
    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 4, "action": "resample", "permutations": 2},
        "evaluation": {
            "alphas": [0.2],
            "delta": 0.2,
            "n_splits": 4,
            "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
            "family": "two_level",
            "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.5},
            "cost_metric": "weighted_tokens",
        },
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    generate_pools(tasks, "t2", FakeClient(SOLUTIONS, bug_every=5), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1", "t2"])
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(10)]
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.5, "t2": 7})
    summary = run_experiment(task_pools, ["t1", "t2"], cfg, cost_fn, tmp_path / "results")
    assert summary["family"] == "two_level"
    assert len(summary["thresholds"]) == 4 * 4  # (1.01, 1.0, 0.5, 0.0) squared
    assert summary["calibrated"][0]["n_splits"] == 4


def test_subsets_family_includes_single_tiers_and_respects_abstain_price(tmp_path):
    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 4, "action": "resample", "permutations": 2},
        "evaluation": {
            "alphas": [0.2],
            "delta": 0.2,
            "n_splits": 4,
            "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
            "family": "subsets",
            "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.5},
            "escalation_thresholds": [1.01, 0.5],
            "cost_metric": "weighted_tokens",
        },
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    generate_pools(tasks, "t2", FakeClient(SOLUTIONS, bug_every=5), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1", "t2"])
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(10)]
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.5, "t2": 7})
    out = run_experiment(task_pools, ["t1", "t2"], cfg, cost_fn, tmp_path / "plain")
    # t1 alone, t2 alone (one path each) and t1+t2 with escalation 0.5 (1.01 is dropped)
    assert {tuple(c[:2]) for c in out["thresholds"]} == {("t1", None), ("t2", None), ("t1+t2", 0.5)}
    assert len(out["thresholds"]) == 3 * 4
    assert out["calibrated"][0]["violation_rate"] <= 0.2 + 1e-9
    # With plain cost, cheap abstention can win; pricing abstention never lowers coverage here.
    cfg["evaluation"]["abstain_factor"] = 2.0
    priced = run_experiment(task_pools, ["t1", "t2"], cfg, cost_fn, tmp_path / "priced")
    assert priced["calibrated"][0]["mean_coverage"] >= out["calibrated"][0]["mean_coverage"]


def test_conditional_risk_and_group_calibration(tmp_path):
    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 4, "action": "resample", "permutations": 2},
        "evaluation": {
            "alphas": [0.3],
            "delta": 0.2,
            "n_splits": 4,
            "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
            "cost_metric": "weighted_tokens",
        },
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1"])
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(10)]
    groups = ["easy" if i % 2 else "hard" for i in range(len(task_pools))]
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.5})
    out = run_experiment(task_pools, ["t1"], cfg, cost_fn, tmp_path / "r", groups=groups)
    cr = out["conditional_risk"][0]
    assert (
        set(cr["groups"]) == {"easy", "hard"}
        and cr["groups"]["easy"]["n_tasks"] == len(task_pools) // 2
    )
    assert out["calibrated_grouped"][0]["n_splits"] == 4


def test_parse_confidence_reads_sentences():
    from codeladder.prompts import parse_confidence

    assert parse_confidence("95\n") == 0.95  # bare numbers unchanged
    assert parse_confidence("As an expert Python 3 programmer, I am 90% confident.") == 0.9
    assert parse_confidence("My confidence that this is correct is 85 out of 100") == 0.85
    assert parse_confidence("I would rate it 70/100") == 0.7
    assert parse_confidence("It handles Python 3 lists; rating: 60") == 0.6
    assert parse_confidence("no idea") is None


def test_annotation_rerun_over_unparsable_replies_returns_zero(tmp_path):
    from codeladder.generate import annotate_confidence
    from codeladder.llm import Completion

    tasks = toy_tasks()
    cfg = {
        "paths": {"pools": str(tmp_path / "pools")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)

    class Mumbler:
        calls = 0

        def complete(self, messages, **kw):
            Mumbler.calls += 1
            return Completion(text="hard to say", tokens_in=5, tokens_out=3, latency_s=0.0)

    by_id = {t.task_id: t for t in tasks}
    assert annotate_confidence(cfg["paths"]["pools"], "toy", "t1", Mumbler(), by_id, workers=1) == 0
    first = Mumbler.calls
    assert first > 0  # every visible-passing candidate was asked
    assert annotate_confidence(cfg["paths"]["pools"], "toy", "t1", Mumbler(), by_id, workers=1) == 0


def test_prefilled_confidence_redo_keeps_old_rating(tmp_path):
    from codeladder.generate import annotate_confidence
    from codeladder.llm import Completion
    from codeladder.prompts import CONF_PREFILL, parse_confidence

    assert parse_confidence("85\n\nThe solution", prefilled=True) == 0.85
    assert parse_confidence(" 7", prefilled=True) == 0.07
    assert parse_confidence("I'm sorry, 3-character", prefilled=True) is None

    seen = []

    class Declines:  # no number: the pre-prefill DeepSeek reply
        def complete(self, messages, **kw):
            return Completion("As an AI, I cannot run a 3-character test.", 9, 9, 0.01)

    class Prefilled:
        def complete(self, messages, **kw):
            seen.append((messages[-1], kw.get("continue_final")))
            return Completion("90\nIt handles", 9, 2, 0.01)

    tasks = toy_tasks()
    by_id = {t.task_id: t for t in tasks}
    cfg = {
        "paths": {"pools": str(tmp_path)},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
    }
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    annotate_confidence(tmp_path, "toy", "t1", Declines(), by_id)
    pool = load_pools(tmp_path, "toy", ["t1"])["toy/add"][0]
    assert {c.extra["self_conf"] for c in pool.samples if c.passes_visible} == {0.03}  # junk
    n = annotate_confidence(tmp_path, "toy", "t1", Prefilled(), by_id, prefill=True, redo=True)
    assert n > 0 and seen[0] == ({"role": "assistant", "content": CONF_PREFILL}, True)
    pool = load_pools(tmp_path, "toy", ["t1"])["toy/add"][0]
    passing = [c for c in pool.samples if c.passes_visible]
    assert all(c.extra["self_conf"] == 0.9 and c.extra["self_conf_v1"] == 0.03 for c in passing)
    assert (
        annotate_confidence(tmp_path, "toy", "t1", Prefilled(), by_id, prefill=True, redo=True) == 0
    )


def test_cli_evaluate_restricts_to_task_ids(tmp_path):
    import json

    import yaml

    from codeladder.cli import main
    from codeladder.config import REPO_ROOT

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "experiment.yaml").read_text())
    cfg["paths"] = {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")}
    cfg["evaluation"].update(n_splits=3, alphas=[0.2], calib_sizes=[])
    exp = tmp_path / "experiment.yaml"
    exp.write_text(yaml.safe_dump(cfg))
    gen = {"paths": cfg["paths"], "generation": GEN, "probes": PROBES, "execution": EXEC}
    generate_pools(toy_tasks(), "t1", FakeClient(SOLUTIONS, bug_every=2), gen, workers=2)
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["toy/add", "toy/is_prime"]))
    args = ["--experiment", str(exp), "evaluate", "--benchmark", "toy", "--tiers", "t1"]
    assert main(args + ["--task-ids", str(ids), "--tag=-sub"]) == 0
    runs = tmp_path / "results" / "runs" / "toy"  # evaluate runs are grouped by benchmark set
    sub = json.loads(next(runs.glob("*-sub/summary.json")).read_text())
    assert sub["n_tasks"] == 2
    assert main(args) == 0
    full = [p for p in runs.glob("*/summary.json") if "-sub" not in str(p)]
    assert json.loads(full[0].read_text())["n_tasks"] == 3
