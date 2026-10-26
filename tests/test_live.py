"""X15: export a calibrated policy and run it online (fake model) or from cached pools."""

import copy
import json

from conftest import FakeClient
from test_pipeline import EXEC, GEN, PROBES, SOLUTIONS, toy_tasks

from codeladder.controller import run_task
from codeladder.evaluate import make_cost_fn, run_experiment
from codeladder.generate import generate_pools, load_pools
from codeladder.live import export_policy, load_policy, replay, solve


def _cfg(tmp_path):
    return {
        "paths": {"pools": str(tmp_path / "pools"), "results": str(tmp_path / "results")},
        "generation": GEN,
        "probes": PROBES,
        "execution": EXEC,
        "controller": {"n_init": 2, "max_draws": 4, "action": "resample", "permutations": 2},
        "evaluation": {
            "alphas": [0.3],
            "delta": 0.2,
            "n_splits": 3,
            "calib_frac": 0.5,
            "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
            "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.5},
            "escalation_thresholds": [0.9],
            "family": "subsets",
            "cost_metric": "weighted_tokens",
        },
    }


def test_export_then_solve_live_and_replay(tmp_path):
    cfg = _cfg(tmp_path)
    tasks = toy_tasks()
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    generate_pools(tasks, "t2", FakeClient(SOLUTIONS, bug_every=5), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1", "t2"])
    task_pools = [copy.deepcopy(p) for p in pools.values() for _ in range(10)]
    run = tmp_path / "results" / "run"
    run_experiment(task_pools, ["t1", "t2"], cfg, make_cost_fn("weighted_tokens", {}), run)
    pol = export_policy(run, cfg, alpha=0.3)
    assert pol["calibration_tasks"] == 30 and pol["calibration"]["risk"] <= 0.3
    assert len(pol["thresholds"]) == 2 and "guarantee" in pol
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(pol))
    policy, meta = load_policy(path)

    task = next(t for t in tasks if t.task_id == "toy/add")
    clients = {t: FakeClient(SOLUTIONS, bug_every=2 if t == "t1" else 5) for t in meta["tiers"]}
    lines = []
    out = solve(task, policy, meta["tiers"], clients, cfg, log=lines.append)
    assert out["trace"][-1]["step"] in ("accept", "abstain") and lines
    assert out["accepted"] is None or out["accepted"].hidden_pass

    # Two tiers, two draws each: the fake model's first sample is buggy and its second right,
    # so the right one has agreement 0.5: below 0.9 at the first tier (escalate), enough at
    # the second (accept).
    policy.thresholds, policy.max_draws, policy.n_init = [0.9, 0.5], 2, 2
    out = solve(task, policy, ["t1", "t2"], clients, cfg, log=lines.append)
    steps = [s["step"] for s in out["trace"]]
    assert steps == ["escalate", "accept"] and out["tier"] == "t2"
    assert out["accepted"].hidden_pass

    rep = replay(pools["toy/add"], policy, 0)
    ref = run_task(pools["toy/add"], policy, 0)
    assert (rep["accepted"] is None) == (ref.accepted is None)


def test_custom_problem_runs_and_abstains_on_wrong_code(tmp_path):
    from codeladder.config import REPO_ROOT, load_experiment
    from codeladder.live import custom_task
    from codeladder.llm import Completion

    task = custom_task(REPO_ROOT / "examples" / "problem.json")
    assert task.entry_point == "second_largest" and len(task.visible_inputs) == 3
    good = "def second_largest(numbers):\n    s = sorted(set(numbers))\n    return s[-2] if len(s) > 1 else None\n"
    bad = "def second_largest(numbers):\n    return sorted(numbers)[-2]\n"

    class Model:
        def __init__(self, code):
            self.code = code

        def complete(self, messages, **kw):
            if kw.get("temperature") == 0.0:  # rating request
                return Completion("95", 10, 1, 0.0)
            return Completion(f"```python\n{self.code}```", 100, 40, 0.0)

    cfg = load_experiment()
    policy, meta = load_policy(REPO_ROOT / "examples" / "policies" / "evalplus-ladder-a0.2.json")
    out = solve(
        task,
        policy,
        meta["tiers"],
        {t: Model(good) for t in meta["tiers"]},
        cfg,
        log=lambda m: None,
    )
    assert out["tier"] == meta["tiers"][0] and out["accepted"].visible_pass_rate == 1.0
    out = solve(
        task, policy, meta["tiers"], {t: Model(bad) for t in meta["tiers"]}, cfg, log=lambda m: None
    )
    assert out["accepted"] is None and out["trace"][-1]["step"] == "abstain"
