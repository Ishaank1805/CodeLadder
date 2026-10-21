"""Model Cascading for Code and Bayesian self-escalation replays."""

import numpy as np

from codeladder.baselines import bse_eval, fit_logistic, mcc_tier_outcome, run_asserts, split_eval


def test_run_asserts_per_line_with_isolation_and_timeouts():
    code = "def add(a, b):\n    return a + b\n"
    lines = [
        "assert add(1, 2) == 3",
        "assert add(1, 2) == 4",
        "add = None",
        "assert add(2, 2) == 4",
        "while True: pass",
    ]
    assert run_asserts(code, lines, timeout=1.0) == [True, False, True, True, False]
    assert run_asserts("def add(:\n", ["assert add(1, 2) == 3"]) == [False]


def test_mcc_scores_agreeing_solutions_higher():
    # 3 solutions, 2 tests x 2 lines; solutions 0 and 1 agree on every line
    passed = [[True, True, True, True], [True, True, True, True], [False, True, False, False]]
    best, score = mcc_tier_outcome(passed, [0, 0, 1, 1], [0, 1, 2], k=2)
    assert best == 0 and score == 1.0  # n_s = 4 lines, n_t = 2 solutions, k x N_t = 2 x 4
    best, score = mcc_tier_outcome(passed, [0, 0, 1, 1], [2, 0, 1], k=1)
    assert best == 2 and abs(score - 0.5) < 1e-9  # first test only: 1 of 2 lines


def test_logistic_fit_separates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(float)
    w = fit_logistic(X, y, l2=0.1)
    p = 1 / (1 + np.exp(-(np.hstack([X, np.ones((400, 1))]) @ w)))
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.95


def test_bse_escalates_doomed_generations_and_split_eval_runs():
    rng = np.random.default_rng(1)
    traces = {}
    for i in range(80):
        good = i % 2 == 0
        stats = [
            [-0.05 if good else -1.5, 0.1 if good else 1.2, 5.0 if good else 0.3] for _ in range(30)
        ]
        stats = (np.array(stats) + rng.normal(0, 0.05, (30, 3))).tolist()
        junior = {"token_stats": stats, "hidden_pass": good, "tokens_in": 100, "tokens_out": 30}
        senior = {"token_stats": stats, "hidden_pass": True, "tokens_in": 100, "tokens_out": 30}
        traces[f"t{i}"] = [junior, senior]
    res = bse_eval(sorted(traces), traces, [1.0, 10.0], [0.05], n_splits=5)
    assert res[0]["tuned_splits"] == 5 and res[0]["mean_test_risk"] <= 0.05
    assert res[0]["mean_solved"] > 0.9 and res[0]["mean_test_cost"] < 30 * 10 + 100
    loss = np.array([[0.3] * 10, [0.0] * 10])
    out = split_eval(loss, np.array([[1.0] * 10, [5.0] * 10]), 1 - loss, [0.1], n_splits=3)
    assert out[0]["mean_test_cost"] == 5.0
