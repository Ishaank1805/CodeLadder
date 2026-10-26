import numpy as np

from codeladder.calibrate import binomial_pvalue, learn_then_test


def test_pvalue_monotone():
    assert (
        binomial_pvalue(0, 200, 0.05)
        < binomial_pvalue(5, 200, 0.05)
        < binomial_pvalue(20, 200, 0.05)
    )


def test_ltt_controls_risk_on_synthetic_data():
    """Configs with true risks 0 .. 0.3 and costs decreasing with risk.

    The chosen config's true risk must exceed alpha in at most ~delta of trials.
    """
    rng = np.random.default_rng(0)
    true_risk = np.linspace(0.0, 0.3, 16)  # conservative -> aggressive
    alpha, delta, n = 0.1, 0.1, 300
    violations, chosen_any = 0, 0
    trials = 400
    for _ in range(trials):
        losses = (rng.random((len(true_risk), n)) < true_risk[:, None]).astype(float)
        costs = np.tile(np.linspace(10, 1, len(true_risk))[:, None], (1, n))
        res = learn_then_test(losses, costs, alpha, delta)
        if res.chosen is not None:
            chosen_any += 1
            violations += true_risk[res.chosen] > alpha
    assert chosen_any > trials * 0.9
    assert violations / trials <= delta + 0.03


def test_ltt_returns_none_when_nothing_is_safe():
    losses = np.ones((3, 100))
    res = learn_then_test(losses, np.ones((3, 100)), 0.05, 0.1)
    assert res.chosen is None


def test_shift_detects_drift_and_sliding_window_adapts():
    from codeladder.evaluate import Replay, run_shift

    rng = np.random.default_rng(1)
    n, m = 600, 11
    dates = [f"2025-{1 + i * 12 // n:02d}-{1 + i % 28:02d}" for i in range(n)]
    # Error rate of each threshold grows over time: later tasks are harder.
    drift = np.linspace(0.0, 0.25, n)
    base = np.linspace(0.0, 0.2, m)[:, None]
    loss = (rng.random((m, n, 1)) < (base + drift)[:, :, None] * 0.6).astype(float)
    rep = Replay(loss=loss, accepted=np.ones_like(loss), correct=1 - loss,
                 cost=np.tile(np.linspace(10, 1, m)[:, None, None], (1, n, 1)),
                 fallbacks=np.zeros_like(loss))  # fmt: skip
    out = run_shift(rep, dates, alpha=0.1, delta=0.1, n_windows=5)
    assert len(out["windows"]) == 5
    late = out["windows"][-1]
    assert late["fixed"]["risk"] >= late["sliding"]["risk"]


def test_pareto_testing_controls_risk_on_2d_family():
    from codeladder.calibrate import pareto_testing

    rng = np.random.default_rng(3)
    a, b = np.meshgrid(np.linspace(0, 0.25, 8), np.linspace(0, 0.25, 8))
    true_risk = (0.5 * a + 0.5 * b).ravel()  # 64 configs, 2-D family
    cost = (20 - 40 * a - 30 * b).ravel()
    alpha, delta, n, trials = 0.1, 0.1, 400, 300
    violations = chosen = 0
    for _ in range(trials):
        losses = (rng.random((len(true_risk), n)) < true_risk[:, None]).astype(float)
        costs = np.tile(cost[:, None], (1, n))
        res = pareto_testing(losses, costs, alpha, delta, rng)
        if res.chosen is not None:
            chosen += 1
            violations += true_risk[res.chosen] > alpha
    assert chosen > trials * 0.9
    assert violations / trials <= delta + 0.03


def test_bonferroni_paths_controls_risk():
    from codeladder.calibrate import bonferroni_paths

    rng = np.random.default_rng(5)
    lf = np.linspace(0, 0.25, 10)
    offsets = [0.0, 0.03, 0.06]
    true_risk = np.array([x + o for o in offsets for x in lf])
    paths = [list(range(k * 10, (k + 1) * 10)) for k in range(len(offsets))]
    cost = np.array([20 - 30 * x - 10 * o for o in offsets for x in lf])
    violations = chosen = 0
    for _ in range(300):
        losses = (rng.random((len(true_risk), 400)) < true_risk[:, None]).astype(float)
        res = bonferroni_paths(losses, np.tile(cost[:, None], (1, 400)), 0.1, 0.1, paths)
        if res.chosen is not None:
            chosen += 1
            violations += true_risk[res.chosen] > 0.1
    assert chosen > 270 and violations / 300 <= 0.13


def test_grouped_calibration_controls_each_group():
    from codeladder.evaluate import Replay, run_splits_grouped

    rng = np.random.default_rng(7)
    m, n = 12, 600
    groups = ["easy"] * 300 + ["hard"] * 300
    base = np.linspace(0.0, 0.3, m)[:, None]
    shift = np.array([0.0] * 300 + [0.08] * 300)[None, :]
    loss = (rng.random((m, n, 1)) < (base + shift)[:, :, None]).astype(float)
    rep = Replay(loss=loss, accepted=np.ones_like(loss), correct=1 - loss,
                 cost=np.tile(np.linspace(10, 1, m)[:, None, None], (1, n, 1)),
                 fallbacks=np.zeros_like(loss))  # fmt: skip
    out = run_splits_grouped(rep, groups, alpha=0.15, delta=0.1, n_splits=40, calib_frac=0.5)
    assert out["violation_rate"] <= 0.1
    picks = out["splits"][0]["chosen"]
    assert picks["easy"] >= picks["hard"]  # the easier group can use a more aggressive setting


def test_learning_curve_uses_fewer_calibration_tasks_with_the_same_test_half():
    import numpy as np

    from codeladder.evaluate import Replay, run_splits

    rng = np.random.default_rng(0)
    m, n = 5, 400
    loss = (rng.random((m, n, 1)) < np.linspace(0.0, 0.3, m)[:, None, None]).astype(float)
    acc = np.ones_like(loss)
    rep = Replay(loss, acc, acc - loss, np.linspace(5, 1, m)[:, None, None] * acc, 0 * acc)
    full = run_splits(rep, 0.1, 0.1, 20, 0.5)
    small = run_splits(rep, 0.1, 0.1, 20, 0.5, n_cal=40)
    # fewer calibration tasks certify less aggressive (costlier) settings, never more risk
    assert small["mean_test_cost"] >= full["mean_test_cost"]
    assert small["violation_rate"] <= 0.1 + 1e-9


def test_transfer_detects_a_riskier_target_distribution():
    import numpy as np

    from codeladder.evaluate import Replay, run_transfer

    rng = np.random.default_rng(0)
    m, n = 4, 600
    risk_src = np.linspace(0.0, 0.15, m)[:, None, None]
    loss = np.zeros((m, n, 1))
    loss[:, :300] = rng.random((m, 300, 1)) < risk_src  # source tasks
    loss[:, 300:] = rng.random((m, 300, 1)) < 2 * risk_src  # target: twice as risky
    acc = np.ones_like(loss)
    rep = Replay(loss, acc, acc - loss, np.linspace(4, 1, m)[:, None, None] * acc, 0 * acc)
    out = run_transfer(rep, np.arange(300), np.arange(300, 600), 0.1, 0.1, 20, 0.5)
    assert out["certified_splits"] > 0 and out["target_risk"] > 0.1  # does not transfer
