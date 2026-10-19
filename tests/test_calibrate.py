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
