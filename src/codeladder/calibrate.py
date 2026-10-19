"""Learn-then-Test calibration (Angelopoulos et al., 2021).

Configurations are ordered from most conservative to most aggressive. For each one
we test H0: R(config) > alpha with an exact binomial p-value on 0/1 losses, and use
fixed-sequence testing (stop at the first non-rejection), which controls the
family-wise error rate at delta. Among the configurations whose null is rejected we
pick the cheapest. Guarantee: P(R(chosen) <= alpha) >= 1 - delta, assuming
calibration and test tasks are exchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


def binomial_pvalue(n_errors: int, n: int, alpha: float) -> float:
    """p-value for H0: risk >= alpha, given n_errors 0/1 losses out of n."""
    return float(binom.cdf(n_errors, n, alpha))


@dataclass
class LTTResult:
    chosen: int | None  # index into the config list, or None if nothing is certified
    valid: list[int]  # all configs whose null was rejected
    pvalues: list[float]


def learn_then_test(losses: np.ndarray, costs: np.ndarray, alpha: float, delta: float) -> LTTResult:
    """
    losses: (m configs, n tasks) array of 0/1 losses, rows ordered conservative -> aggressive.
    costs:  (m configs, n tasks) array of costs.
    """
    m, n = losses.shape
    pvals: list[float] = []
    valid: list[int] = []
    for j in range(m):
        p = binomial_pvalue(int(losses[j].sum()), n, alpha)
        pvals.append(p)
        if p <= delta:
            valid.append(j)
        else:
            break  # fixed-sequence testing
    if not valid:
        return LTTResult(None, [], pvals)
    mean_costs = costs[valid].mean(axis=1)
    return LTTResult(valid[int(np.argmin(mean_costs))], valid, pvals)
