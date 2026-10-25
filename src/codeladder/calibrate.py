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


def pareto_testing(
    losses: np.ndarray,
    costs: np.ndarray,
    alpha: float,
    delta: float,
    rng: np.random.Generator,
    opt_frac: float = 0.5,
) -> LTTResult:
    """Pareto Testing (Laufer-Goldshtein et al., 2023) for multi-dimensional threshold families.

    The calibration tasks are split in two. On the first part we find the configurations on
    the cost/risk Pareto front and order them from lowest to highest empirical risk. On the
    second part we run fixed-sequence binomial testing along that order. The ordering never
    sees the testing part, so the family-wise guarantee P(R(chosen) <= alpha) >= 1 - delta
    holds. Among the certified configurations the one with the lowest cost on the whole
    calibration set is chosen.
    """
    m, n = losses.shape
    idx = rng.permutation(n)
    n_opt = max(1, int(round(opt_frac * n)))
    opt, tst = idx[:n_opt], idx[n_opt:]
    r_opt = losses[:, opt].mean(axis=1)
    c_opt = costs[:, opt].mean(axis=1)
    # Pareto front on (risk, cost): keep configs not dominated by any other.
    order = np.lexsort((c_opt, r_opt))  # by risk, then cost
    front, best_cost = [], np.inf
    for j in order:
        if c_opt[j] < best_cost:
            front.append(int(j))
            best_cost = c_opt[j]
    pvals: list[float] = []
    valid: list[int] = []
    n_tst = len(tst)
    for j in front:  # lowest risk first
        p = binomial_pvalue(int(losses[j, tst].sum()), n_tst, alpha)
        pvals.append(p)
        if p <= delta:
            valid.append(j)
        else:
            break
    if not valid:
        return LTTResult(None, [], pvals)
    mean_costs = costs[valid].mean(axis=1)
    return LTTResult(valid[int(np.argmin(mean_costs))], valid, pvals)


def bonferroni_paths(
    losses: np.ndarray, costs: np.ndarray, alpha: float, delta: float, paths: list[list[int]]
) -> LTTResult:
    """Fixed-sequence testing along several pre-specified paths, Bonferroni-corrected.

    Each path is a list of configuration indices ordered from conservative to aggressive
    (for CodeLadder: a fixed escalation threshold, then final-tier thresholds from high to
    low). Every path is tested on all calibration tasks at level delta / len(paths), so the
    family-wise error over all certified configurations is at most delta. The cheapest
    certified configuration is chosen.
    """
    level = delta / max(len(paths), 1)
    valid: list[int] = []
    pvals: list[float] = []
    n = losses.shape[1]
    for path in paths:
        for j in path:
            p = binomial_pvalue(int(losses[j].sum()), n, alpha)
            pvals.append(p)
            if p <= level:
                valid.append(j)
            else:
                break
    if not valid:
        return LTTResult(None, [], pvals)
    mean_costs = costs[valid].mean(axis=1)
    return LTTResult(valid[int(np.argmin(mean_costs))], valid, pvals)
