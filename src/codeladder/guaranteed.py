"""X16: guaranteed baselines on the same splits as CodeLadder.

Each tier is replayed alone for every threshold (its own draws, scores and checks, exactly as
the controller does at that tier). A cascade over the tiers is then a composition: a task is
answered by the first tier that accepts it, costs the draws of every tier it visited, and is
abstained on if none accepts. On top of these replays:

* ``single``: per-model Learn-then-Test abstention (Task Abstention for LLMs in Code
  Generation, 2026): one model, fixed-sequence testing of its threshold on the marginal risk
  "accepted and wrong", the cheapest certified setting.
* ``toe``: a Trust-or-Escalate cascade (Jung et al., ICLR 2025) adapted to code: rung by rung,
  on the calibration tasks that reach the rung, the most permissive threshold whose error
  among the tasks it accepts is certified at most α (exact binomial, fixed sequence, δ / rungs);
  tasks it does not accept go to the next rung; after the last rung the cascade abstains. This
  controls the error among accepted tasks at every rung, so also the marginal risk.
* ``toe-lean``: the same with one draw at every rung but the last (CodeLadder's lean ladders).

The split draws (calibration half, sample order per task) follow ``evaluate.run_splits`` with
the same seed, so results compare one to one with CodeLadder's run folders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binom

from .controller import Policy
from .evaluate import Replay, replay, threshold_grid
from .types import TierPool


def tier_replays(
    task_pools: list[list[TierPool]], ctl: dict[str, Any], ev: dict[str, Any], cost_fn,
    budget: int | None = None,
) -> tuple[list[float], list[Replay]]:  # fmt: skip
    """Per tier: replay of that tier alone for every threshold (rows), with ``budget`` draws
    if given (lean lower rungs) or the usual n_init / max_draws."""
    grid = threshold_grid(**ev["lambda_grid"])
    n_init = min(ctl["n_init"], budget) if budget else ctl["n_init"]
    draws = budget or ctl["max_draws"]
    perms = int(ctl.get("permutations", 1))
    score = ctl.get("score", "agreement")
    out = []
    for t in range(len(task_pools[0])):
        pols = [Policy([lam], n_init, draws, ctl["action"], {}, score=score) for lam in grid]
        out.append(replay([[ps[t]] for ps in task_pools], pols, perms, cost_fn))
    return grid, out


def compose(reps: list[Replay], ks: list[int | None]) -> dict[str, np.ndarray]:
    """Cascade outcome per (task, perm) for threshold indices ``ks`` (None skips a tier)."""
    shape = reps[0].loss.shape[1:]
    open_ = np.ones(shape, bool)
    out = {k: np.zeros(shape) for k in ("loss", "accepted", "correct", "cost")}
    for rep, k in zip(reps, ks, strict=True):
        if k is None:
            continue
        out["cost"] += np.where(open_, rep.cost[k], 0.0)
        acc = open_ & (rep.accepted[k] > 0)
        for key in ("loss", "accepted", "correct"):
            out[key] += np.where(acc, getattr(rep, key)[k], 0.0)
        open_ &= ~acc
    return out


def _fixed_sequence_conditional(
    acc: np.ndarray, wrong: np.ndarray, alpha: float, level: float
) -> int:
    """Index of the most permissive threshold (rows ordered conservative to aggressive) whose
    error among accepted tasks is certified at most alpha; 0 (never accept) if none."""
    chosen = 0
    for k in range(acc.shape[0]):
        a = int(acc[k].sum())
        if a == 0:
            chosen = k  # accepts nothing here: no error possible
            continue
        if binom.cdf(int(wrong[k].sum()), a, alpha) <= level:
            chosen = k
        else:
            break
    return chosen


def _fixed_sequence_marginal(loss: np.ndarray, cost: np.ndarray, alpha: float, level: float):
    """Cheapest setting certified on the marginal risk (Learn-then-Test, fixed sequence)."""
    n = loss.shape[1]
    valid = []
    for k in range(loss.shape[0]):
        if binom.cdf(int(loss[k].sum()), n, alpha) <= level:
            valid.append(k)
        else:
            break
    if not valid:
        return None
    return valid[int(np.argmin(cost[valid].mean(axis=1)))]


def run_guaranteed(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    cost_fn,
    lean_budget: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    ctl, ev = cfg["controller"], cfg["evaluation"]
    _grid, full = tier_replays(task_pools, ctl, ev, cost_fn)
    _grid, lean = tier_replays(task_pools, ctl, ev, cost_fn, budget=lean_budget)
    T = len(tier_names)
    m, n, perms = full[0].loss.shape
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(ev["n_splits"]):
        idx = rng.permutation(n)
        half = int(round(ev["calib_frac"] * n))
        splits.append((idx[:half], idx[half:], rng.integers(0, perms, size=n)))
    methods = ["toe", "toe-lean"] + [f"single-{t}" for t in tier_names]
    out: dict[str, Any] = {"tiers": tier_names, "n_tasks": n, "lean_budget": lean_budget,
                           "methods": {k: [] for k in methods}}  # fmt: skip

    def pick(a: np.ndarray, ids: np.ndarray, perm: np.ndarray) -> np.ndarray:
        return a[..., ids, perm[ids]]

    for alpha in ev["alphas"]:
        rows: dict[str, list[dict[str, float]]] = {k: [] for k in methods}
        for cal, test, perm in splits:
            for name, reps in (("toe", full), ("toe-lean", [*lean[:-1], full[-1]])):
                ks: list[int | None] = []
                open_ = np.ones(len(cal), bool)
                for rep in reps:
                    acc = pick(rep.accepted, cal, perm) > 0  # (m, cal)
                    wrong = acc & (pick(rep.correct, cal, perm) == 0)
                    k = _fixed_sequence_conditional(
                        acc[:, open_], wrong[:, open_], alpha, ev["delta"] / T
                    )
                    ks.append(k)
                    open_ &= ~acc[k]
                o = compose(reps, ks)
                rows[name].append(_stats(o, test, perm))
            for t, name in enumerate(tier_names):
                rep = full[t]
                k = _fixed_sequence_marginal(
                    pick(rep.loss, cal, perm), pick(rep.cost, cal, perm), alpha, ev["delta"]
                )
                if k is None:
                    continue
                ks = [None] * T
                ks[t] = k
                rows[f"single-{name}"].append(_stats(compose(full, ks), test, perm))
        for name in methods:
            r = rows[name]
            if not r:
                out["methods"][name].append({"alpha": alpha, "certified_splits": 0})
                continue
            out["methods"][name].append(
                {
                    "alpha": alpha,
                    "certified_splits": len(r),
                    "mean_test_risk": float(np.mean([x["test_risk"] for x in r])),
                    "violation_rate": float(np.mean([x["pool_risk"] > alpha for x in r])),
                    "mean_coverage": float(np.mean([x["coverage"] for x in r])),
                    "mean_solved": float(np.mean([x["solved"] for x in r])),
                    "mean_test_cost": float(np.mean([x["cost"] for x in r])),
                    "precision": float(np.mean([x["precision"] for x in r])),
                }
            )
    return out


def _stats(o: dict[str, np.ndarray], test: np.ndarray, perm: np.ndarray) -> dict[str, float]:
    t = lambda a: a[test, perm[test]]  # noqa: E731
    acc, cor = t(o["accepted"]), t(o["correct"])
    return {
        "test_risk": float(t(o["loss"]).mean()),
        "pool_risk": float(o["loss"].mean()),
        "coverage": float(acc.mean()),
        "solved": float(cor.mean()),
        "cost": float(t(o["cost"]).mean()),
        "precision": float(cor.sum() / max(acc.sum(), 1)),
    }
