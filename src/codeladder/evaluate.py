"""Offline replay + calibration experiments over cached pools.

Pipeline:
  1. For every task, permutation and threshold setting, replay the controller and
     record a 0/1 loss (accepted but fails hidden tests), cost, and coverage.
  2. For each of ``n_splits`` random calibration/test splits, run Learn-then-Test on
     the calibration tasks and measure risk, cost and coverage on the test tasks.
  3. Report the same numbers for simple baselines and the label-using oracle.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import learn_then_test
from .controller import Outcome, Policy, learn_action_table, oracle_outcome, run_task
from .types import Candidate, TierPool

CostFn = Callable[[Candidate], float]


def make_cost_fn(metric: str, params_b: dict[str, float]) -> CostFn:
    if metric == "weighted_tokens":
        return lambda c: (c.tokens_out + 0.1 * c.tokens_in) * params_b.get(c.tier, 1.0)
    if metric == "tokens":
        return lambda c: float(c.tokens_out + c.tokens_in)
    if metric == "energy_j":
        return lambda c: float(c.extra.get("energy_j", np.nan))
    if metric == "gpu_s":
        return lambda c: float(c.latency_s)
    raise ValueError(f"unknown cost metric {metric}")


@dataclass
class Replay:
    loss: np.ndarray  # (configs, tasks, perms) 0/1: accepted and hidden-wrong
    accepted: np.ndarray  # (configs, tasks, perms) 0/1
    correct: np.ndarray  # (configs, tasks, perms) 0/1: accepted and hidden-right
    cost: np.ndarray  # (configs, tasks, perms)
    fallbacks: np.ndarray  # (configs, tasks, perms)


def _summarise(o: Outcome, cost_fn: CostFn) -> tuple[int, int, int, float, int]:
    acc = o.accepted is not None
    ok = acc and o.accepted.hidden_pass
    return (
        int(acc and not ok),
        int(acc),
        int(ok),
        float(sum(cost_fn(c) for c in o.drawn)),
        o.repair_fallbacks,
    )


def replay(
    task_pools: list[list[TierPool]], policies: list[Policy], perms: int, cost_fn: CostFn
) -> Replay:
    shape = (len(policies), len(task_pools), perms)
    arrs = {k: np.zeros(shape) for k in ("loss", "accepted", "correct", "cost", "fallbacks")}
    for j, pol in enumerate(policies):
        for i, pools in enumerate(task_pools):
            for p in range(perms):
                vals = _summarise(run_task(pools, pol, p), cost_fn)
                for k, v in zip(
                    ("loss", "accepted", "correct", "cost", "fallbacks"), vals, strict=True
                ):
                    arrs[k][j, i, p] = v
    return Replay(**arrs)


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Thresholds from conservative to aggressive, preceded by 'never accept' (phi <= 1 < 1.01)."""
    n = int(round((start - stop) / step)) + 1
    return [1.01] + [round(start - k * step, 6) for k in range(n)]


def _pick(a: np.ndarray, ids: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Select tasks ``ids`` and, for each, the permutation chosen for this split."""
    return a[:, ids, perm[ids]]


def run_splits(
    rep: Replay, alpha: float, delta: float, n_splits: int, calib_frac: float, seed: int = 0
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m, n, perms = rep.loss.shape
    rows = []
    for s in range(n_splits):
        idx = rng.permutation(n)
        n_cal = int(round(calib_frac * n))
        cal, test = idx[:n_cal], idx[n_cal:]
        perm = rng.integers(0, perms, size=n)  # one shuffled sample order per task per split
        res = learn_then_test(_pick(rep.loss, cal, perm), _pick(rep.cost, cal, perm), alpha, delta)
        if res.chosen is None:
            rows.append({"split": s, "chosen": None})
            continue
        j = res.chosen
        acc = _pick(rep.accepted, test, perm)[j]
        cor = _pick(rep.correct, test, perm)[j]
        rows.append(
            {
                "split": s,
                "chosen": j,
                "test_risk": float(_pick(rep.loss, test, perm)[j].mean()),
                "test_cost": float(_pick(rep.cost, test, perm)[j].mean()),
                "coverage": float(acc.mean()),
                "precision": float(cor.sum() / max(acc.sum(), 1)),
                "solved": float(cor.mean()),
                "repair_fallbacks": float(_pick(rep.fallbacks, test, perm)[j].mean()),
            }
        )
    ok = [r for r in rows if r["chosen"] is not None]
    risks = np.array([r["test_risk"] for r in ok]) if ok else np.array([np.nan])
    return {
        "alpha": alpha,
        "delta": delta,
        "n_splits": n_splits,
        "certified_splits": len(ok),
        "mean_test_risk": float(np.nanmean(risks)),
        "violation_rate": float(np.mean(risks > alpha)) if ok else float("nan"),
        "mean_test_cost": float(np.mean([r["test_cost"] for r in ok])) if ok else float("nan"),
        "mean_coverage": float(np.mean([r["coverage"] for r in ok])) if ok else float("nan"),
        "mean_solved": float(np.mean([r["solved"] for r in ok])) if ok else float("nan"),
        "splits": rows,
    }


def fixed_policy_stats(rep: Replay, j: int) -> dict[str, float]:
    return {
        "risk": float(rep.loss[j].mean()),
        "cost": float(rep.cost[j].mean()),
        "coverage": float(rep.accepted[j].mean()),
        "solved": float(rep.correct[j].mean()),
    }


def run_experiment(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    cost_fn: CostFn,
    out_dir: str | Path,
) -> dict[str, Any]:
    ctl, ev = cfg["controller"], cfg["evaluation"]
    perms = int(ctl.get("permutations", 1))
    grid = threshold_grid(**ev["lambda_grid"])
    T = len(tier_names)
    table = learn_action_table(task_pools) if ctl["action"] == "adaptive" else {}
    policies = [
        Policy(
            [lam] * T, ctl["n_init"], ctl["max_draws"], ctl["action"], table, name=f"lambda={lam}"
        )
        for lam in grid
    ]
    rep = replay(task_pools, policies, perms, cost_fn)

    summary: dict[str, Any] = {
        "tiers": tier_names,
        "n_tasks": len(task_pools),
        "action": ctl["action"],
        "thresholds": grid,
        "calibrated": [
            run_splits(rep, a, ev["delta"], ev["n_splits"], ev["calib_frac"]) for a in ev["alphas"]
        ],
    }

    # Baselines: each tier alone, accepting its best candidate after n_init draws (no abstention).
    baselines = {}
    for t, name in enumerate(tier_names):
        single = [[pools[t]] for pools in task_pools]
        pol = Policy([0.0], ctl["n_init"], ctl["n_init"], "resample", name=f"only-{name}")
        baselines[f"only-{name}"] = fixed_policy_stats(replay(single, [pol], perms, cost_fn), 0)
    # Uncalibrated cascade with a hand-picked threshold (FrugalGPT-style heuristic).
    naive = Policy(
        [0.5] * T, ctl["n_init"], ctl["max_draws"], ctl["action"], table, name="naive-0.5"
    )
    baselines["naive-cascade-0.5"] = fixed_policy_stats(
        replay(task_pools, [naive], perms, cost_fn), 0
    )
    # Oracle: cheapest hidden-correct sample (label-using lower bound on cost).
    orc = [[oracle_outcome(p, ctl["max_draws"], k) for k in range(perms)] for p in task_pools]
    baselines["oracle"] = {
        "cost": float(np.mean([[sum(cost_fn(c) for c in o.drawn) for o in row] for row in orc])),
        "solved": float(np.mean([[o.accepted is not None for o in row] for row in orc])),
    }
    summary["baselines"] = baselines

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out / "replay.npz", **{k: getattr(rep, k) for k in rep.__dataclass_fields__}
    )
    return summary
