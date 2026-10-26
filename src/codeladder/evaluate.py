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

from .calibrate import bonferroni_paths, learn_then_test, pareto_testing
from .controller import Outcome, Policy, learn_action_table, oracle_outcome, run_task
from .types import Candidate, TierPool

CostFn = Callable[[Candidate], float]


def make_cost_fn(metric: str, params_b: dict[str, float], score: str = "agreement") -> CostFn:
    # The verbalised-confidence score needs one extra request per visible-passing candidate.
    uses_conf = score in ("selfconf", "agreement_selfconf")
    uses_ptrue = score in ("ptrue", "agreement_ptrue")
    if uses_conf:
        extra = lambda c: c.extra.get("self_conf_tokens", 0)  # noqa: E731
    elif uses_ptrue:
        extra = lambda c: c.extra.get("p_true_tokens", 0)  # noqa: E731
    else:
        extra = lambda c: 0  # noqa: E731
    if metric == "weighted_tokens":
        # The confidence request is almost entirely input (prompt + code), so it is weighted
        # like other input tokens (0.1); its output is a single number.
        return lambda c: (c.tokens_out + 0.1 * (c.tokens_in + extra(c))) * params_b.get(c.tier, 1.0)
    if metric == "tokens":
        return lambda c: float(c.tokens_out + c.tokens_in + extra(c))
    if metric == "energy_j":
        # Measured generation energy; the confidence request is priced like other input tokens
        # (0.1 x the tier's joules per output token).
        return lambda c: (
            float(c.extra.get("energy_j", np.nan))
            + 0.1 * extra(c) * float(c.extra.get("joules_per_token", 0.0))
        )
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

    def take(self, ids: np.ndarray) -> Replay:
        """The same replay restricted to some tasks (in the given order)."""
        return Replay(**{k: getattr(self, k)[:, ids, :] for k in self.__dataclass_fields__})


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


def selection_cost(cal_cost: np.ndarray, cal_acc: np.ndarray, abstain_factor: float) -> np.ndarray:
    """Per-task objective used to choose among certified settings: cost plus a price for
    abstaining, c = abstain_factor x the mean cost of the costliest setting on these tasks.
    With 0 it is plain cost; then a setting that abstains cheaply can win, which matters once
    the family contains cheap sub-ladders. Any choice among certified settings keeps the
    guarantee (Learn-then-Test controls the error of all of them jointly)."""
    if not abstain_factor:
        return cal_cost
    c = abstain_factor * float(cal_cost.mean(axis=1).max())
    return cal_cost + c * (1.0 - cal_acc)


def run_splits(
    rep: Replay,
    alpha: float,
    delta: float,
    n_splits: int,
    calib_frac: float,
    seed: int = 0,
    method: str = "fixed_sequence",
    paths: list[list[int]] | None = None,
    abstain_factor: float = 0.0,
    n_cal: int | None = None,
) -> dict[str, Any]:
    """Calibrate on random splits. ``method``: "fixed_sequence" (configs ordered from
    conservative to aggressive, 1-D families), "paths" (Bonferroni over fixed-sequence paths)
    or "pareto" (Pareto Testing, any family). Settings are chosen by ``selection_cost``.
    ``n_cal`` uses only that many of the calibration-half tasks (learning curves); the test
    half is unchanged."""
    rng = np.random.default_rng(seed)
    m, n, perms = rep.loss.shape
    rows = []
    for s in range(n_splits):
        idx = rng.permutation(n)
        half = int(round(calib_frac * n))
        cal, test = idx[: min(n_cal or half, half)], idx[half:]
        perm = rng.integers(0, perms, size=n)  # one shuffled sample order per task per split
        cal_loss = _pick(rep.loss, cal, perm)
        cal_cost = selection_cost(
            _pick(rep.cost, cal, perm), _pick(rep.accepted, cal, perm), abstain_factor
        )
        if method == "pareto":
            res = pareto_testing(cal_loss, cal_cost, alpha, delta, rng)
        elif method == "paths":
            res = bonferroni_paths(cal_loss, cal_cost, alpha, delta, paths or [])
        else:
            res = learn_then_test(cal_loss, cal_cost, alpha, delta)
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
                # Risk of the chosen setting over the whole pool (all tasks and sample orders):
                # the stand-in for its true risk, which is what Learn-then-Test bounds.
                "pool_risk": float(rep.loss[j].mean()),
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
        "mean_test_risk": float(np.mean(risks)) if ok else float("nan"),
        # Guarantee check: share of splits whose chosen setting has pool (true) risk above alpha.
        "violation_rate": float(np.mean([r["pool_risk"] > alpha for r in ok]))
        if ok
        else float("nan"),
        # Descriptive only: the finite test half adds sampling noise on top of the true risk.
        "test_exceed_rate": float(np.mean(risks > alpha)) if ok else float("nan"),
        "mean_test_cost": float(np.mean([r["test_cost"] for r in ok])) if ok else float("nan"),
        "mean_coverage": float(np.mean([r["coverage"] for r in ok])) if ok else float("nan"),
        "mean_solved": float(np.mean([r["solved"] for r in ok])) if ok else float("nan"),
        "splits": rows,
    }


def run_transfer(
    rep: Replay,
    cal_pool: np.ndarray,
    test_ids: np.ndarray,
    alpha: float,
    delta: float,
    n_splits: int,
    calib_frac: float,
    seed: int = 0,
    method: str = "fixed_sequence",
    paths: list[list[int]] | None = None,
    abstain_factor: float = 0.0,
) -> dict[str, Any]:
    """X6: calibrate on a random ``calib_frac`` of the source tasks, test on every target task.
    The guarantee only covers the source distribution; the violation rate shows whether it
    transfers (target risk of the chosen setting above alpha)."""
    rng = np.random.default_rng(seed)
    perms = rep.loss.shape[2]
    rows = []
    for _ in range(n_splits):
        cal = rng.permutation(cal_pool)[: int(round(calib_frac * len(cal_pool)))]
        perm = rng.integers(0, perms, size=rep.loss.shape[1])
        cal_loss = _pick(rep.loss, cal, perm)
        cal_cost = selection_cost(
            _pick(rep.cost, cal, perm), _pick(rep.accepted, cal, perm), abstain_factor
        )
        if method == "pareto":
            res = pareto_testing(cal_loss, cal_cost, alpha, delta, rng)
        elif method == "paths":
            res = bonferroni_paths(cal_loss, cal_cost, alpha, delta, paths or [])
        else:
            res = learn_then_test(cal_loss, cal_cost, alpha, delta)
        if res.chosen is None:
            continue
        j = res.chosen
        rows.append(
            (
                float(rep.loss[j][test_ids].mean()),
                float(rep.cost[j][test_ids].mean()),
                float(rep.correct[j][test_ids].mean()),
                float(rep.accepted[j][test_ids].mean()),
            )
        )
    r = np.array(rows) if rows else np.zeros((0, 4))
    return {
        "alpha": alpha,
        "certified_splits": len(r),
        "target_risk": float(r[:, 0].mean()) if len(r) else float("nan"),
        "violation_rate": float((r[:, 0] > alpha).mean()) if len(r) else float("nan"),
        "mean_test_cost": float(r[:, 1].mean()) if len(r) else float("nan"),
        "mean_solved": float(r[:, 2].mean()) if len(r) else float("nan"),
        "mean_coverage": float(r[:, 3].mean()) if len(r) else float("nan"),
    }


def fixed_policy_stats(rep: Replay, j: int) -> dict[str, float]:
    return {
        "risk": float(rep.loss[j].mean()),
        "cost": float(rep.cost[j].mean()),
        "coverage": float(rep.accepted[j].mean()),
        "solved": float(rep.correct[j].mean()),
    }


def _brief(split_summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in split_summary.items() if k != "splits"}


def family_policies(
    family: str, tier_names: list[str], ctl: dict[str, Any], ev: dict[str, Any], table: dict
) -> tuple[list[Any], list[Policy], str, list[list[int]] | None]:
    """Settings of a calibrated family: (configs, policies, calibration method, paths)."""
    T = len(tier_names)
    score = ctl.get("score", "agreement")
    if family == "two_level":
        # Separate thresholds: lam_e to accept at the cheaper tiers, lam_f at the final tier.
        grid2 = threshold_grid(**ev.get("lambda_grid_2d", ev["lambda_grid"]))
        configs = [(le, lf) for le in grid2 for lf in grid2]
        policies = [
            Policy(
                [le] * (T - 1) + [lf],
                ctl["n_init"],
                ctl["max_draws"],
                ctl["action"],
                table,
                name=f"lambda_e={le},lambda_f={lf}",
                score=score,
            )  # fmt: skip
            for le, lf in configs
        ]
        method = ev.get("two_level_method", "paths")
        # Paths: one per pre-specified escalation threshold, final-tier threshold descending.
        esc = ev.get("escalation_thresholds", [1.01, 0.9, 0.7, 0.5])
        paths = [[i for i, (le, _lf) in enumerate(configs) if abs(le - e) < 1e-9] for e in esc]
        paths = [p for p in paths if p]
    elif family in ("subsets", "singles"):
        # Sub-ladders: any non-empty subset of the tiers, in order. A single tier is one path
        # (its threshold descending); a longer ladder is one path per escalation threshold at
        # its earlier tiers (never-accept there is dominated by the last tier alone).
        grid2 = threshold_grid(**ev.get("lambda_grid_2d", ev["lambda_grid"]))
        esc = [e for e in ev.get("escalation_thresholds", [0.9, 0.7, 0.5]) if e <= 1.0]
        configs, policies, paths = [], [], []
        # "singles" keeps only one-tier ladders: calibrated model selection, the fair baseline
        # for "pick the best single model" (it pays the same Bonferroni price per model).
        for mask in range(1, 2**T):
            sub = [t for t in range(T) if mask >> t & 1]
            if family == "singles" and len(sub) > 1:
                continue
            for e in esc if len(sub) > 1 else [None]:
                path = []
                for lf in grid2:
                    th: list[float | None] = [None] * T
                    for t in sub[:-1]:
                        th[t] = e
                    th[sub[-1]] = lf
                    names = "+".join(tier_names[t] for t in sub)
                    configs.append([names, e, lf])
                    path.append(len(policies))
                    policies.append(
                        Policy(
                            th,
                            ctl["n_init"],
                            ctl["max_draws"],
                            ctl["action"],
                            table,
                            name=f"{names},lambda_e={e},lambda_f={lf}",
                            score=score,
                        )  # fmt: skip
                    )
                paths.append(path)
        method = ev.get("two_level_method", "paths")  # paths or pareto (Pareto Testing)
    else:
        grid = threshold_grid(**ev["lambda_grid"])
        configs = grid
        policies = [
            Policy(
                [lam] * T,
                ctl["n_init"],
                ctl["max_draws"],
                ctl["action"],
                table,
                name=f"lambda={lam}",
                score=score,
            )  # fmt: skip
            for lam in grid
        ]
        method, paths = "fixed_sequence", None
    for pol in policies:  # ladder design: draws allowed below the last tier (ablation)
        pol.lower_budget = ctl.get("lower_budget")
    return configs, policies, method, paths


def run_experiment(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    cost_fn: CostFn,
    out_dir: str | Path,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    ctl, ev = cfg["controller"], cfg["evaluation"]
    perms = int(ctl.get("permutations", 1))
    score = ctl.get("score", "agreement")
    grid = threshold_grid(**ev["lambda_grid"])
    T = len(tier_names)
    table = learn_action_table(task_pools) if ctl["action"] == "adaptive" else {}

    def ladder(lam: float, pools_T: int, action: str = ctl["action"], name: str = "") -> Policy:
        return Policy(
            [lam] * pools_T, ctl["n_init"], ctl["max_draws"], action, table, name=name, score=score
        )

    family = ev.get("family", "shared") if T > 1 else "shared"
    factor = float(ev.get("abstain_factor", 0.0))
    configs, policies, method, paths = family_policies(family, tier_names, ctl, ev, table)
    rep = replay(task_pools, policies, perms, cost_fn)
    split_args = (ev["delta"], ev["n_splits"], ev["calib_frac"])

    summary: dict[str, Any] = {
        "tiers": tier_names,
        "n_tasks": len(task_pools),
        "task_ids": [tp[0].task_id for tp in task_pools],  # replay order (paired comparisons)
        "action": ctl["action"],
        "score": score,
        "family": family,
        "lower_budget": ctl.get("lower_budget"),
        "thresholds": configs,
        "calibration_method": method,
        "abstain_factor": factor,
        "calibrated": [
            run_splits(rep, a, *split_args, method=method, paths=paths, abstain_factor=factor)
            for a in ev["alphas"]
        ],
        "calibrated_grouped": (
            [
                run_splits_grouped(
                    rep, groups, a, *split_args, method=method, paths=paths, abstain_factor=factor
                )
                for a in ev["alphas"]
            ]
            if groups and len(set(groups)) > 1
            else []
        ),
    }

    if groups and len(set(groups)) > 1:
        # Conditional risk of the pooled calibration: the guarantee is marginal over tasks, so
        # report each group's true risk under the setting chosen for the whole pool.
        g = np.array(groups)
        summary["conditional_risk"] = []
        for cal in summary["calibrated"]:
            chosen = [r["chosen"] for r in cal["splits"] if r.get("chosen") is not None]
            per = {}
            for name in sorted(set(groups)):
                idx = np.where(g == name)[0]
                risks = [float(rep.loss[j][idx].mean()) for j in chosen]
                per[name] = {
                    "n_tasks": int(len(idx)),
                    "mean_risk": float(np.mean(risks)) if risks else float("nan"),
                    "exceed_rate": float(np.mean([r > cal["alpha"] for r in risks]))
                    if risks
                    else float("nan"),
                }
            summary["conditional_risk"].append({"alpha": cal["alpha"], "groups": per})

    baselines: dict[str, Any] = {}
    for t, name in enumerate(tier_names):
        single = [[pools[t]] for pools in task_pools]
        # Uncalibrated: accept the best visible-passing candidate after n_init draws.
        pol = Policy(
            [0.0], ctl["n_init"], ctl["n_init"], "resample", name=f"only-{name}", score=score
        )
        baselines[f"only-{name}"] = fixed_policy_stats(replay(single, [pol], perms, cost_fn), 0)
        # Calibrated single model (risk-controlled acceptance with abstention, no ladder).
        single_rep = replay(single, [ladder(lam, 1) for lam in grid], perms, cost_fn)
        baselines[f"calibrated-only-{name}"] = [
            _brief(run_splits(single_rep, a, *split_args, abstain_factor=factor))
            for a in ev["alphas"]
        ]
    # Uncalibrated ladder with a hand-picked threshold (FrugalGPT-style heuristic).
    baselines["naive-cascade-0.5"] = fixed_policy_stats(
        replay(task_pools, [ladder(0.5, T, name="naive-0.5")], perms, cost_fn), 0
    )
    # Oracle: cheapest hidden-correct sample (label-using lower bound on cost).
    orc = [[oracle_outcome(p, ctl["max_draws"], k) for k in range(perms)] for p in task_pools]
    baselines["oracle"] = {
        "cost": float(np.mean([[sum(cost_fn(c) for c in o.drawn) for o in row] for row in orc])),
        "solved": float(np.mean([[o.accepted is not None for o in row] for row in orc])),
    }
    summary["baselines"] = baselines
    # X6: calibrate on some benchmarks, test on the others.
    cal_on = ev.get("calib_on") or []
    if cal_on and groups:
        g = np.array(groups)
        src, tgt = np.where(np.isin(g, cal_on))[0], np.where(~np.isin(g, cal_on))[0]
        if len(src) and len(tgt):
            summary["transfer"] = {
                "calibrate_on": cal_on,
                "test_on": sorted(set(g[tgt])),
                "results": [
                    run_transfer(
                        rep,
                        src,
                        tgt,
                        a,
                        ev["delta"],
                        ev["n_splits"],
                        ev["calib_frac"],
                        method=method,
                        paths=paths,
                        abstain_factor=factor,
                    )  # fmt: skip
                    for a in ev["alphas"]
                ],
            }
    # X5: learning curve of the main family (same splits and test halves, fewer calibration tasks).
    summary["learning_curve"] = [
        {"n_cal": k}
        | _brief(
            run_splits(
                rep, a, *split_args, method=method, paths=paths, abstain_factor=factor, n_cal=k
            )
        )
        for k in ev.get("calib_sizes", [])
        for a in ev["alphas"]
    ]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out / "replay.npz", **{k: getattr(rep, k) for k in rep.__dataclass_fields__}
    )
    return summary


def run_shift(
    rep: Replay, dates: list[str], alpha: float, delta: float, n_windows: int = 5
) -> dict[str, Any]:
    """E5: does the guarantee survive temporal shift?

    Tasks are ordered by release date and cut into ``n_windows + 1`` equal chunks. The first
    chunk is the initial calibration set. For every later chunk we compare
      - fixed: thresholds calibrated once on the first chunk, and
      - sliding: thresholds recalibrated on the chunk just before it (same size),
    reporting each window's risk (averaged over sample orders), coverage and cost.
    """
    order = np.argsort(np.array(dates, dtype=object))
    chunks = np.array_split(order, n_windows + 1)
    perm0 = np.zeros(len(dates), dtype=int)

    def calibrate(ids: np.ndarray) -> int | None:
        res = learn_then_test(
            _pick(rep.loss, ids, perm0), _pick(rep.cost, ids, perm0), alpha, delta
        )
        return res.chosen

    def window_stats(j: int | None, ids: np.ndarray) -> dict[str, float]:
        if j is None:
            return {"risk": 0.0, "coverage": 0.0, "cost": float("nan"), "certified": False}
        return {
            "risk": float(rep.loss[j][ids].mean()),
            "coverage": float(rep.accepted[j][ids].mean()),
            "cost": float(rep.cost[j][ids].mean()),
            "certified": True,
        }

    fixed_j = calibrate(chunks[0])
    windows = []
    for k in range(1, len(chunks)):
        ids = chunks[k]
        windows.append(
            {
                "window": k,
                "first_date": str(dates[ids[0]]),
                "last_date": str(dates[ids[-1]]),
                "n_tasks": int(len(ids)),
                "fixed": window_stats(fixed_j, ids),
                "sliding": window_stats(calibrate(chunks[k - 1]), ids),
            }
        )
    return {
        "alpha": alpha,
        "delta": delta,
        "calibration_window": [str(dates[chunks[0][0]]), str(dates[chunks[0][-1]])],
        "windows": windows,
        "fixed_violations": int(sum(w["fixed"]["risk"] > alpha for w in windows)),
        "sliding_violations": int(sum(w["sliding"]["risk"] > alpha for w in windows)),
    }


def run_splits_grouped(
    rep: Replay,
    groups: list[str],
    alpha: float,
    delta: float,
    n_splits: int,
    calib_frac: float,
    seed: int = 0,
    method: str = "fixed_sequence",
    paths: list[list[int]] | None = None,
    abstain_factor: float = 0.0,
) -> dict[str, Any]:
    """Group-conditional (Mondrian) calibration.

    Each group (for example each benchmark) is calibrated on its own tasks at level
    delta / G, so with probability at least 1 - delta every group's true risk is at or below
    alpha, and hence so is the risk of any mixture of the groups. Test metrics are pooled
    over all groups' test tasks, each scored with its own group's chosen setting.
    """
    rng = np.random.default_rng(seed)
    m, n, perms = rep.loss.shape
    labels = np.array(groups)
    names = sorted(set(groups))
    level = delta / len(names)
    rows = []
    for s in range(n_splits):
        perm = rng.integers(0, perms, size=n)
        loss_t, acc_t, cor_t, cost_t, pool_risk_w, chosen = [], [], [], [], 0.0, {}
        certified = True
        for g in names:
            gidx = rng.permutation(np.where(labels == g)[0])
            n_cal = int(round(calib_frac * len(gidx)))
            cal, test = gidx[:n_cal], gidx[n_cal:]
            cal_loss = _pick(rep.loss, cal, perm)
            cal_cost = selection_cost(
                _pick(rep.cost, cal, perm), _pick(rep.accepted, cal, perm), abstain_factor
            )
            if method == "pareto":
                res = pareto_testing(cal_loss, cal_cost, alpha, level, rng)
            elif method == "paths":
                res = bonferroni_paths(cal_loss, cal_cost, alpha, level, paths or [])
            else:
                res = learn_then_test(cal_loss, cal_cost, alpha, level)
            j = res.chosen if res.chosen is not None else 0  # 0 = abstain everywhere
            certified &= res.chosen is not None
            chosen[g] = int(j)
            loss_t.append(_pick(rep.loss, test, perm)[j])
            acc_t.append(_pick(rep.accepted, test, perm)[j])
            cor_t.append(_pick(rep.correct, test, perm)[j])
            cost_t.append(_pick(rep.cost, test, perm)[j])
            gall = np.where(labels == g)[0]
            pool_risk_w += float(rep.loss[j][gall].mean()) * len(gall) / n
        lt, at, ct, cst = (np.concatenate(x) for x in (loss_t, acc_t, cor_t, cost_t))
        rows.append(
            {
                "split": s,
                "chosen": chosen,
                "all_groups_certified": bool(certified),
                "test_risk": float(lt.mean()),
                "pool_risk": pool_risk_w,
                "test_cost": float(cst.mean()),
                "coverage": float(at.mean()),
                "precision": float(ct.sum() / max(at.sum(), 1)),
                "solved": float(ct.mean()),
            }
        )
    risks = np.array([r["test_risk"] for r in rows])
    return {
        "alpha": alpha,
        "delta": delta,
        "groups": names,
        "n_splits": n_splits,
        "certified_splits": int(sum(r["all_groups_certified"] for r in rows)),
        "mean_test_risk": float(risks.mean()),
        "violation_rate": float(np.mean([r["pool_risk"] > alpha for r in rows])),
        "test_exceed_rate": float(np.mean(risks > alpha)),
        "mean_test_cost": float(np.mean([r["test_cost"] for r in rows])),
        "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
        "mean_solved": float(np.mean([r["solved"] for r in rows])),
        "splits": rows,
    }
