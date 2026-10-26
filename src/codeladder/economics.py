"""X13: when do ladders pay? Price sweeps and a break-even model.

A ladder spends cheap-tier tokens on every task and saves the top tier's tokens on the
tasks the cheap tier answers. Replaying the sub-ladder family once with the cost kept per
tier lets any price vector be applied afterwards. For each ratio r = price(top) /
price(cheapest) and each calibration split, both are certified with Learn-then-Test
(Bonferroni paths) and compared at matched coverage: model selection takes its certified
single-model setting with the highest calibration coverage; the ladder takes the cheapest
certified setting of the whole family (sub-ladders and single models) whose calibration
coverage is within ``slack`` of it. Both are scored on the test half. The choice uses only
calibration data, so the guarantee holds for both.

Break-even model for a two-tier ladder (cheap c, top T): the ladder pays when the top
tier's tokens it saves cost more than the cheap tokens it spends,
    r * s_c * tok_T > tok_c   =>   r* = tok_c / (s_c * tok_T),
with s_c the share of tasks the calibrated ladder accepts at the cheap tier, tok_c the cheap
tokens it spends per task and tok_T the top tier's tokens per task when used alone (all from
the replay at the target α). The measured break-even is the smallest swept ratio from which
the ladder is at least 1% cheaper than model selection at matched coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibrate import bonferroni_paths
from .controller import run_task
from .evaluate import family_policies, make_cost_fn
from .types import TierPool


def tier_replay(
    task_pools: list[list[TierPool]], policies, perms: int, tier_names: list[str], score: str
) -> dict[str, np.ndarray]:
    """Replay with unit prices, keeping tokens per tier: tok (tiers, configs, tasks, perms),
    plus loss, accepted, correct and the tier index of acceptance (-1: abstained)."""
    unit = make_cost_fn("weighted_tokens", {t: 1.0 for t in tier_names}, score)
    idx = {t: i for i, t in enumerate(tier_names)}
    m, n, T = len(policies), len(task_pools), len(tier_names)
    out = {k: np.zeros((m, n, perms)) for k in ("loss", "accepted", "correct")}
    out["tier"] = np.full((m, n, perms), -1.0)
    out["tok"] = np.zeros((T, m, n, perms))
    for j, pol in enumerate(policies):
        for i, pools in enumerate(task_pools):
            for p in range(perms):
                o = run_task(pools, pol, p)
                acc = o.accepted is not None
                ok = acc and o.accepted.hidden_pass
                out["loss"][j, i, p] = int(acc and not ok)
                out["accepted"][j, i, p] = int(acc)
                out["correct"][j, i, p] = int(ok)
                if acc:
                    out["tier"][j, i, p] = o.tier_index
                for c in o.drawn:
                    out["tok"][idx[c.tier], j, i, p] += unit(c)
    return out


def _test_stats(cost: np.ndarray, r: dict[str, np.ndarray], test: np.ndarray, perm: np.ndarray,
                j: int) -> tuple[float, float, float]:  # fmt: skip
    """Mean cost, solved share and coverage of setting j on the test tasks."""

    def take(a: np.ndarray) -> float:
        return float(a[j, test, perm[test]].mean())

    return take(cost), take(r["correct"]), take(r["accepted"])


def run_economics(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    ratios: list[float],
    slack: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep the top tier's price (cheapest tier 1, middle tiers geometric in between) and
    compare the cheapest matched-coverage ladder with two baselines at every α: the top
    tier alone ("top", the break-even model's reference) and model selection."""
    ctl, ev = cfg["controller"], cfg["evaluation"]
    score = ctl.get("score", "agreement")
    perms = int(ctl.get("permutations", 1))
    configs, pols, _, paths = family_policies("subsets", tier_names, ctl, ev, {})
    r = tier_replay(task_pools, pols, perms, tier_names, score)
    single = {i for i, c in enumerate(configs) if "+" not in c[0]}
    single_paths = [p for p in paths if all(j in single for j in p)]
    T, n = len(tier_names), r["loss"].shape[1]
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(ev["n_splits"]):
        idx = rng.permutation(n)
        half = int(round(ev["calib_frac"] * n))
        splits.append((idx[:half], idx[half:], rng.integers(0, perms, size=n)))

    def pick(a: np.ndarray, ids: np.ndarray, perm: np.ndarray) -> np.ndarray:
        return a[..., ids, perm[ids]]

    top_path = [p for p in single_paths if configs[p[0]][0] == tier_names[-1]]
    certified = {}  # (alpha, split) -> valid sets; certification does not depend on prices
    for a in ev["alphas"]:
        for s, (cal, _test, perm) in enumerate(splits):
            loss = pick(r["loss"], cal, perm)
            zero = np.zeros_like(loss)
            certified[a, s] = {
                "top": bonferroni_paths(loss, zero, a, ev["delta"], top_path).valid,
                "selection": bonferroni_paths(loss, zero, a, ev["delta"], single_paths).valid,
                "family": bonferroni_paths(loss, zero, a, ev["delta"], paths).valid,
            }
    out: dict[str, Any] = {"tiers": tier_names, "n_tasks": n, "slack": slack, "rows": []}
    for ratio in ratios:
        prices = np.array([ratio ** (k / (T - 1)) if T > 1 else 1.0 for k in range(T)])
        cost = np.tensordot(prices, r["tok"], axes=1)  # (configs, tasks, perms)
        for a in ev["alphas"]:
            row: dict[str, Any] = {"ratio": ratio, "alpha": a}
            for base in ("top", "selection"):
                stats = []
                for s, (cal, test, perm) in enumerate(splits):
                    vb, vf = certified[a, s][base], certified[a, s]["family"]
                    if not vb or not vf:
                        continue
                    cov = pick(r["accepted"], cal, perm).mean(axis=1)
                    ccost = pick(cost, cal, perm).mean(axis=1)
                    jb = max(vb, key=lambda j: (cov[j], -ccost[j]))  # noqa: B023
                    ok = [j for j in vf if cov[j] >= cov[jb] - slack]  # noqa: B023
                    jl = min(ok, key=lambda j: ccost[j]) if ok else jb  # noqa: B023
                    cb, sb, _ = _test_stats(cost, r, test, perm, jb)
                    cl, sl, _ = _test_stats(cost, r, test, perm, jl)
                    stats.append(
                        {
                            "cost_b": cb, "cost_l": cl, "solved_b": sb, "solved_l": sl,
                            "loss_l": float(pick(r["loss"], test, perm)[jl].mean()),
                            "cheap_share": float((pick(r["tier"], test, perm)[jl] == 0).mean()),
                            "cheap_tokens": float(pick(r["tok"][0], test, perm)[jl].mean()),
                            "top_tokens": float(pick(r["tok"][-1], test, perm)[jb].mean()),
                        }
                    )  # fmt: skip
                if not stats:
                    row[base] = {"certified_splits": 0}
                    continue
                m = {k: float(np.mean([x[k] for x in stats])) for k in stats[0]}
                row[base] = {
                    "certified_splits": len(stats),
                    "cost_ratio": m["cost_l"] / m["cost_b"],
                    "solved_baseline": m["solved_b"],
                    "solved_ladder": m["solved_l"],
                    "ladder_test_risk": m["loss_l"],
                    "cheap_share": m["cheap_share"],
                    "cheap_tokens": m["cheap_tokens"],
                    "top_tokens": m["top_tokens"],
                }
            out["rows"].append(row)
    out["break_even"] = []  # against the top tier alone
    for a in ev["alphas"]:
        rows = [x for x in out["rows"] if x["alpha"] == a and x["top"].get("certified_splits")]
        if not rows:
            continue
        hi = rows[-1]["top"]  # largest ratio: the ladder is used wherever it can be
        s_c, tok_c, tok_t = hi["cheap_share"], hi["cheap_tokens"], hi["top_tokens"]
        pred = tok_c / (s_c * tok_t) if s_c > 0 and tok_t > 0 else float("inf")
        wins = [x["ratio"] for x in rows if x["top"]["cost_ratio"] < 0.99]
        meas = next(
            (w for w in sorted(wins) if all(x["ratio"] in wins for x in rows if x["ratio"] >= w)),
            None,
        )
        out["break_even"].append(
            {
                "alpha": a,
                "cheap_share": s_c,
                "cheap_tokens": tok_c,
                "top_tokens": tok_t,
                "predicted": pred,
                "measured": meas,
                "asymptote": hi["cost_ratio"],
                "solved_diff_at_max": hi["solved_ladder"] - hi["solved_baseline"],
            }
        )
    return out
