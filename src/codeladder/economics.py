"""X13: when do ladders pay? Price sweeps and a break-even model.

A ladder spends cheap-tier tokens on every task and saves the top tier's tokens on the
tasks the cheap tier answers. Replaying a family once with the cost kept per tier lets any
price vector be applied afterwards: for each ratio r = price(top) / price(cheapest) the
calibrated sub-ladders (``subsets``) are compared with calibrated model selection
(``singles``) on the same splits.

Break-even model for a two-tier ladder (cheap c, top T): the ladder pays when the top
tier's tokens it saves cost more than the cheap tokens it spends,
    r * s_c * tok_T > tok_c   =>   r* = tok_c / (s_c * tok_T),
with s_c the share of tasks the calibrated ladder accepts at the cheap tier, tok_c the cheap
tokens it spends per task and tok_T the top tier's tokens per task when used alone (all from
the replay at the target α). The measured break-even is the smallest swept ratio from which
the ladder's selection objective (cost + κ x abstention) is lower than model selection's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .controller import run_task
from .evaluate import Replay, family_policies, make_cost_fn, run_splits
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


def _priced(r: dict[str, np.ndarray], prices: np.ndarray, rows: np.ndarray) -> Replay:
    cost = np.tensordot(prices, r["tok"][:, rows], axes=1)
    return Replay(
        loss=r["loss"][rows],
        accepted=r["accepted"][rows],
        correct=r["correct"][rows],
        cost=cost,
        fallbacks=np.zeros_like(cost),
    )


def _chosen_stats(rep: Replay, res: dict[str, Any], tier: np.ndarray, tok: np.ndarray, seed: int,
                  calib_frac: float) -> dict[str, float]:  # fmt: skip
    """Mean over certified splits, on the test halves (same split draws as run_splits):
    share of tasks accepted at the first tier, cheap tokens and top tokens per task."""
    rng = np.random.default_rng(seed)
    m, n, perms = rep.loss.shape
    s_c, tok_c, tok_t = [], [], []
    for row in res["splits"]:
        idx = rng.permutation(n)
        half = int(round(calib_frac * n))
        test = idx[half:]
        perm = rng.integers(0, perms, size=n)
        j = row["chosen"]
        if j is None:
            continue
        s_c.append(float((tier[j, test, perm[test]] == 0).mean()))
        tok_c.append(float(tok[0, j, test, perm[test]].mean()))
        tok_t.append(float(tok[-1, j, test, perm[test]].mean()))
    f = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {"cheap_share": f(s_c), "cheap_tokens": f(tok_c), "top_tokens": f(tok_t)}


def run_economics(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    ratios: list[float],
) -> dict[str, Any]:
    """Sweep the top tier's price (cheapest tier = 1, middle tiers at their real params_b
    ratio to the cheapest if ``tier_prices`` is given in cfg, else geometric) and compare
    sub-ladders with model selection at every α."""
    ctl, ev = cfg["controller"], cfg["evaluation"]
    score = ctl.get("score", "agreement")
    perms = int(ctl.get("permutations", 1))
    factor = float(ev.get("abstain_factor", 0.0))
    seed = 0
    configs, pols, _, paths = family_policies("subsets", tier_names, ctl, ev, {})
    method = "paths"  # _chosen_stats replays run_splits' split draws, which Pareto changes
    r = tier_replay(task_pools, pols, perms, tier_names, score)
    single_rows = np.array([i for i, c in enumerate(configs) if "+" not in c[0]])
    all_rows = np.arange(len(configs))
    # Paths of the singles family inside the subsets family (re-indexed).
    pos = {int(j): k for k, j in enumerate(single_rows)}
    single_paths = [[pos[j] for j in p] for p in paths if all(j in pos for j in p)]
    T = len(tier_names)
    base = cfg.get("tier_prices") or {}
    out: dict[str, Any] = {"tiers": tier_names, "n_tasks": len(task_pools), "rows": []}
    # Top-tier tokens per task when it is used alone (calibrated on its own path), for the
    # break-even model; tokens do not depend on prices.
    top_rows = np.array([i for i, c in enumerate(configs) if c[0] == tier_names[-1]])
    top_tok = {}
    for a in ev["alphas"]:
        rep_t = _priced(r, np.ones(T), top_rows)
        rt = run_splits(rep_t, a, ev["delta"], ev["n_splits"], ev["calib_frac"], seed=seed,
                        method="fixed_sequence", abstain_factor=factor)  # fmt: skip
        top_tok[a] = _chosen_stats(rep_t, rt, r["tier"][top_rows], r["tok"][:, top_rows], seed,
                                   ev["calib_frac"])["top_tokens"]  # fmt: skip
    for ratio in ratios:
        prices = np.array(
            [
                base.get(t, ratio ** (k / (T - 1)) if T > 1 else 1.0)
                for k, t in enumerate(tier_names)
            ]
        )
        prices[0], prices[-1] = 1.0, ratio
        rep_s = _priced(r, prices, single_rows)
        rep_l = _priced(r, prices, all_rows)
        for a in ev["alphas"]:
            args = (a, ev["delta"], ev["n_splits"], ev["calib_frac"])
            rs = run_splits(rep_s, *args, seed=seed, method="paths", paths=single_paths,
                            abstain_factor=factor)  # fmt: skip
            rl = run_splits(rep_l, *args, seed=seed, method=method, paths=paths,
                            abstain_factor=factor)  # fmt: skip
            # Selection objective on the test halves, with the same abstention price for both
            # (κ x the costliest setting of the larger family).
            c = factor * float(rep_l.cost.mean(axis=(1, 2)).max())

            def obj(x: dict[str, Any], c: float = c) -> float:
                return x["mean_test_cost"] + c * (1 - x["mean_coverage"])

            row = {
                "ratio": ratio,
                "alpha": a,
                "singles": {k: v for k, v in rs.items() if k != "splits"},
                "subsets": {k: v for k, v in rl.items() if k != "splits"},
                "objective_singles": obj(rs),
                "objective_subsets": obj(rl),
                "ladder": _chosen_stats(rep_l, rl, r["tier"], r["tok"], seed, ev["calib_frac"]),
                "single": _chosen_stats(
                    rep_s,
                    rs,
                    r["tier"][single_rows],
                    r["tok"][:, single_rows],
                    seed,
                    ev["calib_frac"],
                ),  # fmt: skip
            }
            out["rows"].append(row)
    # Break-even per α: predicted from the ladder's behaviour at the largest ratio (where it
    # is used), measured as the smallest ratio from which the ladder's objective is lower.
    out["break_even"] = []
    for a in ev["alphas"]:
        rows = [x for x in out["rows"] if x["alpha"] == a]
        hi = rows[-1]
        s_c, tok_c = hi["ladder"]["cheap_share"], hi["ladder"]["cheap_tokens"]
        tok_t = top_tok[a]
        pred = tok_c / (s_c * tok_t) if s_c and tok_t else float("inf")
        wins = [x["ratio"] for x in rows if x["objective_subsets"] < x["objective_singles"] * 0.99]
        meas = next((x for x in sorted(wins) if all(y in wins for y in ratios if y >= x)), None)
        out["break_even"].append(
            {
                "alpha": a,
                "cheap_share": s_c,
                "cheap_tokens": tok_c,
                "top_tokens": tok_t,
                "predicted": pred,
                "measured": meas,
            }  # fmt: skip
        )
    return out
