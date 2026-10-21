"""X4: a learned repair-or-resample policy, evaluated offline under the same guarantee.

When the controller has not accepted yet it either repairs the most promising failing
candidate (one cached repair round with the failure as feedback) or draws a fresh sample.
The learned policy decides from the context of that choice: tier, error class of the
candidate to repair, its visible pass-rate bucket and its repair depth
(``controller.action_keys``). The pools hold both outcomes for every context (the cached
repair child and the tier's fresh samples), so the policy is fit with full information
(a contextual bandit with counterfactual rewards): repair wherever a repair child is more
often correct than a fresh sample. It is trained on a held-out share of the problems and
compared, on the rest and the same calibration splits, with always-resample, always-repair
and the earlier label-free table (``controller.learn_action_table``).

Rewards: "hidden" uses the hidden-test labels of the training problems (allowed there, as for
the X9 router); "visible" uses only the visible tests and stress checks (label-free).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .controller import action_keys, learn_action_table
from .evaluate import Replay, family_policies, replay, run_splits
from .scoring import checks_ok
from .types import Candidate, TierPool


def _reward(c: Candidate, kind: str) -> int:
    if kind == "hidden":
        return int(checks_ok(c) and c.hidden_pass)
    return int(checks_ok(c))


def learn_action_policy(
    task_pools: list[list[TierPool]], reward: str = "hidden", min_n: int = 30
) -> tuple[dict[str, str], dict[str, Any]]:
    """Table over ``action_keys`` contexts (all three levels): "repair" where the repair child
    of a failing candidate in that context is more often useful than a fresh sample of the
    same tier, with at least ``min_n`` observed repairs; contexts with fewer are left out so
    the controller backs off to a coarser level (or resamples)."""
    fresh: dict[int, list[int]] = defaultdict(list)
    rep: dict[str, list[int]] = defaultdict(list)
    for pools in task_pools:
        for t, pool in enumerate(pools):
            fresh[t].extend(_reward(c, reward) for c in pool.samples)
            for s, chain in pool.repairs.items():
                prev = pool.samples[s]
                for child in chain:
                    if not prev.passes_visible:  # only failing candidates are repaired
                        for key in action_keys(t, prev):
                            rep[key].append(_reward(child, reward))
                    prev = child
    table, stats = {}, {}
    for key, outs in rep.items():
        t = int(key.split("|")[0])
        f = float(np.mean(fresh[t])) if fresh[t] else 0.0
        r = float(np.mean(outs))
        stats[key] = {"n": len(outs), "repair": r, "fresh": f}
        if len(outs) >= min_n:
            table[key] = "repair" if r > f else "resample"
    return table, stats


def _subset(rep: Replay, ids: np.ndarray) -> Replay:
    return Replay(**{k: getattr(rep, k)[:, ids, :] for k in rep.__dataclass_fields__})


def run_actions(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    cfg: dict[str, Any],
    cost_fn,
    train_frac: float = 0.3,
    seeds: int = 3,
    family: str = "singles",
) -> dict[str, Any]:
    ctl, ev = cfg["controller"], cfg["evaluation"]
    perms = int(ctl.get("permutations", 1))
    factor = float(ev.get("abstain_factor", 0.0))
    split_args = (ev["delta"], ev["n_splits"], ev["calib_frac"])
    n = len(task_pools)

    def fam(action: str, table: dict[str, str]):
        _, pols, method, paths = family_policies(
            family, tier_names, {**ctl, "action": action}, ev, table
        )
        return pols, method, paths

    fixed = {}
    for action in ("resample", "repair"):  # independent of the training split: replay once
        pols, method, paths = fam(action, {})
        fixed[action] = (replay(task_pools, pols, perms, cost_fn), method, paths)

    out: dict[str, Any] = {
        "tiers": tier_names,
        "n_tasks": n,
        "family": family,
        "train_frac": train_frac,
        "seeds": [],
    }
    for seed in range(seeds):
        perm = np.random.default_rng(2000 + seed).permutation(n)
        n_tr = int(round(train_frac * n))
        tr, rest = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        train = [task_pools[i] for i in tr]
        test_pools = [task_pools[i] for i in rest]
        tables = {
            "label_free_table": (learn_action_table(train), None),
            "learned_visible": learn_action_policy(train, reward="visible"),
            "learned_hidden": learn_action_policy(train, reward="hidden"),
        }
        variants: dict[str, tuple[Replay, str, Any]] = {
            a: (_subset(r, rest), m, p) for a, (r, m, p) in fixed.items()
        }
        for name, (table, _stats) in tables.items():
            pols, method, paths = fam("adaptive", table)
            variants[name] = (replay(test_pools, pols, perms, cost_fn), method, paths)
        res: dict[str, Any] = {
            "seed": seed,
            "tables": {
                name: {
                    "contexts": len(table),
                    "repair_contexts": sorted(k for k, v in table.items() if v == "repair"),
                }
                for name, (table, _s) in tables.items()
            },
            "learned_hidden_stats": tables["learned_hidden"][1],
            "variants": {},
        }
        for name, (rep, method, paths) in variants.items():
            rows = []
            for a in ev["alphas"]:
                r = run_splits(
                    rep, a, *split_args, seed=seed, method=method, paths=paths,
                    abstain_factor=factor,
                )  # fmt: skip
                rows.append({k: v for k, v in r.items() if k != "splits"})
            rows_fallback = float(rep.fallbacks.mean())
            res["variants"][name] = {"rows": rows, "mean_repair_fallbacks": rows_fallback}
        out["seeds"].append(res)
    keys = ("mean_test_risk", "violation_rate", "mean_test_cost", "mean_coverage", "mean_solved")
    out["mean"] = {
        v: [
            {"alpha": a}
            | {
                k: float(np.nanmean([s["variants"][v]["rows"][i][k] for s in out["seeds"]]))
                for k in keys
            }
            | {
                "certified_splits": int(
                    sum(s["variants"][v]["rows"][i]["certified_splits"] for s in out["seeds"])
                )
            }
            for i, a in enumerate(ev["alphas"])
        ]
        for v in out["seeds"][0]["variants"]
    }
    return out
