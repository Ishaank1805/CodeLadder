"""X9: a learned up-front router (RouteLLM style) under the same risk guarantee.

The router sees only the problem (statement text, benchmark, format, known difficulty, size
of the stated constraints) and predicts, for every tier, the share of that tier's samples
that pass the hidden tests. With a routing threshold tau a task goes to the cheapest tier
whose prediction is at least tau (the last tier if none is). The routed tier then runs the
usual single-tier acceptance rule with threshold lambda, and (tau, lambda) is calibrated
with Learn-then-Test like every other family: one fixed-sequence path per tau (lambda
descending), Bonferroni over the paths. The router is trained on a held-out share of the
tasks that is never used for calibration or testing.

The same procedure with the true pass shares in place of the predictions gives an oracle
router: an upper bound on what any up-front router could add (it reads labels).
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any

import numpy as np

from .baselines import fit_logistic
from .evaluate import Replay, family_policies, replay, run_splits
from .types import Task, TierPool

DIM = 1024  # hashed text features
_TOKEN = re.compile(r"[a-z_][a-z0-9_]*|\d+")
_POW = re.compile(r"10\s*\^\s*\{?\s*(\d+)")


def _tokens(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def text_counts(texts: list[str]) -> np.ndarray:
    """Hashed unigram and bigram counts, log-scaled."""
    X = np.zeros((len(texts), DIM))
    for i, t in enumerate(texts):
        for w in _tokens(t):
            X[i, zlib.crc32(w.encode()) % DIM] += 1.0
    return np.log1p(X)


def max_constraint(text: str) -> float:
    """log10 of the largest number stated in the problem (10^5, 200000, ...), 0 if none."""
    best = 0.0
    for m in _POW.finditer(text):
        best = max(best, float(m.group(1)))
    for m in re.finditer(r"\d[\d,]{2,}", text):
        v = m.group(0).replace(",", "")
        if v.isdigit():
            best = max(best, math.log10(max(int(v), 1)))
    return min(best, 30.0)


def meta_features(tasks: list[Task]) -> np.ndarray:
    # Columns that are zero for a run's tasks get zero weight, so adding a benchmark leaves the
    # routers of runs without it unchanged.
    benches = ["humaneval", "mbpp", "livecodebench", "bigcodebench"]
    modes = ["function", "method", "stdin", "bcb"]
    diffs = ["easy", "medium", "hard"]
    rows = []
    for t in tasks:
        rows.append(
            [math.log1p(len(t.prompt)), math.log1p(len(t.visible_inputs)), max_constraint(t.prompt)]
            + [float(t.benchmark == b) for b in benches]
            + [float(t.mode == m) for m in modes]
            + [float((t.difficulty or "").lower() == d) for d in diffs]
        )
    return np.array(rows, dtype=float)


class Router:
    """One L2-regularised logistic model per tier on [tf-idf text, standardised metadata]."""

    def __init__(self, l2_grid: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0), folds: int = 3):
        self.l2_grid, self.folds = l2_grid, folds

    def _design(self, counts: np.ndarray, meta: np.ndarray) -> np.ndarray:
        X = counts * self.idf
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return np.hstack([X, (meta - self.mu) / self.sd])

    def fit(self, counts: np.ndarray, meta: np.ndarray, y: np.ndarray, seed: int = 0) -> Router:
        """``y``: (tasks, tiers) pass shares in [0, 1] (soft labels)."""
        df = (counts > 0).sum(axis=0)
        self.idf = np.log((1 + len(counts)) / (1 + df)) + 1.0
        self.mu, self.sd = meta.mean(axis=0), meta.std(axis=0) + 1e-6
        X = self._design(counts, meta)
        rng = np.random.default_rng(seed)
        fold = rng.permutation(len(X)) % self.folds
        self.l2, self.w = [], []
        for t in range(y.shape[1]):
            best, best_l2 = np.inf, self.l2_grid[0]
            for l2 in self.l2_grid:  # cross-validated log loss on the router-training tasks
                ll = 0.0
                for f in range(self.folds):
                    tr, va = fold != f, fold == f
                    w = fit_logistic(X[tr], y[tr, t], l2=l2, iters=30)
                    p = np.clip(_sigmoid(X[va] @ w[:-1] + w[-1]), 1e-6, 1 - 1e-6)
                    ll -= float(np.sum(y[va, t] * np.log(p) + (1 - y[va, t]) * np.log(1 - p)))
                if ll < best:
                    best, best_l2 = ll, l2
            self.l2.append(best_l2)
            self.w.append(fit_logistic(X, y[:, t], l2=best_l2, iters=50))
        return self

    def predict(self, counts: np.ndarray, meta: np.ndarray) -> np.ndarray:
        X = self._design(counts, meta)
        return np.stack([_sigmoid(X @ w[:-1] + w[-1]) for w in self.w], axis=1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0], score[label <= 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pass_shares(task_pools: list[list[TierPool]]) -> np.ndarray:
    return np.array(
        [
            [np.mean([c.hidden_pass for c in p.samples]) if p.samples else 0.0 for p in ps]
            for ps in task_pools
        ]  # fmt: skip
    )


def route(pred: np.ndarray, tau: float) -> np.ndarray:
    """Cheapest tier whose predicted pass share is at least tau, else the last tier."""
    ok = pred >= tau
    return np.where(ok.any(axis=1), ok.argmax(axis=1), pred.shape[1] - 1)


def routed_replay(
    single: Replay, single_paths: list[list[int]], pred: np.ndarray, taus: list[float]
) -> tuple[Replay, list[list[int]], np.ndarray]:
    """Replay of the (tau, lambda) family built from the per-tier replays of the singles
    family (``single_paths[t][k]`` = tier t at the k-th threshold), restricted to the tasks
    of ``pred``'s rows (the replay's task axis must match). Returns the replay, its paths and
    the routed tier per (tau, task)."""
    K = len(single_paths[0])
    n = pred.shape[0]
    tiers = np.stack([route(pred, tau) for tau in taus])  # (taus, tasks)
    idx = np.array(single_paths)  # (tiers, K)
    rows = idx[tiers]  # (taus, tasks, K): config row of the singles replay
    cols = np.arange(n)
    arrs = {}
    for name in single.__dataclass_fields__:
        a = getattr(single, name)  # (configs, tasks, perms)
        arrs[name] = np.stack(
            [a[rows[ti, :, k], cols, :] for ti in range(len(taus)) for k in range(K)]
        )
    paths = [[ti * K + k for k in range(K)] for ti in range(len(taus))]
    return Replay(**arrs), paths, tiers


def run_router(
    task_pools: list[list[TierPool]],
    tasks: list[Task],
    tier_names: list[str],
    cfg: dict[str, Any],
    cost_fn,
    router_frac: float = 0.3,
    router_seeds: int = 3,
    taus: list[float] | None = None,
    with_subsets: bool = False,
) -> dict[str, Any]:
    ctl, ev = cfg["controller"], cfg["evaluation"]
    perms = int(ctl.get("permutations", 1))
    factor = float(ev.get("abstain_factor", 0.0))
    taus = taus or [1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    _, sp, _, s_paths = family_policies("singles", tier_names, ctl, ev, {})
    single = replay(task_pools, sp, perms, cost_fn)
    if with_subsets:
        _, bp, b_method, b_paths = family_policies("subsets", tier_names, ctl, ev, {})
        subsets = replay(task_pools, bp, perms, cost_fn)
    y = pass_shares(task_pools)
    counts, meta = text_counts([t.prompt for t in tasks]), meta_features(tasks)
    n = len(task_pools)
    split_args = (ev["delta"], ev["n_splits"], ev["calib_frac"])
    out: dict[str, Any] = {
        "tiers": tier_names,
        "n_tasks": n,
        "router_frac": router_frac,
        "router_seeds": router_seeds,
        "taus": taus,
        "abstain_factor": factor,
        "seeds": [],
    }
    for seed in range(router_seeds):
        perm = np.random.default_rng(1000 + seed).permutation(n)
        n_tr = int(round(router_frac * n))
        tr, rest = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        model = Router().fit(counts[tr], meta[tr], y[tr], seed=seed)
        pred = model.predict(counts[rest], meta[rest])
        quality = {
            name: {
                "auroc_pass_majority": auroc(pred[:, t], (y[rest, t] >= 0.5).astype(float)),
                "corr": _corr(pred[:, t], y[rest, t]),
                "l2": model.l2[t],
            }
            for t, name in enumerate(tier_names)
        }
        s_rest = single.take(rest)
        families = {"singles": (s_rest, s_paths)}
        r_rep, r_paths, r_tiers = routed_replay(s_rest, s_paths, pred, taus)
        families["router"] = (r_rep, r_paths)
        o_rep, o_paths, o_tiers = routed_replay(s_rest, s_paths, y[rest], taus)
        families["oracle_router"] = (o_rep, o_paths)
        if with_subsets:
            families["subsets"] = (subsets.take(rest), b_paths)
        res: dict[str, Any] = {"seed": seed, "router_quality": quality, "families": {}}
        for fam, (rep, paths) in families.items():
            method = b_method if fam == "subsets" else "paths"
            rows = []
            for a in ev["alphas"]:
                r = run_splits(
                    rep, a, *split_args, seed=seed, method=method, paths=paths,
                    abstain_factor=factor,
                )  # fmt: skip
                chosen = [s["chosen"] for s in r["splits"] if s["chosen"] is not None]
                brief = {k: v for k, v in r.items() if k != "splits"}
                if fam in ("router", "oracle_router") and chosen:
                    K = len(s_paths[0])
                    tiers_used = (r_tiers if fam == "router" else o_tiers)[
                        [c // K for c in chosen]
                    ]  # (splits, tasks)
                    brief["chosen_taus"] = [taus[c // K] for c in chosen]
                    brief["routed_share"] = {
                        name: float((tiers_used == t).mean()) for t, name in enumerate(tier_names)
                    }
                rows.append(brief)
            res["families"][fam] = rows
        out["seeds"].append(res)
    # Mean over router seeds, per family and alpha.
    keys = ("mean_test_risk", "violation_rate", "mean_test_cost", "mean_coverage", "mean_solved")
    out["mean"] = {
        fam: [
            {"alpha": a}
            | {k: float(np.nanmean([s["families"][fam][i][k] for s in out["seeds"]])) for k in keys}
            | {
                "certified_splits": int(
                    sum(s["families"][fam][i]["certified_splits"] for s in out["seeds"])
                )
            }  # fmt: skip
            for i, a in enumerate(ev["alphas"])
        ]
        for fam in out["seeds"][0]["families"]
    }
    return out
