"""X14: task-level bootstrap intervals for comparisons between two calibrated runs.

Each run folder holds the replay of every setting on every task (replay.npz). Two runs on
the same benchmarks and complete tiers list the same tasks in the same order (pools sorted by
task id), so a comparison can be paired: resample the tasks with replacement, rerun the whole
calibration (random calibration/test halves, Learn-then-Test, choice of a setting) on both
runs with the same split draws, and record the differences. The percentile interval over
resamples reflects both the finite task sample and the random splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import Replay, family_policies, run_splits


def load_run(path: str | Path, cfg: dict[str, Any]) -> tuple[Replay, dict[str, Any]]:
    """Replay arrays and calibration setup (method, paths, κ) of one evaluate run."""
    p = Path(path)
    s = json.loads((p / "summary.json").read_text())
    z = np.load(p / "replay.npz")
    rep = Replay(**{k: z[k] for k in ("loss", "accepted", "correct", "cost", "fallbacks")})
    ev = {**cfg["evaluation"], "abstain_factor": s.get("abstain_factor", 0.0)}
    family = s.get("family", "shared")
    setup = {"method": "fixed_sequence", "paths": None, "abstain_factor": ev["abstain_factor"]}
    if family != "shared" and len(s["tiers"]) > 1:
        ctl = {**cfg["controller"], "score": s.get("score", "agreement")}
        configs, _pols, method, paths = family_policies(family, s["tiers"], ctl, ev, {})
        if len(configs) != rep.loss.shape[0]:
            raise ValueError(f"{p}: settings do not match the current configuration")
        setup.update(method=s.get("calibration_method", method), paths=paths)
    return rep, setup | {"n_tasks": s["n_tasks"], "name": p.name}


def _take(rep: Replay, idx: np.ndarray) -> Replay:
    return Replay(**{k: getattr(rep, k)[:, idx, :] for k in rep.__dataclass_fields__})


def compare(
    run_a: str | Path,
    run_b: str | Path,
    cfg: dict[str, Any],
    alphas: list[float],
    n_boot: int = 200,
    n_splits: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap of A minus B: solved share and coverage differences, cost ratio."""
    ev = cfg["evaluation"]
    rep_a, sa = load_run(run_a, cfg)
    rep_b, sb = load_run(run_b, cfg)
    if sa["n_tasks"] != sb["n_tasks"]:
        raise ValueError(f"different task sets: {sa['n_tasks']} against {sb['n_tasks']}")
    n = sa["n_tasks"]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"a": sa["name"], "b": sb["name"], "n_tasks": n, "n_boot": n_boot,
                           "n_splits": n_splits, "alphas": []}  # fmt: skip

    def summary(rep: Replay, setup: dict[str, Any], a: float, s: int) -> dict[str, Any]:
        return run_splits(
            rep, a, ev["delta"], n_splits, ev["calib_frac"], seed=s, method=setup["method"],
            paths=setup["paths"], abstain_factor=setup["abstain_factor"],
        )  # fmt: skip

    for a in alphas:
        d_solved, d_cov, ratio = [], [], []
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ra = summary(_take(rep_a, idx), sa, a, 10_000 + b)
            rb = summary(_take(rep_b, idx), sb, a, 10_000 + b)
            if not ra["certified_splits"] or not rb["certified_splits"]:
                continue
            d_solved.append(ra["mean_solved"] - rb["mean_solved"])
            d_cov.append(ra["mean_coverage"] - rb["mean_coverage"])
            ratio.append(ra["mean_test_cost"] / rb["mean_test_cost"])
        pa = summary(rep_a, sa, a, 0)
        pb = summary(rep_b, sb, a, 0)

        def ci(v: list[float]) -> list[float]:
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else []

        out["alphas"].append(
            {
                "alpha": a,
                "solved_a": pa["mean_solved"],
                "solved_b": pb["mean_solved"],
                "cost_a": pa["mean_test_cost"],
                "cost_b": pb["mean_test_cost"],
                "solved_diff_ci": ci(d_solved),
                "coverage_diff_ci": ci(d_cov),
                "cost_ratio_ci": ci(ratio),
                "p_a_solves_more": float(np.mean(np.array(d_solved) > 0)) if d_solved else None,
                "p_a_cheaper": float(np.mean(np.array(ratio) < 1)) if ratio else None,
                "resamples_used": len(d_solved),
            }
        )
    return out
