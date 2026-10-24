"""X14 paired bootstrap between two evaluate runs."""

import json

import numpy as np

from codeladder.bootstrap import compare
from codeladder.evaluate import threshold_grid

CFG = {
    "controller": {"n_init": 2, "max_draws": 4, "action": "resample"},
    "evaluation": {
        "alphas": [0.2],
        "delta": 0.1,
        "n_splits": 5,
        "calib_frac": 0.5,
        "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
    },
}


def _run(path, solved_rate, cost, seed):
    """A fake single-tier run: settings from never-accept to always-accept; accepted tasks
    are right with probability solved_rate."""
    rng = np.random.default_rng(seed)
    m, n, perms = len(threshold_grid(1.0, 0.0, 0.25)), 300, 1
    acc = np.zeros((m, n, perms))
    for j in range(1, m):
        acc[j] = 1.0
    right = rng.random((1, n, perms)) < solved_rate
    corr = acc * right
    np.savez(path / "replay.npz", loss=acc - corr, accepted=acc, correct=corr,
             cost=np.full((m, n, perms), cost), fallbacks=np.zeros((m, n, perms)))  # fmt: skip
    (path / "summary.json").write_text(
        json.dumps({"tiers": ["t1"], "family": "shared", "n_tasks": n, "abstain_factor": 0.0})
    )


def test_paired_bootstrap_separates_a_cheaper_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _run(a, 0.95, 1.0, 0)
    _run(b, 0.95, 2.0, 0)
    out = compare(a, b, CFG, [0.2], n_boot=30, n_splits=5)
    row = out["alphas"][0]
    assert row["cost_ratio_ci"] == [0.5, 0.5]
    lo, hi = row["solved_diff_ci"]
    assert lo <= 0 <= hi and hi - lo < 0.02  # same tasks, same labels: no difference
    assert row["p_a_cheaper"] == 1.0
