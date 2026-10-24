"""X16 guaranteed baselines: composition of per-tier replays and the Trust-or-Escalate cascade."""

import numpy as np
from conftest import cand, pool

from codeladder.evaluate import make_cost_fn
from codeladder.guaranteed import compose, run_guaranteed, tier_replays

CFG = {
    "controller": {"n_init": 2, "max_draws": 2, "action": "resample", "permutations": 2},
    "evaluation": {
        "alphas": [0.1],
        "delta": 0.1,
        "n_splits": 5,
        "calib_frac": 0.5,
        "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.5},
    },
}


def _pools(n=400):
    """Cheap tier: right and agreeing on easy tasks, two disagreeing wrong answers on hard ones
    (visible-passing, so only agreement separates them); top tier always right."""
    out = []
    for i in range(n):
        easy = i % 2 == 0
        cheap = [
            cand("c", j, visible=True, hidden=easy, sig="A" if easy else f"w{j}", tokens=10)
            for j in range(2)
        ]
        top = [cand("T", j, visible=True, hidden=True, sig="A", tokens=10) for j in range(2)]
        out.append([pool("c", cheap), pool("T", top)])
    return out


def test_compose_takes_the_first_accepting_tier():
    pools = _pools(20)
    grid, reps = tier_replays(
        pools, CFG["controller"], CFG["evaluation"], make_cost_fn("tokens", {})
    )
    k1 = grid.index(1.0)  # cheap tier accepts only full agreement: the easy half
    o = compose(reps, [k1, k1])
    assert o["accepted"].all() and o["correct"].all()
    assert np.allclose(o["cost"][0::2].mean(), reps[0].cost[k1][0::2].mean())  # easy: cheap only
    assert o["cost"][1::2].mean() > o["cost"][0::2].mean()  # hard: both tiers paid
    assert compose(reps, [None, 0])["accepted"].sum() == 0  # never-accept at the top


def test_trust_or_escalate_routes_hard_tasks_up_and_controls_risk():
    out = run_guaranteed(_pools(), ["c", "T"], CFG, make_cost_fn("tokens", {}))
    toe = out["methods"]["toe"][0]
    assert toe["certified_splits"] == 5 and toe["mean_test_risk"] <= 0.1
    assert toe["mean_solved"] > 0.95  # easy half at the cheap tier, hard half at the top
    single_c = out["methods"]["single-c"][0]
    assert single_c.get("mean_solved", 0) <= 0.55  # the cheap model alone answers only easy tasks
