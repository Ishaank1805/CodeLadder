"""X9 learned router: features, routing rule, routed replay and the calibrated comparison."""

import numpy as np
from conftest import cand, pool

from codeladder.evaluate import family_policies, make_cost_fn, replay
from codeladder.router import (
    Router,
    auroc,
    max_constraint,
    meta_features,
    route,
    routed_replay,
    run_router,
    text_counts,
)
from codeladder.types import Task

CFG = {
    "controller": {"n_init": 2, "max_draws": 4, "action": "resample", "permutations": 2},
    "evaluation": {
        "alphas": [0.1],
        "delta": 0.1,
        "n_splits": 5,
        "calib_frac": 0.5,
        "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
        "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.25},
        "escalation_thresholds": [0.9, 0.5],
        "abstain_factor": 4.0,
    },
}


def _task(i: int, easy: bool) -> Task:
    word = "sum two numbers" if easy else "shortest path in a weighted graph with queries"
    return Task(f"x/{i}", "livecodebench", f"Problem {i}: {word}.", "", "", [[1]], [[1]])


def _data(n: int = 120):
    tasks, pools = [], []
    for i in range(n):
        easy = i % 2 == 0
        cheap = [cand("t1", j, visible=easy, hidden=easy, sig="A") for j in range(4)]
        big = [cand("t2", j, visible=True, hidden=True, sig="A", tokens=100) for j in range(4)]
        tasks.append(_task(i, easy))
        pools.append([pool("t1", cheap), pool("t2", big)])
    return tasks, pools


def test_features_and_constraints():
    assert max_constraint("1 <= N <= 2 \\times 10^5 and A_i <= 10^{9}") == 9.0
    assert abs(max_constraint("n up to 200000") - np.log10(200000)) < 1e-9
    tasks, _ = _data(4)
    assert text_counts([t.prompt for t in tasks]).shape == (4, 1024)
    m = meta_features(tasks)
    assert m.shape[0] == 4 and m[0, 5] == 1.0  # livecodebench one-hot


def test_route_picks_cheapest_confident_tier():
    pred = np.array([[0.9, 0.95], [0.2, 0.8], [0.1, 0.3]])
    assert route(pred, 0.5).tolist() == [0, 1, 1]
    assert route(pred, 1.01).tolist() == [1, 1, 1]
    assert route(pred, 0.0).tolist() == [0, 0, 0]
    assert auroc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0


def test_router_learns_and_routed_replay_matches_tier_replays():
    tasks, pools = _data()
    y = np.array([[np.mean([c.hidden_pass for c in p.samples]) for p in ps] for ps in pools])
    counts, meta = text_counts([t.prompt for t in tasks]), meta_features(tasks)
    pred = Router(l2_grid=(1.0,)).fit(counts[:60], meta[:60], y[:60]).predict(counts, meta)
    assert auroc(pred[60:, 0], y[60:, 0]) > 0.95
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.0, "t2": 10.0})
    _, sp, _, paths = family_policies(
        "singles", ["t1", "t2"], CFG["controller"], CFG["evaluation"], {}
    )
    single = replay(pools, sp, 2, cost_fn)
    rep, rpaths, tiers = routed_replay(single, paths, pred, [1.01, 0.5])
    K = len(paths[0])
    assert rep.loss.shape == (2 * K, len(pools), 2) and rpaths[1] == list(range(K, 2 * K))
    assert (tiers[0] == 1).all()  # tau above 1: always the last tier
    np.testing.assert_array_equal(rep.cost[:K], single.cost[paths[1]])
    for i in range(len(pools)):  # tau 0.5: each task replays its routed tier
        np.testing.assert_array_equal(rep.cost[K:, i], single.cost[paths[tiers[1, i]], i])


def test_run_router_certifies_and_routes_easy_tasks_cheaply():
    tasks, pools = _data(300)
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.0, "t2": 10.0})
    out = run_router(pools, tasks, ["t1", "t2"], CFG, cost_fn, router_seeds=1)
    mean = out["mean"]
    assert set(mean) == {"singles", "router", "oracle_router"}
    r, s = mean["router"][0], mean["singles"][0]
    assert r["certified_splits"] > 0 and r["mean_test_risk"] <= 0.1
    assert r["mean_test_cost"] < s["mean_test_cost"]  # easy half goes to the cheap tier
    assert out["seeds"][0]["families"]["router"][0]["routed_share"]["t1"] > 0.3
