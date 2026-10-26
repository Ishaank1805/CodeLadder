"""X4 learned repair-or-resample policy."""

from conftest import cand, pool

from codeladder.actions import learn_action_policy, run_actions
from codeladder.controller import Policy, action_keys, run_task
from codeladder.evaluate import make_cost_fn

CFG = {
    "controller": {"n_init": 1, "max_draws": 3, "action": "resample", "permutations": 2},
    "evaluation": {
        "alphas": [0.2],
        "delta": 0.1,
        "n_splits": 5,
        "calib_frac": 0.5,
        "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.25},
        "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.25},
        "abstain_factor": 4.0,
    },
}


def _task_pool(i: int):
    """Fresh samples are wrong; repairing a timeout fixes it, repairing a wrong answer does not."""
    samples = []
    for j in range(4):
        c = cand("t1", j, visible=False, hidden=False, sig=f"s{i}{j}")
        c.error_type = "timeout" if j % 2 == 0 else "wrong_answer"
        c.visible_pass_rate = 0.5
        samples.append(c)
    repairs = {}
    for j, c in enumerate(samples):
        fixed = c.error_type == "timeout"
        r = cand(
            "t1", 10 + j, visible=fixed, hidden=fixed, sig="good" if fixed else f"r{i}{j}", depth=1
        )
        r.sample_idx = j
        repairs[j] = [r]
    return [pool("t1", samples, repairs)]


def test_action_keys_back_off_from_specific_to_tier():
    c = cand("t1", 0, visible=False, hidden=False)
    c.error_type, c.visible_pass_rate = "exception:ValueError", 0.0
    assert action_keys(2, c) == ["2|runtime|0|0", "2|runtime", "2"]


def test_learned_policy_repairs_only_where_it_helps_and_controller_uses_it():
    pools = [_task_pool(i) for i in range(40)]
    table, stats = learn_action_policy(pools, reward="hidden", min_n=10)
    assert table["0|timeout|high|0"] == "repair" and table["0|wrong_answer"] == "resample"
    assert stats["0|timeout"]["repair"] == 1.0 and stats["0|timeout"]["fresh"] == 0.0


def test_fresh_rate_is_matched_to_the_problems_where_repairs_happen():
    # Easy problems: every sample right (nothing to repair). Hard problems: samples wrong,
    # repairs right 30% of the time, other samples right 10% of the time. Against the overall
    # fresh rate (well above 30%) resampling would look better; on the same problems repair wins.
    pools = []
    for i in range(60):
        easy = [cand("t1", j, visible=True, hidden=True, sig="e") for j in range(10)]
        pools.append([pool("t1", easy)])
        hard = [cand("t1", j, visible=j == 0, hidden=j == 0, sig=f"h{j}") for j in range(10)]
        repairs = {}
        for j in range(1, 10):
            ok = (i * 9 + j) % 10 < 3
            r = cand("t1", 10 + j, visible=ok, hidden=ok, sig="r", depth=1)
            r.sample_idx = j
            repairs[j] = [r]
        pools.append([pool("t1", hard, repairs)])
    table, stats = learn_action_policy(pools, reward="hidden", min_n=10)
    assert stats["0"]["fresh"] < 0.15 < 0.25 < stats["0"]["repair"]
    assert table["0"] == "repair"
    pol = Policy([0.0], n_init=1, max_draws=3, action="adaptive", action_table=table)
    accepted = [run_task(p, pol, perm_seed=k).accepted for p in pools[:10] for k in range(2)]
    assert sum(a is not None and a.hidden_pass for a in accepted) >= 10


def test_run_actions_compares_variants_on_the_same_splits():
    pools = [_task_pool(i) for i in range(200)]
    cost_fn = make_cost_fn("weighted_tokens", {"t1": 1.0})
    out = run_actions(pools, ["t1"], CFG, cost_fn, seeds=1, family="singles")
    m = out["mean"]
    assert set(m) == {"resample", "repair", "label_free_table", "learned_visible", "learned_hidden"}
    assert m["learned_hidden"][0]["mean_solved"] > m["resample"][0]["mean_solved"]
    assert m["resample"][0]["mean_solved"] == 0.0  # fresh samples are never right
