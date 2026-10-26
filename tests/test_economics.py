"""X13 price sweep and break-even model."""

from conftest import cand, pool

from codeladder.economics import run_economics

CFG = {
    "controller": {"n_init": 2, "max_draws": 2, "action": "resample", "permutations": 2},
    "evaluation": {
        "alphas": [0.1],
        "delta": 0.1,
        "n_splits": 6,
        "calib_frac": 0.5,
        "lambda_grid": {"start": 1.0, "stop": 0.0, "step": 0.5},
        "lambda_grid_2d": {"start": 1.0, "stop": 0.0, "step": 0.5},
        "escalation_thresholds": [0.9],
        "abstain_factor": 4.0,
    },
}


def _pools(n=400):
    """Cheap tier right (and agreeing) on half the tasks, failing visibly on the rest; the top
    tier is always right. Equal tokens per draw, so the ladder saves the top tier on half."""
    out = []
    for i in range(n):
        easy = i % 2 == 0
        cheap = [cand("c", j, visible=easy, hidden=easy, sig="A", tokens=10) for j in range(2)]
        top = [cand("T", j, visible=True, hidden=True, sig="A", tokens=10) for j in range(2)]
        out.append([pool("c", cheap), pool("T", top)])
    return out


def test_ladder_pays_above_the_predicted_break_even():
    out = run_economics(_pools(), ["c", "T"], CFG, ratios=[1, 2, 4, 8, 16])
    be = out["break_even"][0]
    assert abs(be["cheap_share"] - 0.5) < 0.05  # the ladder answers the easy half cheaply
    # spends 2 cheap draws per task, saves 2 top draws on half: break-even ratio 2
    assert 1.5 < be["predicted"] < 2.5
    assert be["measured"] in (2, 4)
    r1 = next(x for x in out["rows"] if x["ratio"] == 1)["top"]
    r16 = next(x for x in out["rows"] if x["ratio"] == 16)["top"]
    assert r1["cost_ratio"] >= 0.99  # at equal prices the top model alone is as cheap
    assert r16["cost_ratio"] < 0.7 and abs(r16["solved_ladder"] - r16["solved_baseline"]) < 0.02
    assert be["asymptote"] < 0.7  # tends to 1 - cheap share (plus the cheap tokens' price)
