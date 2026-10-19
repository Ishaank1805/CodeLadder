import copy

from conftest import cand, pool

from codeladder.controller import Policy, error_class, learn_action_table, run_task
from codeladder.scoring import phi_scores


def test_phi_is_agreement_among_visible_passing():
    cs = [
        cand("t1", 0, visible=True, hidden=True, sig="A"),
        cand("t1", 1, visible=True, hidden=True, sig="A"),
        cand("t1", 2, visible=True, hidden=False, sig="B"),
        cand("t1", 3, visible=False, hidden=False, sig="A"),
    ]
    assert phi_scores(cs) == [0.5, 0.5, 0.25, 0.0]


def test_accepts_at_first_tier_when_samples_agree():
    t1 = pool("t1", [cand("t1", i, visible=True, hidden=True) for i in range(5)])
    t2 = pool("t2", [cand("t2", i, visible=True, hidden=True) for i in range(5)])
    out = run_task([t1, t2], Policy([0.9, 0.9], n_init=3, max_draws=5))
    assert out.tier_index == 0 and out.accepted is not None
    assert len(out.drawn) == 3


def test_escalates_then_abstains():
    bad = [cand("t", i, visible=False, hidden=False) for i in range(5)]
    out = run_task([pool("t1", bad), pool("t2", copy.deepcopy(bad))], Policy([0.5, 0.5], 2, 4))
    assert out.accepted is None and out.tier_index is None
    assert len(out.drawn) == 8  # max_draws at each tier


def test_threshold_controls_acceptance():
    mixed = [
        cand("t1", 0, visible=True, hidden=True, sig="A"),
        cand("t1", 1, visible=True, hidden=False, sig="B"),
        cand("t1", 2, visible=True, hidden=False, sig="C"),
    ]
    strict = run_task([pool("t1", mixed)], Policy([0.9], 3, 3))
    loose = run_task([pool("t1", mixed)], Policy([0.3], 3, 3))
    assert strict.accepted is None
    assert loose.accepted is not None


def test_controller_never_uses_hidden_labels():
    samples = [
        cand("t1", i, visible=i % 2 == 0, hidden=i % 3 == 0, sig=str(i % 2)) for i in range(8)
    ]
    flipped = copy.deepcopy(samples)
    for c in flipped:
        c.hidden_pass = not c.hidden_pass
    pol = Policy([0.4], 3, 6)
    for seed in range(5):
        a, b = (
            run_task([pool("t1", samples)], pol, seed),
            run_task([pool("t1", flipped)], pol, seed),
        )
        assert [c.cand_id for c in a.drawn] == [c.cand_id for c in b.drawn]


def test_repair_draws_from_cached_chain():
    samples = [
        cand("t1", 0, visible=False, hidden=False),
        cand("t1", 1, visible=False, hidden=False),
    ]
    chain = {
        0: [cand("t1", 0, visible=True, hidden=True, depth=1)],
        1: [cand("t1", 1, visible=True, hidden=True, depth=1)],
    }
    out = run_task(
        [pool("t1", samples, chain)], Policy([0.3], n_init=2, max_draws=3, action="repair")
    )
    assert out.accepted is not None and out.accepted.kind == "repair"
    assert out.repair_fallbacks == 0


def test_repair_falls_back_to_resample_without_chain():
    samples = [cand("t1", i, visible=False, hidden=False) for i in range(3)]
    out = run_task([pool("t1", samples)], Policy([0.3], n_init=1, max_draws=3, action="repair"))
    assert out.repair_fallbacks == 2
    assert len(out.drawn) == 3


def test_error_class_and_action_table():
    assert error_class("exception:TypeError") == "runtime"
    assert error_class("syntax_error") == "broken"
    samples = [cand("t1", i, visible=False, hidden=False) for i in range(4)]
    chain = {i: [cand("t1", i, visible=True, hidden=True, depth=1)] for i in range(4)}
    table = learn_action_table([[pool("t1", samples, chain)]])
    assert table["0|wrong_answer"] == "repair"  # repairs always fix; fresh samples never pass
