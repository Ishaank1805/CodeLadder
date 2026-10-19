"""Stdin probes, topping up pools (--extend) and truncated pools for ablations."""

from conftest import FakeClient
from test_pipeline import EXEC, GEN, PROBES, SOLUTIONS, toy_tasks

from codeladder.generate import build_pool, truncate_pool
from codeladder.llm import Completion
from codeladder.probes import _is_int, make_stdin_probes

INPUTS = [
    "8 3\napzbqrcs\n1 2 3 1 2 2 1 2\n",  # string + colour array
    "8\n.......#\n.####..#\n.##....#\n.#######\n",  # grid
    "5\n00011\n3 9 2 6 4\n",  # binary string + costs
    "5\n2 1 3 4 5\n",  # permutation
    "4\n1 3 3 7\n",  # sorted array
]


def _same_shape(probe, orig):
    """Same lines and token counts; strings keep their length (a shuffled "00011" may look
    like a number, so kinds come from the original)."""
    pl, ol = [ln.split() for ln in probe.split("\n")], [ln.split() for ln in orig.split("\n")]
    return [len(t) for t in pl] == [len(t) for t in ol] and all(
        _is_int(o) or len(p) == len(o)
        for pt, ot in zip(pl, ol, strict=True)
        for p, o in zip(pt, ot, strict=True)
    )


def test_stdin_probes_keep_the_input_format():
    for x in INPUTS:
        probes = make_stdin_probes([x], 20, seed=0)
        assert probes and x not in probes and len(set(probes)) == len(probes)
        assert probes == make_stdin_probes([x], 20, seed=0)  # deterministic
        for p in probes:
            assert p.split("\n")[0] == x.split("\n")[0]  # sizes and counts untouched
            assert _same_shape(p, x)
    for p in make_stdin_probes([INPUTS[1]], 20):
        assert set(p.split("\n", 1)[1]) <= set(".#\n")  # grids stay grids
    for p in make_stdin_probes([INPUTS[2]], 20):
        assert set(p.split("\n")[1]) <= {"0", "1"}  # binary strings stay binary
    for p in make_stdin_probes([INPUTS[3]], 20):
        assert sorted(p.split("\n")[1].split()) == ["1", "2", "3", "4", "5"]  # still a permutation
    for p in make_stdin_probes([INPUTS[4]], 20):
        vals = [int(v) for v in p.split("\n")[1].split()]
        assert vals == sorted(vals)  # sorted input stays sorted


def test_one_line_inputs_are_data():
    for x in ["7\n", "3 5\n", "1 0 1 1 0 0 1 0\n", "-4 0\n"]:
        probes = make_stdin_probes([x], 20, seed=0)
        assert len(probes) >= 5
        for p in probes:
            assert _same_shape(p, x)
            if len(x.split()) < 3:  # scalars: negatives stay negative, the rest non-negative
                for v, w in zip(x.split(), p.split(), strict=True):
                    assert (int(v) < 0) == (int(w) < 0) and (int(v) <= 0 or int(w) > 0)
    for p in make_stdin_probes(["1 0 1 1 0 0 1 0\n"], 20):
        assert set(p.split()) <= {"0", "1"}  # bit arrays stay bits


def test_stdin_probes_need_something_to_mutate():
    assert make_stdin_probes(["3\n5\n5\n"], 20) == []  # counts and identical scalar lines
    assert make_stdin_probes([], 20) == []


class SlowRepairClient(FakeClient):
    """Repairs only succeed at round 2, so chains have more than one round."""

    def complete(self, messages, *, temperature, top_p, max_tokens, seed) -> Completion:
        comp = super().complete(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
        )
        if len(messages) > 2 and seed % 100 < 2:
            entry = next(e for e in self.solutions if f"`{e}`" in messages[1]["content"])
            return Completion(
                text=f"```python\n{self.solutions[entry][1]}```",
                tokens_in=50,
                tokens_out=40,
                latency_s=0.01,
            )
        return comp


def _ids(pool):
    cands = pool.samples + [c for ch in pool.repairs.values() for c in ch]
    return sorted((c.cand_id, c.code) for c in cands)


def test_extend_tops_up_to_the_same_pool_as_a_fresh_run():
    task = next(t for t in toy_tasks() if t.entry_point == "add")
    small = dict(GEN, n_samples=4, repair={"parents": 4, "depth": 1})
    big = dict(GEN, n_samples=6, repair={"parents": 6, "depth": 3})
    fresh = build_pool(task, "t1", SlowRepairClient(SOLUTIONS), big, PROBES, EXEC)
    assert any(len(ch) == 2 for ch in fresh.repairs.values())  # needed the second round
    first = build_pool(task, "t1", SlowRepairClient(SOLUTIONS), small, PROBES, EXEC)
    client = SlowRepairClient(SOLUTIONS)
    topped = build_pool(task, "t1", client, big, PROBES, EXEC, existing=first)
    assert _ids(topped) == _ids(fresh)
    assert client.calls == len(_ids(fresh)) - len(_ids(first))  # only the new candidates
    again = SlowRepairClient(SOLUTIONS)
    assert _ids(build_pool(task, "t1", again, big, PROBES, EXEC, existing=topped)) == _ids(fresh)
    assert again.calls == 0  # nothing left to do


def test_truncate_pool_matches_generating_with_smaller_settings():
    task = next(t for t in toy_tasks() if t.entry_point == "add")
    small = dict(GEN, n_samples=4, repair={"parents": 4, "depth": 1})
    big = dict(GEN, n_samples=6, repair={"parents": 6, "depth": 3})
    full = build_pool(task, "t1", SlowRepairClient(SOLUTIONS), big, PROBES, EXEC)
    direct = build_pool(task, "t1", SlowRepairClient(SOLUTIONS), small, PROBES, EXEC)
    assert _ids(truncate_pool(full, max_samples=4, max_depth=1)) == _ids(direct)
    assert _ids(truncate_pool(full)) == _ids(full)
