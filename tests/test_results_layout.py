"""The results folder layout: names map to groups, and lookups accept names or old paths."""

from codeladder.results import find_result, group_of, iter_runs, result_path


def test_groups_by_kind_and_benchmark():
    assert (
        str(group_of("humaneval+mbpp-t1-t2-resample-agreement_selfconf-singles-k64"))
        == "runs/humaneval+mbpp"
    )
    assert str(group_of("livecodebench-t4-resample-agreement-shared-k64")) == "runs/livecodebench"
    assert str(group_of("route-livecodebench-t1-t2-agreement_selfconf-k64")) == "router"
    assert str(group_of("economics-bigcodebench-t1-t4-agreement_ptrue-k64-lb1")) == "economics"
    assert str(group_of("baselines-humaneval+mbpp-t1-t4.json")) == "baselines"


def test_find_result_accepts_names_and_old_paths(tmp_path):
    name = "livecodebench-t1-resample-agreement-shared-k64"
    d = result_path(tmp_path, name)
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    assert find_result(tmp_path, name) == d
    assert find_result(tmp_path, f"results/{name}") == d  # a path from before the layout
    old = tmp_path / "flat-run"  # a result not yet moved is still found
    old.mkdir()
    assert find_result(tmp_path, "flat-run") == old
    (tmp_path / "archive" / "x").mkdir(parents=True)
    (tmp_path / "archive" / "x" / "summary.json").write_text("{}")
    assert list(iter_runs(tmp_path)) == [d]  # archive is never read
