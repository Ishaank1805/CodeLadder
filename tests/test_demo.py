"""X15 demo: bundle, streamed replay events and the HTTP endpoints (toy tasks, fake model)."""

import copy
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from conftest import FakeClient
from test_live import _cfg
from test_pipeline import SOLUTIONS, toy_tasks

from codeladder.config import REPO_ROOT
from codeladder.demo import Demo, make_bundle, make_handler
from codeladder.evaluate import make_cost_fn, run_experiment
from codeladder.generate import generate_pools, load_pools
from codeladder.live import export_policy


def _setup(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["paths"]["llm_cache"] = str(tmp_path / "cache")
    tasks = toy_tasks()
    generate_pools(tasks, "t1", FakeClient(SOLUTIONS, bug_every=2), cfg, workers=2)
    generate_pools(tasks, "t2", FakeClient(SOLUTIONS, bug_every=5), cfg, workers=2)
    pools = load_pools(cfg["paths"]["pools"], "toy", ["t1", "t2"])
    run = tmp_path / "results" / "toy-t1-t2-resample-agreement-subsets"
    run_experiment([copy.deepcopy(p) for p in pools.values() for _ in range(10)], ["t1", "t2"], cfg,
                   make_cost_fn("weighted_tokens", {}), run)  # fmt: skip
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "toy.json").write_text(json.dumps(export_policy(run, cfg, alpha=0.3)))
    assert make_bundle(tmp_path / "demo", ("toy",)) == {"toy": 3}
    return Demo(
        cfg, str(REPO_ROOT / "configs" / "models.yaml"), tmp_path / "demo", [tmp_path / "policies"]
    )


def test_replay_streams_every_step(tmp_path):
    app = _setup(tmp_path)
    assert [p["id"] for p in app.policies()] == ["toy"]
    assert {t["id"] for t in app.task_list("toy", ["t1", "t2"]) if t["replay"]} == {
        "toy/add",
        "toy/remove_duplicates",
        "toy/is_prime",
    }
    events = []
    app.run(
        {"policy": "toy", "benchmark": "toy", "task_id": "toy/add", "mode": "replay"}, events.append
    )
    json.dumps(events)  # everything the page receives is plain JSON
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[1] == "problem" and types[-1] == "result"
    assert "draw" in types and "scores" in types and ("accept" in types or "abstain" in types)
    draws = [e for e in events if e["type"] == "draw"]
    assert all(d["cand"]["code"] and "checks_ok" in d["cand"] for d in draws)
    result = events[-1]
    assert result["drawn"] == len(draws) and result["guarantee"]


def test_http_endpoints(tmp_path):
    app = _setup(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"CodeLadder" in urllib.request.urlopen(base + "/").read()
        assert json.load(urllib.request.urlopen(base + "/api/policies"))[0]["id"] == "toy"
        task = json.load(urllib.request.urlopen(base + "/api/task?benchmark=toy&id=toy/is_prime"))
        assert "def is_prime" in task["prompt"]
        req = urllib.request.Request(
            base + "/api/run",
            data=json.dumps(
                {"policy": "toy", "benchmark": "toy", "task_id": "toy/is_prime"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        lines = [json.loads(ln) for ln in urllib.request.urlopen(req).read().decode().splitlines()]
        assert lines[0]["type"] == "start" and lines[-1]["type"] == "result"
    finally:
        server.shutdown()
