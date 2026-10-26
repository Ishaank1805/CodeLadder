"""X15 demo: a certified CodeLadder policy in the browser.

    codeladder demo-bundle --out data/demo           # on the cluster: prompts and visible tests
    codeladder demo [--port 8765]                    # anywhere with data/demo and data/pools

A small standard-library web server (no extra packages) serves one page. Pick a certified
policy (``export-policy`` output in examples/policies or results/policies), a problem (any
benchmark task in the bundle, or your own function with tests) and a mode: **replay** runs
the controller on the cached pools (instant, no model), **live** asks the policy's models for
new candidates (needs their servers or API keys). Every step streams to the page as one JSON
line: tier entered, candidate drawn (code, checks, ratings), scores against the certified
threshold, then accept, resample / repair, escalate or abstain, and finally the hidden-test
verdict (ground truth, which the controller never sees), the cost and the guarantee.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import REPO_ROOT, load_tiers
from .types import Candidate, Task

PAGE = Path(__file__).resolve().parent / "demo_page.html"
BENCHMARKS = ("humaneval", "mbpp", "livecodebench", "bigcodebench")


# ---------------------------------------------------------------- bundle (run on the cluster)


def make_bundle(out: str | Path, benchmarks: tuple[str, ...] = BENCHMARKS) -> dict[str, int]:
    """Prompts, metadata and visible tests of every task, one JSON file per benchmark."""
    from .tasks import load_tasks, prepare_task, release_tests

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for bench in benchmarks:
        try:
            tasks = load_tasks(bench)
        except Exception as e:  # noqa: BLE001 (benchmark data not present here)
            print(f"skipping {bench}: {e}")
            continue
        rows = {}
        for t in tasks:
            if bench != "bigcodebench":
                prepare_task(t)
            rows[t.task_id] = {
                "prompt": t.prompt,
                "entry_point": t.entry_point,
                "mode": t.mode,
                "difficulty": t.difficulty,
                "release_date": t.release_date,
                "oracle": t.oracle,
                "atol": t.atol,
                "time_limit_s": t.time_limit_s,
                "visible_inputs": t.visible_inputs,
                "visible_expected": t.visible_expected,
                "tests": t.tests,  # BigCodeBench: unit tests (first method visible)
                "probe_code": t.probe_code,
                "n_hidden": len(t.hidden_inputs),
            }
            if bench == "livecodebench":
                release_tests(t)
        (out / f"{bench}.json").write_text(json.dumps(rows))
        counts[bench] = len(rows)
    return counts


def bundle_task(row: dict[str, Any], task_id: str, benchmark: str) -> Task:
    return Task(
        task_id=task_id, benchmark=benchmark, prompt=row["prompt"], entry_point=row["entry_point"],
        reference="", visible_inputs=row["visible_inputs"], hidden_inputs=[],
        visible_expected=row["visible_expected"], hidden_expected=[], atol=row.get("atol") or 0.0,
        oracle=row.get("oracle"), mode=row["mode"], time_limit_s=row.get("time_limit_s"),
        difficulty=row.get("difficulty"), tests=row.get("tests"), probe_code=row.get("probe_code"),
    )  # fmt: skip


# ---------------------------------------------------------------- server state


class Demo:
    def __init__(self, cfg: dict[str, Any], models: str, bundle: Path, policy_dirs: list[Path]):
        self.cfg, self.models, self.bundle, self.policy_dirs = cfg, models, bundle, policy_dirs
        self.tiers = {t.name: t for t in load_tiers(models)}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- data
    def tasks(self, bench: str) -> dict[str, Any]:
        with self._lock:
            if bench not in self._tasks:
                f = self.bundle / f"{bench}.json"
                self._tasks[bench] = json.loads(f.read_text()) if f.exists() else {}
            return self._tasks[bench]

    def policies(self) -> list[dict[str, Any]]:
        out = []
        for d in self.policy_dirs:
            for f in sorted(d.glob("*.json")) if d.exists() else []:
                try:
                    meta = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                if "thresholds" not in meta:
                    continue
                run = meta.get("source_run", "")
                benches = [b for b in BENCHMARKS if b in run.split("-")[0].split("+")] or sorted(
                    f.stem for f in self.bundle.glob("*.json") if not f.stem.startswith("_")
                )
                out.append(meta | {"file": str(f), "id": f.stem, "benchmarks": benches})
        return out

    def policy(self, pid: str) -> tuple[Any, dict[str, Any]]:
        from .live import load_policy

        meta = next(p for p in self.policies() if p["id"] == pid)
        return load_policy(meta["file"])[0], meta

    def task_list(self, bench: str, tiers: list[str]) -> list[dict[str, Any]]:
        pools = Path(self.cfg["paths"]["pools"]) / bench
        have = None
        for t in tiers:
            ids = {f.stem.replace("__", "/") for f in (pools / t).glob("*.json")}
            have = ids if have is None else have & ids
        rows = self.tasks(bench)
        return [
            {"id": tid, "difficulty": r.get("difficulty"), "replay": tid in (have or set())}
            for tid, r in rows.items()
        ]

    # -- runs
    def cost_fn(self, score: str):
        from .evaluate import make_cost_fn

        return make_cost_fn(
            "weighted_tokens", {n: t.params_b for n, t in self.tiers.items()}, score
        )

    def run(self, req: dict[str, Any], send) -> None:
        from .generate import load_pools
        from .live import custom_task, replay, solve
        from .scoring import checks_ok, phi_scores

        policy, meta = self.policy(req["policy"])
        tiers = meta["tiers"]
        cost_fn = self.cost_fn(policy.score)
        drawn: dict[int, list[Candidate]] = {}
        clusters: dict[str, str] = {}
        t0 = time.time()

        def cand_view(c: Candidate, t: int) -> dict[str, Any]:
            sig = c.extra.get("stress_sig") or c.signature
            if checks_ok(c) and sig not in clusters:
                clusters[sig] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(clusters) % 26]
            return {
                "id": c.cand_id, "tier": tiers[t], "kind": c.kind, "sample": c.sample_idx,
                "depth": c.depth, "code": c.code, "visible": c.visible_pass_rate,
                "checks_ok": checks_ok(c), "error": c.error_type,
                "feedback": _feedback_line(c), "cluster": clusters.get(sig) if checks_ok(c) else None,
                "self_conf": c.extra.get("self_conf"), "p_true": c.extra.get("p_true"),
                "tokens": c.tokens_in + c.tokens_out, "cost": round(cost_fn(c), 1),
            }  # fmt: skip

        def on_event(e: dict[str, Any]) -> None:
            e = dict(e)
            t = e.get("tier")
            if e["type"] == "draw":
                drawn.setdefault(t, []).append(e["cand"])
                e["cand"] = cand_view(e["cand"], t)
            elif e["type"] == "accept":
                e["cand"] = e["cand"].cand_id
            elif e["type"] == "scores":
                ds = drawn.get(t, [])
                e["agreement"] = [round(x, 3) for x in phi_scores(ds, "agreement")]
                e["scores"] = [round(x, 3) for x in e["scores"]]
                e["ids"] = [c.cand_id for c in ds]
            if t is not None:
                e["tier_name"] = tiers[t]
            send(e)

        send({"type": "start", "tiers": tiers, "thresholds": policy.thresholds, "score": policy.score,
              "alpha": meta["alpha"], "setting": meta.get("setting"), "mode": req.get("mode", "replay")})  # fmt: skip
        if req.get("problem"):
            tmp = Path(self.bundle) / "_custom_problem.json"
            tmp.write_text(json.dumps(req["problem"]))
            task = custom_task(tmp)
            bench = "custom"
        else:
            bench, tid = req["benchmark"], req["task_id"]
            task = bundle_task(self.tasks(bench)[tid], tid, bench)
        send({"type": "problem", "id": task.task_id, "prompt": task.prompt, "mode": task.mode})
        if req.get("mode") == "live":
            from .llm import OpenAIChatClient, ResponseCache

            cache = Path(self.cfg["paths"]["llm_cache"])
            clients = {
                n: OpenAIChatClient(self.tiers[n], ResponseCache(cache / f"live-{n}.sqlite"))
                for n, th in zip(tiers, policy.thresholds, strict=True)
                if th is not None
            }
            out = solve(
                task, policy, tiers, clients, self.cfg, log=lambda m: None, on_event=on_event
            )
        else:
            pools = load_pools(self.cfg["paths"]["pools"], bench, tiers).get(task.task_id)
            if pools is None:
                send({"type": "error", "message": f"no cached pools of {task.task_id} for {tiers}"})
                return
            out = replay(pools, policy, int(req.get("perm", 0)), on_event=on_event)
        acc = out["accepted"]
        hidden = None
        if acc is not None and bench != "custom":
            hidden = (
                bool(acc.hidden_pass) if req.get("mode") != "live" or task.mode == "bcb" else None
            )
        send({
            "type": "result",
            "accepted": acc.cand_id if acc is not None else None,
            "tier": out["tier"],
            "code": acc.code if acc is not None else None,
            "hidden_pass": hidden,
            "drawn": len(out["drawn"]),
            "tokens": sum(c.tokens_in + c.tokens_out for c in out["drawn"]),
            "cost": round(sum(cost_fn(c) for c in out["drawn"]), 1),
            "calibration": meta.get("calibration"),
            "guarantee": meta.get("guarantee"),
            "seconds": round(time.time() - t0, 2),
        })  # fmt: skip


def _feedback_line(c: Candidate) -> str:
    if c.visible_pass_rate >= 1.0:
        return ""
    f = c.feedback or {}
    kind = f.get("kind", c.error_type)
    if kind == "wrong_answer":
        return f"wrong answer on a visible test (expected {str(f.get('expected'))[:80]}, got {str(f.get('got'))[:80]})"
    if kind == "test_failure":
        return (
            "fails the visible unit test: "
            + str(f.get("detail", "")).strip().splitlines()[-1][:160]
            if f.get("detail")
            else "fails the visible unit test"
        )
    return f"{kind}: {str(f.get('detail', ''))[:160]}"


# ---------------------------------------------------------------- HTTP


def make_handler(app: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/policies":
                    self._json(app.policies())
                elif url.path == "/api/tasks":
                    self._json(app.task_list(q["benchmark"], q["tiers"].split(",")))
                elif url.path == "/api/task":
                    row = app.tasks(q["benchmark"]).get(q["id"], {})
                    self._json({"prompt": row.get("prompt", ""), "mode": row.get("mode")})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/run":
                return self._json({"error": "not found"}, 404)
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(obj: dict[str, Any]) -> None:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()

            try:
                app.run(req, send)
            except Exception as e:  # noqa: BLE001
                send({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Handler


def serve(cfg: dict[str, Any], models: str, port: int = 8765, host: str = "127.0.0.1",
          bundle: Path | None = None) -> None:  # fmt: skip
    app = Demo(cfg, models, bundle or REPO_ROOT / "data" / "demo",
               [REPO_ROOT / "examples" / "policies", Path(cfg["paths"]["results"]) / "policies"])  # fmt: skip
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"CodeLadder demo on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
