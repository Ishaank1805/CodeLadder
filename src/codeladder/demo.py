"""X15 demo: a certified CodeLadder policy in the browser.

    codeladder demo-bundle --out data/demo           # on the cluster: prompts and visible tests
    codeladder demo [--port 8765]                    # anywhere with data/demo and data/pools

A small standard-library web server (no extra packages) serves one page.

* **Build a ladder**: choose a benchmark, the models (Qwen, DeepSeek, mixed, with the API
  tier), the threshold family, the score and the lower-rung budget, then a risk level α; the
  server certifies the policy on the spot from the matching evaluate run in results/ (its
  replay of every setting on every task), as ``export-policy`` does. Saved policies
  (examples/policies, results/policies) remain available.
* **One problem** (a benchmark task, or your own function with tests) in **replay** mode (the
  cached pools, instant) or **live** mode (the policy's models; needs their servers or API
  keys). Every step streams to the page as one JSON line: tier entered, candidate drawn (code,
  checks, ratings), scores against the certified threshold, then accept, resample / repair,
  escalate or abstain, and finally the hidden-test verdict (ground truth, which the controller
  never sees), the cost and the guarantee.
* **Many problems**: replay the policy over a random sample of the benchmark and stream the
  tally (answered, correct, accepted but wrong, abstained, cost, the rung that answered).
"""

from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import REPO_ROOT, load_tiers
from .results import find_result, iter_runs
from .scoring import agreement_key
from .types import Candidate, Task

PAGE = Path(__file__).resolve().parent / "demo_page.html"
BENCHMARKS = ("humaneval", "mbpp", "livecodebench", "bigcodebench")
# Evaluate runs the ladder builder offers: one benchmark set, resample action, the reference baseline's price
# of abstaining (κ = 64), full or lean (one draw) lower rungs.
RUN_NAME = re.compile(
    r"^(?P<bench>[a-z+]+)-(?P<tiers>(?:(?:t\d|d\d|api)-)+)resample-(?P<score>[a-z_]+)"
    r"-(?P<family>shared|singles|subsets|two_level)(?P<tag>.*)$"
)
CATALOG_TAG = re.compile(r"^-k64(?P<lean>-lb1)?(?:-all7)?(?:-agr)?$")
RUN_BENCHES = {
    "humaneval+mbpp": ["humaneval", "mbpp"],
    "livecodebench": ["livecodebench"],
    "bigcodebench": ["bigcodebench"],
    "toy": ["toy"],
}  # fmt: skip  (toy: tests)


# ---------------------------------------------------------------- bundle (run on the cluster)


def make_bundle(out: str | Path, benchmarks: tuple[str, ...] = BENCHMARKS) -> dict[str, int]:
    """Prompts, metadata and visible tests of every task, one file per benchmark (gzipped
    pickle: test inputs hold tuples, sets and complex numbers that JSON would change)."""
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
        (out / f"{bench}.pkl.gz").write_bytes(gzip.compress(pickle.dumps(rows)))
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
        self.results = Path(cfg["paths"]["results"])
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._catalog: list[dict[str, Any]] | None = None
        self._certified: dict[tuple[str, float], dict[str, Any]] = {}
        self._pools: tuple[tuple[str, tuple[str, ...]], dict[str, Any]] | None = None
        self._live: tuple[float, dict[str, Any]] | None = None

    # -- data
    def tasks(self, bench: str) -> dict[str, Any]:
        with self._lock:
            if bench not in self._tasks:
                f = self.bundle / f"{bench}.pkl.gz"
                # Our own bundle (demo-bundle), so unpickling it is safe.
                self._tasks[bench] = (
                    pickle.loads(gzip.decompress(f.read_bytes())) if f.exists() else {}
                )
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
                    f.name.removesuffix(".pkl.gz") for f in self.bundle.glob("*.pkl.gz")
                )
                out.append(meta | {"file": str(f), "id": f.stem, "benchmarks": benches})
        return out

    def catalog(self) -> list[dict[str, Any]]:
        """Evaluate runs a ladder can be certified from (see RUN_NAME and CATALOG_TAG)."""
        if self._catalog is not None:
            return self._catalog
        seen, out = set(), []
        for d in iter_runs(self.results):
            if not (d / "replay.npz").exists():
                continue
            m = RUN_NAME.match(d.name)
            t = CATALOG_TAG.match(m.group("tag")) if m else None
            if not m or not t or m.group("bench") not in RUN_BENCHES:
                continue
            tiers = m.group("tiers").strip("-").split("-")
            family = m.group("family") if len(tiers) > 1 else "single"
            lean = bool(t.group("lean")) and family in ("subsets", "two_level")
            key = (m.group("bench"), tuple(tiers), family, m.group("score"), lean)
            if key in seen:  # e.g. a -agr duplicate of the same setting
                continue
            seen.add(key)
            s = json.loads((d / "summary.json").read_text())
            split = {
                f"{c['alpha']:g}": {
                    "solved": c.get("mean_solved"), "cost": c.get("mean_test_cost"),
                    "risk": c.get("mean_test_risk"), "coverage": c.get("mean_coverage"),
                    "violation": c.get("violation_rate"), "certified": c.get("certified_splits"),
                    "splits": c.get("n_splits"),
                }
                for c in s.get("calibrated", []) if c.get("certified_splits")
            }  # fmt: skip
            out.append({
                "run": d.name, "bench": m.group("bench"), "benchmarks": RUN_BENCHES[m.group("bench")],
                "tiers": tiers, "family": family, "score": m.group("score"), "lean": lean,
                "n_tasks": s.get("n_tasks"), "split": split,
            })  # fmt: skip
        self._catalog = out
        return out

    def certify(self, run: str, alpha: float) -> dict[str, Any]:
        """Certify the run's family at alpha on all its tasks (as export-policy does)."""
        from .live import export_policy

        alpha = round(float(alpha), 4)
        key = (run, alpha)
        if key not in self._certified:
            entry = next((r for r in self.catalog() if r["run"] == run), None)
            if entry is None:
                raise ValueError(f"unknown run {run}")
            meta = export_policy(find_result(self.results, run), self.cfg, alpha)
            self._certified[key] = meta | {"id": f"run:{run}@{alpha}",
                                           "benchmarks": entry["benchmarks"]}  # fmt: skip
        return self._certified[key]

    def policy(self, pid: str) -> tuple[Any, dict[str, Any]]:
        from .live import load_policy, policy_from_meta

        if pid.startswith("run:"):
            run, alpha = pid[4:].rsplit("@", 1)
            meta = self.certify(run, float(alpha))
            return policy_from_meta(meta), meta
        meta = next(p for p in self.policies() if p["id"] == pid)
        return load_policy(meta["file"])[0], meta

    def live_status(self) -> dict[str, Any]:
        """Which tiers live mode can call from here: an API tier needs a key (only its presence
        is checked), a local tier a running server (scripts/ada/serve_vllm.sbatch writes its
        address; it must answer within a second). Cached for 30 seconds."""
        import urllib.request

        if self._live and time.time() - self._live[0] < 30:
            return self._live[1]
        out = {}
        for name, t in self.tiers.items():
            if t.api_key_env:
                try:
                    ok, why = bool(t.api_keys()), "API key found"
                except RuntimeError:
                    ok, why = False, "no API key here"
            else:
                try:
                    url = t.resolve_base_url()
                    with urllib.request.urlopen(url + "/models", timeout=1):
                        ok, why = True, f"server at {url}"
                except FileNotFoundError:
                    ok, why = False, "no server running"
                except Exception:  # noqa: BLE001 (unreachable, stale address)
                    ok, why = False, "server not reachable from here"
            out[name] = {"ok": ok, "why": why}
        self._live = (time.time(), out)
        return out

    def task_pools(self, bench: str, tiers: list[str], task_id: str) -> list[Any] | None:
        """One task's pools, read directly (no need to load the whole benchmark)."""
        from .types import TierPool

        root = Path(self.cfg["paths"]["pools"]) / bench
        out = []
        for t in tiers:
            f = root / t / f"{task_id.replace('/', '__')}.json"
            if not f.exists():
                return None
            out.append(TierPool.from_dict(json.loads(f.read_text())))
        return out

    def bench_pools(self, bench: str, tiers: list[str]) -> dict[str, Any]:
        """Every task's pools for one benchmark and tier list (the last one is kept in memory)."""
        from .generate import load_pools

        key = (bench, tuple(tiers))
        with self._lock:
            if self._pools is None or self._pools[0] != key:
                self._pools = (key, load_pools(self.cfg["paths"]["pools"], bench, tiers))
            return self._pools[1]

    def task_list(self, bench: str, tiers: list[str]) -> list[dict[str, Any]]:
        pools = Path(self.cfg["paths"]["pools"]) / bench
        have = None
        for t in tiers:
            ids = {f.stem.replace("__", "/") for f in (pools / t).glob("*.json")}
            have = ids if have is None else have & ids
        rows = self.tasks(bench)
        return [
            {"id": tid, "difficulty": r.get("difficulty"), "replay": tid in (have or set()),
             "mode": r.get("mode")}
            for tid, r in rows.items()
        ]  # fmt: skip

    # -- runs
    def cost_fn(self, score: str):
        from .evaluate import make_cost_fn

        return make_cost_fn(
            "weighted_tokens", {n: t.params_b for n, t in self.tiers.items()}, score
        )

    def run(self, req: dict[str, Any], send) -> None:
        from .live import custom_task, replay, solve
        from .scoring import checks_ok, phi_scores

        policy, meta = self.policy(req["policy"])
        tiers = meta["tiers"]
        cost_fn = self.cost_fn(policy.score)
        drawn: dict[int, list[Candidate]] = {}
        clusters: dict[str, str] = {}
        t0 = time.time()

        def cand_view(c: Candidate, t: int) -> dict[str, Any]:
            sig = agreement_key(c)  # the clusters the score counts
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
            with tempfile.TemporaryDirectory(prefix="codeladder-demo-") as d:  # one per request
                tmp = Path(d) / "problem.json"
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
            pools = self.task_pools(bench, tiers, task.task_id)
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

    def batch(self, req: dict[str, Any], send) -> None:
        """Replay the policy over a random sample of tasks, one line per task, then a total."""
        from .live import replay

        policy, meta = self.policy(req["policy"])
        tiers = meta["tiers"]
        cost_fn = self.cost_fn(policy.score)
        benches = req.get("benchmarks") or meta.get("benchmarks") or []
        items = []
        for b in benches:
            send({"type": "loading", "benchmark": b})
            pools = self.bench_pools(b, tiers)
            items += [(b, tid, p) for tid, p in pools.items()]
        rng = random.Random(int(req.get("seed", 0)))
        rng.shuffle(items)
        n = int(req.get("n") or 0)
        items = items[:n] if n > 0 else items
        send({"type": "batch_start", "n": len(items), "tiers": tiers, "alpha": meta["alpha"],
              "thresholds": policy.thresholds})  # fmt: skip
        perm = int(req.get("perm", 0))
        for b, tid, pools in items:
            o = replay(pools, policy, perm)
            acc = o["accepted"]
            send({
                "type": "task", "benchmark": b, "id": tid, "tier": o["tier"],
                "accepted": acc is not None,
                "hidden_pass": bool(acc.hidden_pass) if acc is not None else None,
                "cost": round(sum(cost_fn(c) for c in o["drawn"]), 1), "drawn": len(o["drawn"]),
                "difficulty": self.tasks(b).get(tid, {}).get("difficulty"),
            })  # fmt: skip
        send({"type": "batch_done"})


def _feedback_line(c: Candidate) -> str:
    """One readable line about a visible-test failure (values as Python, not our encoding)."""
    from .prompts import _show

    if c.visible_pass_rate >= 1.0:
        return ""
    f = c.feedback or {}
    kind = f.get("kind", c.error_type)

    def show(v: Any) -> str:
        try:
            text = _show(v) if isinstance(v, list) and v and isinstance(v[0], str) else str(v)
        except Exception:  # noqa: BLE001
            text = str(v)
        return text if len(text) <= 90 else text[:90] + "..."

    if kind == "wrong_answer":
        return f"wrong answer on a visible test: expected {show(f.get('expected'))}, got {show(f.get('got'))}"
    if kind == "timeout":
        return "too slow on a visible test"
    if kind == "exception":
        return f"raises {f.get('error', 'an error')} on a visible test: {str(f.get('detail', ''))[:140]}"
    if kind == "test_failure":
        detail = str(f.get("detail", "")).strip().splitlines()
        return "fails the visible unit test" + (f": {detail[-1][:160]}" if detail else "")
    return f"{kind.replace('exception:', 'raises ')}: {str(f.get('detail', ''))[:160]}"


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
                elif url.path == "/api/live":
                    self._json(app.live_status())
                elif url.path == "/api/catalog":
                    self._json(app.catalog())
                elif url.path == "/api/certify":
                    try:
                        self._json(app.certify(q["run"], float(q["alpha"])))
                    except ValueError as e:  # nothing certified at this alpha
                        self._json({"error": str(e)})
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
            path = urlparse(self.path).path
            if path not in ("/api/run", "/api/batch"):
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
                (app.batch if path == "/api/batch" else app.run)(req, send)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page stopped reading (a new run started)
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
