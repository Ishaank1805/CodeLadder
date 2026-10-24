"""X17: how much does a certificate depend on the strength of the test oracle?

Pools label a candidate correct when it passes every hidden test. A practitioner usually has
far fewer tests. Here every distinct visible-passing program is run on *all* hidden tests
(no early stop) and the failing test indices are stored once per (benchmark, tier). A weak
oracle keeps a random fraction f of each task's hidden tests (f = 0: the visible tests only,
i.e. HumanEval's / MBPP's original tests for EvalPlus tasks); a candidate is weakly correct if
it fails none of them. Calibrating on weak labels and measuring the true risk on all hidden
tests shows how far the certificate under-reports.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .controller import run_task
from .evaluate import family_policies, run_splits
from .generate import _limits, _run, hidden_chunks
from .oracles import check
from .tasks import prepare_task, release_tests
from .types import Task, TierPool


def hidden_failures(task: Task, code: str, exec_cfg: dict[str, Any]) -> list[int]:
    """Indices of the hidden tests the program fails (all of them run, in chunks)."""
    expected = task.hidden_expected or []
    limits = _limits(task.hidden_ref_times, len(task.hidden_inputs), exec_cfg, task)
    mem = int(exec_cfg.get("memory_mb_hidden", 4096))
    failing: list[int] = []
    for i, j in hidden_chunks(len(task.hidden_inputs), int(exec_cfg.get("hidden_chunk", 250))):
        ins, exp = task.hidden_inputs[i:j], expected[i:j]
        run = _run(task, code, ins, limits[i:j], exec_cfg, memory_mb=mem)
        results = run.results or []
        for k in range(len(ins)):
            res = results[k] if k < len(results) else None
            ok = (
                not run.fatal
                and res is not None
                and res["status"] == "ok"
                and check(task.oracle, ins[k], res["out"], exp[k], task.atol)
            )
            if not ok:
                failing.append(i + k)
    return failing


def outcomes_path(root: str | Path, benchmark: str, tier: str) -> Path:
    return Path(root) / "oracle" / benchmark / f"{tier}.json.gz"


def compute_outcomes(
    pools_root: str | Path,
    data_root: str | Path,
    benchmark: str,
    tier: str,
    tasks: dict[str, Task],
    exec_cfg: dict[str, Any],
    workers: int = 8,
) -> dict[str, Any]:
    """Failing hidden-test indices for every visible-passing candidate of a tier (identical
    programs of a task run once). Resumable per task; writes outcomes_path."""
    out_path = outcomes_path(data_root, benchmark, tier)
    done: dict[str, Any] = {}
    if out_path.exists():
        done = json.loads(gzip.decompress(out_path.read_bytes()))
    files = sorted((Path(pools_root) / benchmark / tier).glob("*.json"))

    def work(f: Path) -> tuple[str, dict[str, Any]]:
        pool = TierPool.from_dict(json.loads(f.read_text()))
        if pool.task_id in done:
            return pool.task_id, done[pool.task_id]
        task = prepare_task(tasks[pool.task_id])
        by_code: dict[str, list[int]] = {}
        cands: dict[str, list[int]] = {}
        mismatch = 0
        for c in pool.samples + [r for ch in pool.repairs.values() for r in ch]:
            if not c.passes_visible:
                continue
            h = hashlib.sha1(c.code.encode()).hexdigest()
            if h not in by_code:
                by_code[h] = hidden_failures(task, c.code, exec_cfg)
            cands[c.cand_id] = by_code[h]
            mismatch += int((not by_code[h]) != bool(c.hidden_pass))
        entry = {"n_hidden": len(task.hidden_inputs), "failing": cands, "label_mismatch": mismatch}
        release_tests(task)
        return pool.task_id, entry

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(work, f) for f in files])):
            tid, entry = fut.result()
            done[tid] = entry
            if k % 50 == 49:  # checkpoint
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(gzip.compress(json.dumps(done).encode()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gzip.compress(json.dumps(done).encode()))
    return {
        "tasks": len(done),
        "candidates": sum(len(e["failing"]) for e in done.values()),
        "label_mismatch": sum(e["label_mismatch"] for e in done.values()),
    }


def load_outcomes(data_root: str | Path, benchmark: str, tier: str) -> dict[str, Any]:
    return json.loads(gzip.decompress(outcomes_path(data_root, benchmark, tier).read_bytes()))


def weak_labels(outcomes: dict[str, Any], fraction: float, seed: int = 0) -> dict[str, bool]:
    """cand_id -> weakly correct: fails none of a random ``fraction`` of the task's hidden
    tests (the same subset for every candidate of the task, as a fixed test suite would)."""
    labels: dict[str, bool] = {}
    for tid, e in outcomes.items():
        n = e["n_hidden"]
        rng = np.random.default_rng(int(hashlib.sha1(f"{seed}|{tid}".encode()).hexdigest()[:8], 16))
        keep = set(rng.permutation(n)[: int(np.ceil(fraction * n))].tolist()) if n else set()
        for cid, failing in e["failing"].items():
            labels[cid] = not any(i in keep for i in failing)
    return labels


def accepted_ids(task_pools, policies, perms: int, cost_fn):
    """Replay keeping the accepted candidate's id: (configs, tasks, perms) object array, and
    the cost array."""
    m, n = len(policies), len(task_pools)
    ids = np.empty((m, n, perms), dtype=object)
    cost = np.zeros((m, n, perms))
    for j, pol in enumerate(policies):
        for i, pools in enumerate(task_pools):
            for p in range(perms):
                o = run_task(pools, pol, p)
                ids[j, i, p] = o.accepted.cand_id if o.accepted is not None else None
                cost[j, i, p] = sum(cost_fn(c) for c in o.drawn)
    return ids, cost


def _replay_with(ids: np.ndarray, cost: np.ndarray, label: dict[str, bool]):
    from .evaluate import Replay

    acc = np.vectorize(lambda x: x is not None, otypes=[float])(ids)
    cor = np.vectorize(lambda x: x is not None and label.get(x, False), otypes=[float])(ids)
    return Replay(
        loss=acc - cor, accepted=acc, correct=cor, cost=cost, fallbacks=np.zeros_like(cost)
    )


def run_oracle_shift(
    task_pools: list[list[TierPool]],
    tier_names: list[str],
    outcomes: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    cost_fn,
    family: str = "singles",
    fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
) -> dict[str, Any]:
    """Calibrate on weak labels, measure on full labels, per oracle strength."""
    ctl, ev = cfg["controller"], cfg["evaluation"]
    perms = int(ctl.get("permutations", 1))
    fam = family if len(tier_names) > 1 else "shared"
    _configs, pols, method, paths = family_policies(fam, tier_names, ctl, ev, {})
    ids, cost = accepted_ids(task_pools, pols, perms, cost_fn)
    merged = {
        cid: f for o in outcomes.values() for e in o.values() for cid, f in e["failing"].items()
    }
    full = {cid: not f for cid, f in merged.items()}
    rep_full = _replay_with(ids, cost, full)
    out: dict[str, Any] = {
        "tiers": tier_names,
        "family": fam,
        "n_tasks": len(task_pools),
        "rows": [],
    }
    factor = float(ev.get("abstain_factor", 0.0))
    for f in fractions:
        weak = {}
        for o in outcomes.values():
            weak.update(weak_labels(o, f))
        rep_weak = _replay_with(ids, cost, weak)
        for a in ev["alphas"]:
            r = run_splits(rep_weak, a, ev["delta"], ev["n_splits"], ev["calib_frac"],
                           method=method, paths=paths, abstain_factor=factor)  # fmt: skip
            chosen = [s for s in r["splits"] if s["chosen"] is not None]
            true_pool = [float(rep_full.loss[s["chosen"]].mean()) for s in chosen]
            out["rows"].append(
                {
                    "fraction": f,
                    "alpha": a,
                    "certified_splits": len(chosen),
                    "claimed_risk_weak": float(np.mean([s["test_risk"] for s in chosen]))
                    if chosen
                    else None,
                    "true_risk": float(np.mean(true_pool)) if chosen else None,
                    "violation_rate_true": float(np.mean([t > a for t in true_pool]))
                    if chosen
                    else None,
                    "coverage": float(np.mean([s["coverage"] for s in chosen])) if chosen else None,
                }
            )
    return out
