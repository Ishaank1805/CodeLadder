"""One-time generation pass: build the cached candidate pool for each (task, tier).

For every task and tier we draw ``n_samples`` independent samples, then build
repair chains for the first ``parents`` samples that fail a visible test. Every
candidate is executed on the visible tests (signals), the probe inputs
(agreement) and the hidden tests (label). Pools are written as JSON so all later
experiments are offline replays.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .executor import run_program
from .llm import ChatClient
from .probes import make_probes
from .prompts import extract_code, initial_messages, repair_messages
from .scoring import behaviour_signature, visible_report
from .types import Candidate, Task, TierPool

log = logging.getLogger(__name__)


def pool_path(root: str | Path, task: Task, tier: str) -> Path:
    safe = task.task_id.replace("/", "__")
    return Path(root) / task.benchmark / tier / f"{safe}.json"


def _limits(ref_times: list[float] | None, n: int, exec_cfg: dict[str, Any]) -> list[float]:
    """Per-input time limits, as in EvalPlus: max(min limit, factor x reference time)."""
    floor = exec_cfg["per_call_timeout_s"]
    factor = exec_cfg.get("ref_time_factor", 4.0)
    if not ref_times or len(ref_times) != n:
        return [floor] * n
    return [max(floor, factor * t) for t in ref_times]


def _run(
    task: Task, code: str, ins, limits: list[float], exec_cfg: dict[str, Any], memory_mb: int = 0
):
    cap = max(exec_cfg.get("total_timeout_cap_s", 60.0), 1.5 * sum(limits))
    total = min(sum(limits) + 2 * exec_cfg["per_call_timeout_s"] + 5.0, cap)
    return run_program(
        code,
        task.entry_point,
        ins,
        exec_cfg["per_call_timeout_s"],
        exec_cfg["memory_mb"],
        total_timeout=total,
        timeouts=limits,
    )


def passes_hidden(task: Task, code: str, exec_cfg: dict[str, Any]) -> bool:
    """Hidden-test label, run in chunks that stop at the first failing chunk."""
    chunk = int(exec_cfg.get("hidden_chunk", 250))
    expected = task.hidden_expected or []
    limits = _limits(task.hidden_ref_times, len(task.hidden_inputs), exec_cfg)
    for i in range(0, len(task.hidden_inputs), chunk):
        ins, exp = task.hidden_inputs[i : i + chunk], expected[i : i + chunk]
        mem = int(exec_cfg.get("memory_mb_hidden", 4096))  # a few tasks need ~4 GB (Mbpp/255)
        run = _run(task, code, ins, limits[i : i + chunk], exec_cfg, memory_mb=mem)
        rate, _, _ = visible_report(run, ins, exp, task.atol, task.oracle)
        if rate < 1.0:
            return False
    return True


def evaluate_code(
    task: Task, code: str, probes: list[list[Any]], exec_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run one program on visible and probe inputs, and on hidden inputs if it passes visible."""
    ins = task.visible_inputs
    vis = _run(task, code, ins, _limits(task.visible_ref_times, len(ins), exec_cfg), exec_cfg)
    rate, err, feedback = visible_report(
        vis, ins, task.visible_expected or [], task.atol, task.oracle
    )
    sig = ""
    if probes:
        sig = behaviour_signature(
            _run(task, code, probes, _limits(None, len(probes), exec_cfg), exec_cfg)
        )
    # A candidate that fails a visible test is wrong by definition: skip the hidden tests.
    hidden = rate >= 1.0 and passes_hidden(task, code, exec_cfg)
    return {
        "visible_pass_rate": rate,
        "error_type": err,
        "feedback": feedback,
        "signature": sig,
        "hidden_pass": hidden,
    }


def build_pool(
    task: Task,
    tier: str,
    client: ChatClient,
    gen_cfg: dict[str, Any],
    probe_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
) -> TierPool:
    probes = make_probes(task.visible_inputs, probe_cfg["n"], probe_cfg["seed"])
    sampling = dict(
        temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"], max_tokens=gen_cfg["max_tokens"]
    )

    def make(kind: str, sample_idx: int, depth: int, comp) -> Candidate:
        code = extract_code(comp.text, task)
        res = evaluate_code(task, code, probes, exec_cfg)
        return Candidate(
            cand_id=f"{task.task_id}|{tier}|{sample_idx}|{depth}",
            tier=tier,
            kind=kind,
            sample_idx=sample_idx,
            depth=depth,
            code=code,
            tokens_in=comp.tokens_in,
            tokens_out=comp.tokens_out,
            latency_s=comp.latency_s,
            mean_logprob=comp.mean_logprob,
            extra={"started_at": comp.started_at},
            **res,
        )

    base_seed = int(gen_cfg.get("seed", 0))
    samples: list[Candidate] = []
    for i in range(gen_cfg["n_samples"]):
        comp = client.complete(initial_messages(task), seed=base_seed * 100_003 + i, **sampling)
        samples.append(make("sample", i, 0, comp))

    repairs: dict[int, list[Candidate]] = {}
    rep_cfg = gen_cfg.get("repair", {})
    failing = [c for c in samples if not c.passes_visible][: rep_cfg.get("parents", 0)]
    for parent in failing:
        chain: list[Candidate] = []
        current = parent
        for d in range(1, rep_cfg.get("depth", 0) + 1):
            msgs = repair_messages(task, current.code, current.feedback)
            comp = client.complete(
                msgs, seed=base_seed * 100_003 + 10_000 + parent.sample_idx * 100 + d, **sampling
            )
            current = make("repair", parent.sample_idx, d, comp)
            chain.append(current)
            if current.passes_visible:
                break  # nothing left to repair against
        repairs[parent.sample_idx] = chain

    return TierPool(task_id=task.task_id, tier=tier, samples=samples, repairs=repairs)


def generate_pools(
    tasks: list[Task],
    tier: str,
    client: ChatClient,
    cfg: dict[str, Any],
    workers: int = 8,
    overwrite: bool = False,
) -> list[Path]:
    """Build and save pools for all tasks (resumable: existing pools are skipped)."""
    root = cfg["paths"]["pools"]
    todo = [t for t in tasks if overwrite or not pool_path(root, t, tier).exists()]
    log.info(
        "tier %s: %d tasks to generate (%d already done)", tier, len(todo), len(tasks) - len(todo)
    )
    written: list[Path] = []

    def work(task: Task) -> Path:
        pool = build_pool(task, tier, client, cfg["generation"], cfg["probes"], cfg["execution"])
        path = pool_path(root, task, tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pool.to_dict()))
        return path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, t): t for t in todo}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                written.append(fut.result())
            except Exception:
                log.exception("failed on %s", task.task_id)
    return written


def load_pools(root: str | Path, benchmark: str, tiers: list[str]) -> dict[str, list[TierPool]]:
    """Return {task_id: [pool for tier 1, pool for tier 2, ...]} for tasks present in every tier."""
    by_task: dict[str, dict[str, TierPool]] = {}
    for tier in tiers:
        d = Path(root) / benchmark / tier
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            pool = TierPool.from_dict(json.loads(f.read_text()))
            by_task.setdefault(pool.task_id, {})[tier] = pool
    return {tid: [p[t] for t in tiers] for tid, p in by_task.items() if all(t in p for t in tiers)}
