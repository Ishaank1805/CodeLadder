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
from .llm import ChatClient, QuotaExhausted
from .probes import make_probes, make_stdin_probes
from .prompts import (
    confidence_messages,
    extract_code,
    initial_messages,
    p_true,
    parse_confidence,
    ptrue_messages,
    repair_messages,
)
from .scoring import behaviour_signature, visible_report
from .tasks import prepare_task, release_tests
from .types import Candidate, Task, TierPool

log = logging.getLogger(__name__)

STDIN_PROBES = "stdin-v2"  # tag on stdin candidates whose signature uses make_stdin_probes


def pool_path(root: str | Path, task: Task, tier: str) -> Path:
    safe = task.task_id.replace("/", "__")
    return Path(root) / task.benchmark / tier / f"{safe}.json"


def _limits(
    ref_times: list[float] | None, n: int, exec_cfg: dict[str, Any], task: Task | None = None
) -> list[float]:
    """Per-input time limits: max(floor, factor x reference time) as in EvalPlus. The floor is
    the task's own limit when it has one (LiveCodeBench: 6 s), else ``per_call_timeout_s``."""
    floor = (task.time_limit_s if task and task.time_limit_s else None) or exec_cfg[
        "per_call_timeout_s"
    ]
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
        memory_mb or exec_cfg["memory_mb"],
        total_timeout=total,
        timeouts=limits,
        mode=task.mode,
    )


def hidden_chunks(n: int, largest: int = 250) -> list[tuple[int, int]]:
    """Chunk boundaries 1, 4, 16, 64, then ``largest`` at a time: a wrong or looping program
    usually fails within the first few inputs, so it stops early instead of running (and
    timing out on) a full chunk, while a passing program costs only a few extra processes."""
    out, i, size = [], 0, 1
    while i < n:
        size = min(size, largest)
        out.append((i, min(n, i + size)))
        i, size = i + size, size * 4
    return out


def passes_hidden(task: Task, code: str, exec_cfg: dict[str, Any]) -> bool:
    """Hidden-test label: True iff every hidden input passes. Runs in growing chunks and
    stops at the first failing chunk (the label does not depend on the chunking)."""
    expected = task.hidden_expected or []
    limits = _limits(task.hidden_ref_times, len(task.hidden_inputs), exec_cfg, task)
    mem = int(exec_cfg.get("memory_mb_hidden", 4096))  # a few tasks need ~4 GB (Mbpp/255)
    for i, j in hidden_chunks(len(task.hidden_inputs), int(exec_cfg.get("hidden_chunk", 250))):
        ins, exp = task.hidden_inputs[i:j], expected[i:j]
        run = _run(task, code, ins, limits[i:j], exec_cfg, memory_mb=mem)
        rate, _, _ = visible_report(run, ins, exp, task.atol, task.oracle)
        if rate < 1.0:
            return False
    return True


def task_probes(task: Task, probe_cfg: dict[str, Any]) -> list[Any]:
    """Probe inputs for agreement: mutated arguments, or format-preserving stdin mutations
    (BigCodeBench uses model-written probe calls run by bcb_runner instead)."""
    if task.mode == "bcb":
        return []
    make = make_stdin_probes if task.mode == "stdin" else make_probes
    return make(task.visible_inputs, probe_cfg["n"], probe_cfg["seed"])


def probe_signature(task: Task, code: str, probes: list[Any], exec_cfg: dict[str, Any]) -> str:
    if not probes:
        return ""
    if task.mode == "stdin":
        # Mutated inputs can be outside the problem's constraints; keep a short fixed limit
        # so a program that loops on such an input costs little.
        limits = [exec_cfg["per_call_timeout_s"]] * len(probes)
    else:
        limits = _limits(None, len(probes), exec_cfg, task)
    return behaviour_signature(_run(task, code, probes, limits, exec_cfg))


def evaluate_code(
    task: Task, code: str, probes: list[list[Any]], exec_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run one program on visible and probe inputs, and on hidden inputs if it passes visible."""
    if task.mode == "bcb":  # unit tests in the benchmark's own environment (bcb.py)
        from .bcb import evaluate_bcb

        return evaluate_bcb(task, code, exec_cfg)
    ins = task.visible_inputs
    vis = _run(task, code, ins, _limits(task.visible_ref_times, len(ins), exec_cfg, task), exec_cfg)
    rate, err, feedback = visible_report(
        vis, ins, task.visible_expected or [], task.atol, task.oracle
    )
    # Only visible-passing candidates are ever scored, so only they need a signature.
    sig = probe_signature(task, code, probes, exec_cfg) if rate >= 1.0 else ""
    # A candidate that fails a visible test is wrong by definition: skip the hidden tests.
    hidden = rate >= 1.0 and passes_hidden(task, code, exec_cfg)
    return {
        "visible_pass_rate": rate,
        "error_type": err,
        "feedback": feedback,
        "signature": sig,
        "hidden_pass": hidden,
    }


def make_candidate(
    task: Task,
    tier: str,
    kind: str,
    sample_idx: int,
    depth: int,
    comp,
    probes: list[Any],
    stress: list[str],
    exec_cfg: dict[str, Any],
) -> Candidate:
    """Extract the code from a completion and run it: visible tests, probe signature, stress
    signature (stdin problems with stress inputs) and hidden tests (label)."""
    code = extract_code(comp.text, task)
    res = evaluate_code(task, code, probes, exec_cfg)
    extra: dict[str, Any] = {"started_at": comp.started_at}
    if task.mode == "stdin" and res["signature"]:
        extra["probes"] = STDIN_PROBES
    if stress and res["visible_pass_rate"] >= 1.0:
        extra["stress_sig"] = stress_signature(task, code, stress, exec_cfg)
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
        extra=extra,
        **res,
    )


def build_pool(
    task: Task,
    tier: str,
    client: ChatClient,
    gen_cfg: dict[str, Any],
    probe_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    existing: TierPool | None = None,
) -> TierPool:
    """Build a pool, or top up ``existing`` to the current settings (more samples, longer
    repair chains) while keeping every candidate it already has. Seeds depend only on the
    sample index and repair round, so a top-up gives the same pool as generating from scratch.
    """
    probes = task_probes(task, probe_cfg)
    stress: list[str] = []
    if task.mode == "stdin" and probe_cfg.get("stress_root"):
        from .stress import load_stress_inputs

        stress = load_stress_inputs(probe_cfg["stress_root"], task)
    sampling = dict(
        temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"], max_tokens=gen_cfg["max_tokens"]
    )

    def make(kind: str, sample_idx: int, depth: int, comp) -> Candidate:
        return make_candidate(task, tier, kind, sample_idx, depth, comp, probes, stress, exec_cfg)

    base_seed = int(gen_cfg.get("seed", 0))
    samples: list[Candidate] = list(existing.samples) if existing else []
    for i in range(len(samples), gen_cfg["n_samples"]):
        comp = client.complete(initial_messages(task), seed=base_seed * 100_003 + i, **sampling)
        samples.append(make("sample", i, 0, comp))

    repairs: dict[int, list[Candidate]] = dict(existing.repairs) if existing else {}
    rep_cfg = gen_cfg.get("repair", {})
    failing = [c for c in samples if not c.passes_visible][: rep_cfg.get("parents", 0)]
    for parent in failing:
        chain = list(repairs.get(parent.sample_idx, []))
        if chain and chain[-1].passes_visible:
            continue  # already repaired
        current = chain[-1] if chain else parent
        for d in range(len(chain) + 1, rep_cfg.get("depth", 0) + 1):
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
    retries: int = 2,
    extend: bool = False,
    max_consecutive_failures: int = 12,
) -> list[Path]:
    """Build and save pools for all tasks.

    Resumable: existing pools are skipped, or with ``extend`` topped up to the current
    settings (see ``build_pool``). Tasks that fail (server hiccups, timeouts) are retried up
    to ``retries`` more times within the run; after ``max_consecutive_failures`` failures in a
    row the run stops (the server is most likely gone) so the job can end and be resubmitted.
    """
    root = cfg["paths"]["pools"]
    todo = [t for t in tasks if overwrite or extend or not pool_path(root, t, tier).exists()]
    log.info(
        "tier %s: %d tasks to generate (%d already done)", tier, len(todo), len(tasks) - len(todo)
    )
    written: list[Path] = []

    def work(task: Task) -> Path:
        prepare_task(task)  # load tests / expected outputs lazily, inside the worker
        path = pool_path(root, task, tier)
        old = (
            TierPool.from_dict(json.loads(path.read_text()))
            if extend and not overwrite and path.exists()
            else None
        )
        probe_cfg = dict(cfg["probes"], stress_root=cfg["paths"].get("stress"))
        pool = build_pool(
            task, tier, client, cfg["generation"], probe_cfg, cfg["execution"], existing=old
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pool.to_dict()))
        return path

    failed = todo
    streak = 0  # consecutive failed tasks
    for attempt in range(1, retries + 2):
        if not failed:
            break
        if attempt > 1:
            log.warning("retrying %d failed tasks (attempt %d)", len(failed), attempt)
        still_failed: list[Task] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, t): t for t in failed}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    written.append(fut.result())
                    streak = 0
                    continue
                except QuotaExhausted as e:
                    # API quota spent: stop cleanly; rerunning later resumes from here.
                    log.warning("%s; stopping this run (%d pools written)", e, len(written))
                    ex.shutdown(wait=True, cancel_futures=True)
                    return written
                except Exception:
                    log.exception("failed on %s (attempt %d)", task.task_id, attempt)
                    still_failed.append(task)
                    streak += 1
                    if streak >= max_consecutive_failures:
                        # The model server is most likely gone: every further request would
                        # wait through its retries. Stop; a rerun resumes from the pools written.
                        log.error("tier %s: %d tasks failed in a row; stopping this run (%d pools "
                                  "written)", tier, streak, len(written))  # fmt: skip
                        ex.shutdown(wait=False, cancel_futures=True)
                        return written
        failed = still_failed
    if failed:
        log.error(
            "tier %s: %d tasks still failing: %s", tier, len(failed), [t.task_id for t in failed]
        )
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


def truncate_pool(pool: TierPool, max_samples: int = 0, max_depth: int = 0) -> TierPool:
    """A smaller pool for ablations: the first ``max_samples`` samples and repair chains cut
    to ``max_depth`` rounds (0 keeps everything). Equivalent to generating with those settings,
    since samples and repair rounds are drawn in order with fixed seeds."""
    samples = pool.samples[:max_samples] if max_samples else pool.samples
    keep = {c.sample_idx for c in samples}
    repairs = {
        s: (ch[:max_depth] if max_depth else ch) for s, ch in pool.repairs.items() if s in keep
    }
    return TierPool(task_id=pool.task_id, tier=pool.tier, samples=samples, repairs=repairs)


def stress_signature(task: Task, code: str, stress: list[str], exec_cfg: dict[str, Any]) -> str:
    """Behaviour on the stress inputs, with the problem's own time limit (a too-slow program
    shows up as a timeout here, as it would on the hidden tests)."""
    limit = task.time_limit_s or exec_cfg["per_call_timeout_s"]
    mem = int(exec_cfg.get("memory_mb_hidden", 4096))
    return behaviour_signature(_run(task, code, stress, [limit] * len(stress), exec_cfg, mem))


def add_stdin_probes(
    root: str | Path,
    benchmark: str,
    tier: str,
    tasks: dict[str, Task],
    probe_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    workers: int = 8,
    stress_root: str | Path | None = None,
) -> int:
    """Give visible-passing candidates of stdin tasks a probe signature (pools generated before
    stdin probes existed) and, with ``stress_root``, a stress signature (``extra["stress_sig"]``,
    see ``stress.py``). Resumable: candidates already done are skipped."""
    from .stress import load_stress_inputs

    files = sorted((Path(root) / benchmark / tier).glob("*.json"))

    def work(f: Path) -> int:
        pool = TierPool.from_dict(json.loads(f.read_text()))
        task = tasks[pool.task_id]
        if task.mode != "stdin":
            return 0
        stress = load_stress_inputs(stress_root, task) if stress_root else []
        passing = [
            c
            for c in pool.samples + [r for ch in pool.repairs.values() for r in ch]
            if c.passes_visible
        ]
        todo = [c for c in passing if c.extra.get("probes") != STDIN_PROBES]
        todo_stress = [c for c in passing if stress and "stress_sig" not in c.extra]
        if not todo and not todo_stress:
            return 0
        prepare_task(task)
        probes = task_probes(task, probe_cfg)
        release_tests(task)  # only visible inputs are needed
        for c in todo:
            c.signature = probe_signature(task, c.code, probes, exec_cfg)
            c.extra["probes"] = STDIN_PROBES
        for c in todo_stress:
            c.extra["stress_sig"] = stress_signature(task, c.code, stress, exec_cfg)
        f.write_text(json.dumps(pool.to_dict()))
        return len(set(map(id, todo)) | set(map(id, todo_stress)))

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in files]):
            try:
                total += fut.result()
            except Exception:
                log.exception("adding stdin probes failed")
    return total


def annotate_confidence(
    root: str | Path,
    benchmark: str,
    tier: str,
    client: ChatClient,
    tasks: dict[str, Task],
    workers: int = 8,
    max_tokens: int = 8,
    prefill: bool = False,
    redo: bool = False,
) -> int:
    """Add the model's verbalised confidence (extra["self_conf"]) to every visible-passing
    candidate of a tier. Resumable: candidates with a rating are skipped; unparsed ones (None)
    are asked again. Returns how many candidates got a parsed rating in this run, so a rerun
    over cached unparsable replies returns 0.

    ``prefill`` starts the answer for the model (``confidence_messages``). ``redo`` asks again
    for candidates rated without prefill (the old rating is kept in extra["self_conf_v1"]);
    candidates already rated with prefill are skipped, so it is resumable too."""
    files = sorted((Path(root) / benchmark / tier).glob("*.json"))

    def work(f: Path) -> int:
        pool = TierPool.from_dict(json.loads(f.read_text()))
        task = tasks[pool.task_id]
        n = changed = 0
        for c in pool.samples + [r for ch in pool.repairs.values() for r in ch]:
            if not c.passes_visible:
                continue
            if redo and prefill and not c.extra.get("self_conf_prefill"):
                if "self_conf" in c.extra and "self_conf_v1" not in c.extra:
                    c.extra["self_conf_v1"] = c.extra["self_conf"]
                c.extra.pop("self_conf", None)
            if c.extra.get("self_conf") is not None:
                continue  # unparsed ratings (None) are asked again
            kw = {"continue_final": True} if prefill else {}
            comp = client.complete(
                confidence_messages(task, c.code, prefill=prefill),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                seed=0,
                **kw,
            )
            had = "self_conf" in c.extra
            c.extra["self_conf"] = parse_confidence(comp.text, prefilled=prefill)
            if prefill:
                c.extra["self_conf_prefill"] = True
            c.extra["self_conf_tokens"] = comp.tokens_in + comp.tokens_out
            n += c.extra["self_conf"] is not None
            changed += not had or c.extra["self_conf"] is not None
        if changed:
            f.write_text(json.dumps(pool.to_dict()))
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in files]):
            try:
                total += fut.result()
            except Exception:
                log.exception("confidence annotation failed")
    return total


def annotate_ptrue(
    root: str | Path,
    benchmark: str,
    tier: str,
    client: ChatClient,
    tasks: dict[str, Task],
    workers: int = 8,
    top_logprobs: int = 10,
) -> int:
    """Add P(True) (extra["p_true"]) to every visible-passing candidate of a tier: one
    request per candidate with a one-token answer and its top alternatives (local servers).
    Resumable: candidates with a value are skipped; returns how many got one in this run."""
    files = sorted((Path(root) / benchmark / tier).glob("*.json"))

    def work(f: Path) -> int:
        pool = TierPool.from_dict(json.loads(f.read_text()))
        task = tasks[pool.task_id]
        n = 0
        for c in pool.samples + [r for ch in pool.repairs.values() for r in ch]:
            if not c.passes_visible or c.extra.get("p_true") is not None:
                continue
            comp = client.complete(
                ptrue_messages(task, c.code),
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                seed=0,
                top_logprobs=top_logprobs,
                continue_final=True,
            )
            c.extra["p_true"] = p_true(comp.first_top)
            c.extra["p_true_tokens"] = comp.tokens_in + comp.tokens_out
            n += c.extra["p_true"] is not None
        if n:
            f.write_text(json.dumps(pool.to_dict()))
        return n

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in files]):
            try:
                total += fut.result()
            except Exception:
                log.exception("P(True) annotation failed")
    return total
