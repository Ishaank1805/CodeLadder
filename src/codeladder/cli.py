"""Command-line entry point: ``codeladder <command>``.

Commands
  check-env     print Python/package versions and test each enabled tier's endpoint
  generate      build cached candidate pools for one tier and benchmark
  evaluate      replay the controller over pools and run Learn-then-Test splits
  annotate-confidence  ask each tier's model for its confidence in its visible-passing candidates
  annotate-ptrue  P(True): the model's probability that each visible-passing candidate is correct
  add-probes    give stdin candidates of existing pools a probe signature (agreement)
  make-stress   have a model write large valid inputs for stdin tasks (stress probes)
  make-selftests  baselines: each tier writes assert-style tests per task (Model Cascading for Code)
  make-traces   baselines: greedy answers with token statistics (Bayesian self-escalation)
  baseline-exec baselines: run every candidate against the self-written tests
  baselines     replay the published cascades on the same tasks, over 100 random splits
  route         X9: learned up-front router, calibrated like the other families
  actions       X4: learned repair-or-resample policy against fixed actions, same guarantee
  economics     X13: sweep the top tier's price; when do sub-ladders beat model selection?
  compare       X14: paired task-level bootstrap intervals between two evaluate runs
  guaranteed    X16: guaranteed baselines (per-model abstention, Trust-or-Escalate cascade)
  hidden-outcomes  X17: failing hidden tests of every visible-passing candidate (CPU job)
  bcb-check     X18: run BigCodeBench reference solutions; exclude tasks failing in our environment
  make-bcb-probes  X18: a model writes label-free probe calls per BigCodeBench task
  oracle-shift  X17: calibrate on weaker test oracles, measure the true risk
  export-policy X15: certify a run's family on all its tasks and save the chosen ladder
  solve         X15: run a saved policy on one problem (live models, or --replay from pools)
  demo          X15: interactive demo in the browser (codeladder demo-bundle first, on the cluster)
  demo-bundle   X15: pack the tasks and pools the demo replays into data/demo
  energy-log    log GPU energy (run on the GPU node, next to the model server)
  attach-energy add energy_j to pools using an energy log and J/token for the run
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .results import find_result, result_path


def _cmd_check_env(args: argparse.Namespace) -> int:
    from .config import enabled_tiers, load_tiers

    print(f"codeladder {__version__} | python {platform.python_version()} | {platform.platform()}")
    for mod in ("numpy", "scipy", "yaml", "openai", "evalplus", "pynvml"):
        try:
            m = __import__(mod)
            print(f"  {mod:10s} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"  {mod:10s} MISSING")
    for tier in enabled_tiers(load_tiers(args.models)):
        try:
            url = tier.resolve_base_url()
        except Exception as e:  # endpoint file missing etc.
            print(f"  tier {tier.name}: not reachable ({e})")
            continue
        try:
            from openai import OpenAI

            models = [
                m.id for m in OpenAI(base_url=url, api_key=tier.api_key(), timeout=10).models.list()
            ]
            print(f"  tier {tier.name}: {url} serves {models}")
        except Exception as e:
            print(f"  tier {tier.name}: {url} ERROR {e}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .config import load_experiment, load_tiers
    from .generate import generate_pools
    from .live import generation_settings
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    cfg["generation"] = generation_settings(cfg, args.benchmark, args.tier)
    if args.n_samples:
        cfg["generation"]["n_samples"] = args.n_samples
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    # One cache file per (tier, benchmark): each generation job is then the only writer of
    # its SQLite file, which matters on NFS where concurrent writers corrupt or fail.
    cache_path = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-{args.benchmark}.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache_path))
    tasks = load_tasks(args.benchmark, args.limit)  # prepared lazily inside the workers
    if args.task_ids:
        keep = set(json.loads(Path(args.task_ids).read_text()))
        tasks = [t for t in tasks if t.task_id in keep]
    t0 = time.time()
    written = generate_pools(
        tasks,
        tier.name,
        client,
        cfg,
        workers=args.workers,
        overwrite=args.overwrite,
        extend=args.extend,
    )
    run = {
        "tier": tier.name,
        "benchmark": args.benchmark,
        "start": t0,
        "end": time.time(),
        "pools": len(written),
    }
    runs = Path(cfg["paths"]["pools"]) / "runs.jsonl"
    runs.parent.mkdir(parents=True, exist_ok=True)
    with open(runs, "a") as f:
        f.write(json.dumps(run) + "\n")
    print(json.dumps(run))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .config import enabled_tiers, load_experiment, load_tiers
    from .evaluate import make_cost_fn, run_experiment
    from .generate import load_pools, truncate_pool

    cfg = load_experiment(args.experiment)
    if args.action:
        cfg["controller"]["action"] = args.action
    if args.score:
        cfg["controller"]["score"] = args.score
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.family:
        cfg["evaluation"]["family"] = args.family
    if args.abstain_factor is not None:
        cfg["evaluation"]["abstain_factor"] = args.abstain_factor
    if args.calib_sizes:
        cfg["evaluation"]["calib_sizes"] = args.calib_sizes
    if args.calib_on:
        cfg["evaluation"]["calib_on"] = args.calib_on
    if args.lower_budget:
        cfg["controller"]["lower_budget"] = args.lower_budget
    if args.cost_metric:
        cfg["evaluation"]["cost_metric"] = args.cost_metric
    if args.method:
        cfg["evaluation"]["two_level_method"] = args.method
    enabled = {t.name: t for t in enabled_tiers(load_tiers(args.models))}
    # Ladder order is the order given (cheapest first), so cross-family ladders such as
    # "--tiers d1 t2" run DeepSeek 1.3B before Qwen 7B.
    tiers = [enabled[n] for n in args.tiers] if args.tiers else list(enabled.values())
    names = [t.name for t in tiers]
    pools, group_of = {}, {}
    # A fixed task list keeps a run on the same problems while other pools are still being
    # generated (for example the API LiveCodeBench subset while the full set runs).
    keep = set(json.loads(Path(args.task_ids).read_text())) if args.task_ids else None
    for bench in args.benchmark:  # several benchmarks are pooled for calibration
        bp = load_pools(cfg["paths"]["pools"], bench, names)
        if keep is not None:
            bp = {tid: ps for tid, ps in bp.items() if tid in keep}
        if args.max_samples or args.max_repair_depth:  # E7 ablations on generation settings
            bp = {
                tid: [truncate_pool(p, args.max_samples, args.max_repair_depth) for p in ps]
                for tid, ps in bp.items()
            }
        pools.update(bp)
        if args.group_by == "difficulty":  # e.g. LiveCodeBench easy / medium / hard
            from .tasks import load_tasks

            diff = {t.task_id: t.difficulty for t in load_tasks(bench)}
            group_of.update({tid: f"{bench}:{diff.get(tid) or 'none'}" for tid in bp})
        else:
            group_of.update({tid: bench for tid in bp})
    if not pools:
        print(
            f"no tasks with pools for all tiers {names} in {cfg['paths']['pools']}/{args.benchmark}"
        )
        return 1
    cost_fn = make_cost_fn(
        cfg["evaluation"]["cost_metric"],
        {t.name: t.params_b for t in tiers},
        cfg["controller"].get("score", "agreement"),
    )
    out = result_path(
        cfg["paths"]["results"],
        (
            f"{'+'.join(args.benchmark)}-{'-'.join(names)}-{cfg['controller']['action']}"
            f"-{cfg['controller'].get('score', 'agreement')}"
            f"-{cfg['evaluation'].get('family', 'shared') if len(names) > 1 else 'shared'}{args.tag}"
        ),
    )
    if args.shift:
        return _shift(args, cfg, pools, names, cost_fn, out)
    summary = run_experiment(
        list(pools.values()), names, cfg, cost_fn, out, groups=[group_of[t] for t in pools]
    )
    brief = {k: v for k, v in summary.items() if k != "calibrated"}
    brief["calibrated"] = [
        {k: v for k, v in c.items() if k != "splits"} for c in summary["calibrated"]
    ]
    print(json.dumps(brief, indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .config import load_experiment
    from .demo import serve

    serve(load_experiment(args.experiment), args.models, port=args.port, host=args.host,
          bundle=Path(args.bundle) if args.bundle else None)  # fmt: skip
    return 0


def _cmd_demo_bundle(args: argparse.Namespace) -> int:
    from .demo import make_bundle

    print(json.dumps(make_bundle(args.out, tuple(args.benchmark))))
    return 0


def _cmd_export_policy(args: argparse.Namespace) -> int:
    from .config import load_experiment
    from .live import export_policy

    cfg = load_experiment(args.experiment)
    run = find_result(cfg["paths"]["results"], args.run)
    pol = export_policy(run, cfg, args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pol, indent=2))
    print(json.dumps({k: v for k, v in pol.items() if k != "guarantee"}, indent=2))
    print(pol["guarantee"])
    print(f"wrote {args.out}")
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    from .config import load_experiment, load_tiers
    from .live import custom_task, load_policy, replay, solve
    from .tasks import load_tasks, prepare_task

    cfg = load_experiment(args.experiment)
    policy, meta = load_policy(args.policy)
    tiers = meta["tiers"]
    print(f"policy: {meta.get('setting')} from {meta.get('source_run')} (alpha {meta['alpha']})")
    if args.problem:  # a custom function problem: prompt, entry_point, tests [{input, output}]
        task = custom_task(args.problem)
    else:
        task = next((t for t in load_tasks(args.benchmark) if t.task_id == args.task_id), None)
        if task is None:
            print(f"no task {args.task_id} in {args.benchmark}")
            return 1
    if args.replay:
        from .generate import load_pools

        pools = load_pools(cfg["paths"]["pools"], args.benchmark, tiers).get(task.task_id)
        if pools is None:
            print(f"no cached pools of {task.task_id} for tiers {tiers}")
            return 1
        out = replay(pools, policy, args.perm)
        for step in out["trace"]:
            print("  ", step)
    else:
        from .llm import OpenAIChatClient, ResponseCache

        if task.benchmark != "custom":
            prepare_task(task)
        enabled = {t.name: t for t in load_tiers(args.models)}
        cache = Path(cfg["paths"]["llm_cache"])
        clients = {
            n: OpenAIChatClient(enabled[n], ResponseCache(cache / f"live-{n}.sqlite"))
            for n, th in zip(tiers, policy.thresholds, strict=True)
            if th is not None
        }
        out = solve(task, policy, tiers, clients, cfg, log=lambda m: print("  ", m))
    acc = out["accepted"]
    tokens = sum(c.tokens_out + c.tokens_in for c in out["drawn"])
    print(f"candidates drawn: {len(out['drawn'])}, tokens: {tokens}")
    if acc is None:
        print("result: abstained (no candidate certified at this risk level)")
        return 0
    label = "pass" if acc.hidden_pass else "FAIL"
    if task.benchmark == "custom":
        label = "not available (custom problem)"
    print(f"result: accepted at tier {out['tier']}; hidden tests: {label}")
    print(acc.code)
    return 0


def _cmd_bcb_check(args: argparse.Namespace) -> int:
    from .bcb import BCB_DIR, check_references
    from .config import load_experiment
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    excluded = BCB_DIR / "excluded.json"
    if excluded.exists():
        excluded.unlink()  # check every task again
    tasks = load_tasks("bigcodebench", args.limit)
    print(json.dumps(check_references(tasks, cfg["execution"], args.workers)))
    return 0


def _cmd_make_bcb_probes(args: argparse.Namespace) -> int:
    from .bcb import make_bcb_probes
    from .config import load_experiment, load_tiers
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    cache = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-bigcodebench-probes.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache))
    print(json.dumps(make_bcb_probes(load_tasks("bigcodebench"), client, workers=args.workers)))
    return 0


def _cmd_hidden_outcomes(args: argparse.Namespace) -> int:
    from .config import REPO_ROOT, load_experiment
    from .oracle import compute_outcomes
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    stats = compute_outcomes(cfg["paths"]["pools"], REPO_ROOT / "data", args.benchmark,
                             args.tier, tasks, cfg["execution"], args.workers)  # fmt: skip
    print(f"{args.tier}/{args.benchmark}: {json.dumps(stats)}")
    return 0


def _cmd_oracle_shift(args: argparse.Namespace) -> int:
    from .config import REPO_ROOT, enabled_tiers, load_experiment, load_tiers
    from .evaluate import make_cost_fn
    from .generate import load_pools
    from .oracle import load_outcomes, run_oracle_shift

    cfg = load_experiment(args.experiment)
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.score:
        cfg["controller"]["score"] = args.score
    if args.abstain_factor is not None:
        cfg["evaluation"]["abstain_factor"] = args.abstain_factor
    enabled = {t.name: t for t in enabled_tiers(load_tiers(args.models))}
    tiers = [enabled[n] for n in args.tiers]
    names = [t.name for t in tiers]
    pools, outcomes = {}, {}
    for bench in args.benchmark:
        pools.update(load_pools(cfg["paths"]["pools"], bench, names))
        for t in names:
            outcomes[f"{bench}/{t}"] = load_outcomes(REPO_ROOT / "data", bench, t)
    cost_fn = make_cost_fn(
        cfg["evaluation"]["cost_metric"],
        {t.name: t.params_b for t in tiers},
        cfg["controller"].get("score", "agreement"),
    )
    summary = run_oracle_shift(
        list(pools.values()), names, outcomes, cfg, cost_fn, family=args.family
    )
    out = result_path(
        cfg["paths"]["results"],
        (
            f"oracle-{'+'.join(args.benchmark)}-{'-'.join(names)}"
            f"-{cfg['controller'].get('score', 'agreement')}-{summary['family']}{args.tag}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_guaranteed(args: argparse.Namespace) -> int:
    from .config import enabled_tiers, load_experiment, load_tiers
    from .evaluate import make_cost_fn
    from .generate import load_pools
    from .guaranteed import run_guaranteed

    cfg = load_experiment(args.experiment)
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.score:
        cfg["controller"]["score"] = args.score
    enabled = {t.name: t for t in enabled_tiers(load_tiers(args.models))}
    tiers = [enabled[n] for n in args.tiers]
    names = [t.name for t in tiers]
    pools = {}
    for bench in args.benchmark:
        pools.update(load_pools(cfg["paths"]["pools"], bench, names))
    if not pools:
        print(f"no tasks with pools for all tiers {names}")
        return 1
    cost_fn = make_cost_fn(
        cfg["evaluation"]["cost_metric"],
        {t.name: t.params_b for t in tiers},
        cfg["controller"].get("score", "agreement"),
    )
    summary = run_guaranteed(
        list(pools.values()), names, cfg, cost_fn, lean_budget=args.lean_budget
    )
    out = result_path(
        cfg["paths"]["results"],
        (
            f"guaranteed-{'+'.join(args.benchmark)}-{'-'.join(names)}"
            f"-{cfg['controller'].get('score', 'agreement')}{args.tag}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .bootstrap import compare
    from .config import load_experiment

    cfg = load_experiment(args.experiment)
    runs = [find_result(cfg["paths"]["results"], r) for r in (args.run_a, args.run_b)]
    alphas = args.alphas or cfg["evaluation"]["alphas"]
    out = compare(runs[0], runs[1], cfg, alphas, n_boot=args.boot, n_splits=args.splits)
    for row in out["alphas"]:
        lo, hi = row["solved_diff_ci"] or [float("nan")] * 2
        clo, chi = row["cost_ratio_ci"] or [float("nan")] * 2
        print(
            f"a={row['alpha']:g}: solved {row['solved_a'] * 100:.1f}% vs {row['solved_b'] * 100:.1f}%"
            f" (diff 95% CI {lo * 100:+.1f} to {hi * 100:+.1f} points),"
            f" cost ratio {row['cost_a'] / row['cost_b']:.2f} (CI {clo:.2f} to {chi:.2f})"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


def _cmd_economics(args: argparse.Namespace) -> int:
    from .config import load_experiment
    from .economics import run_economics
    from .generate import load_pools

    cfg = load_experiment(args.experiment)
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.abstain_factor is not None:
        cfg["evaluation"]["abstain_factor"] = args.abstain_factor
    if args.score:
        cfg["controller"]["score"] = args.score
    if args.lower_budget is not None:
        cfg["controller"]["lower_budget"] = args.lower_budget
    pools = {}
    for bench in args.benchmark:
        pools.update(load_pools(cfg["paths"]["pools"], bench, args.tiers))
    if not pools:
        print(f"no tasks with pools for all tiers {args.tiers}")
        return 1
    summary = run_economics(list(pools.values()), args.tiers, cfg, args.ratios)
    out = result_path(
        cfg["paths"]["results"],
        (
            f"economics-{'+'.join(args.benchmark)}-{'-'.join(args.tiers)}"
            f"-{cfg['controller'].get('score', 'agreement')}{args.tag}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["break_even"], indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_actions(args: argparse.Namespace) -> int:
    from .actions import run_actions
    from .config import enabled_tiers, load_experiment, load_tiers
    from .evaluate import make_cost_fn
    from .generate import load_pools

    cfg = load_experiment(args.experiment)
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.abstain_factor is not None:
        cfg["evaluation"]["abstain_factor"] = args.abstain_factor
    if args.score:
        cfg["controller"]["score"] = args.score
    enabled = {t.name: t for t in enabled_tiers(load_tiers(args.models))}
    tiers = [enabled[n] for n in args.tiers]
    names = [t.name for t in tiers]
    pools = {}
    for bench in args.benchmark:
        pools.update(load_pools(cfg["paths"]["pools"], bench, names))
    if not pools:
        print(f"no tasks with pools for all tiers {names}")
        return 1
    cost_fn = make_cost_fn(
        cfg["evaluation"]["cost_metric"],
        {t.name: t.params_b for t in tiers},
        cfg["controller"].get("score", "agreement"),
    )
    summary = run_actions(
        list(pools.values()),
        names,
        cfg,
        cost_fn,
        train_frac=args.train_frac,
        seeds=args.seeds,
        family=args.family,
    )
    out = result_path(
        cfg["paths"]["results"],
        (
            f"actions-{'+'.join(args.benchmark)}-{'-'.join(names)}"
            f"-{cfg['controller'].get('score', 'agreement')}-{args.family}{args.tag}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean"], indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from .config import enabled_tiers, load_experiment, load_tiers
    from .evaluate import make_cost_fn
    from .generate import load_pools
    from .router import run_router
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    if args.alphas:
        cfg["evaluation"]["alphas"] = args.alphas
    if args.abstain_factor is not None:
        cfg["evaluation"]["abstain_factor"] = args.abstain_factor
    if args.score:
        cfg["controller"]["score"] = args.score
    enabled = {t.name: t for t in enabled_tiers(load_tiers(args.models))}
    tiers = [enabled[n] for n in args.tiers]
    names = [t.name for t in tiers]
    pools, tasks = {}, {}
    for bench in args.benchmark:
        bp = load_pools(cfg["paths"]["pools"], bench, names)
        by_id = {t.task_id: t for t in load_tasks(bench)}
        pools.update(bp)
        tasks.update({tid: by_id[tid] for tid in bp})
    if not pools:
        print(f"no tasks with pools for all tiers {names}")
        return 1
    cost_fn = make_cost_fn(
        cfg["evaluation"]["cost_metric"],
        {t.name: t.params_b for t in tiers},
        cfg["controller"].get("score", "agreement"),
    )
    ids = list(pools)
    summary = run_router(
        [pools[i] for i in ids],
        [tasks[i] for i in ids],
        names,
        cfg,
        cost_fn,
        router_frac=args.router_frac,
        router_seeds=args.router_seeds,
        with_subsets=args.subsets,
    )
    out = result_path(
        cfg["paths"]["results"],
        (
            f"route-{'+'.join(args.benchmark)}-{'-'.join(names)}"
            f"-{cfg['controller'].get('score', 'agreement')}{args.tag}"
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean"], indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_annotate_ptrue(args: argparse.Namespace) -> int:
    from .config import load_experiment, load_tiers
    from .generate import annotate_ptrue
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    cache_path = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-{args.benchmark}-ptrue.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache_path))
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    n = annotate_ptrue(
        cfg["paths"]["pools"], args.benchmark, tier.name, client, tasks, args.workers
    )
    print(f"{tier.name}/{args.benchmark}: P(True) for {n} candidates")
    return 0


def _cmd_annotate_confidence(args: argparse.Namespace) -> int:
    from .config import load_experiment, load_tiers
    from .generate import annotate_confidence
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    cache_path = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-{args.benchmark}-conf.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache_path))
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    over = (cfg.get("tier_overrides") or {}).get(tier.name, {})
    n = annotate_confidence(
        cfg["paths"]["pools"],
        args.benchmark,
        tier.name,
        client,
        tasks,
        args.workers,
        over.get("conf_max_tokens", 8),
        prefill=bool(over.get("conf_prefill")),
        redo=args.redo,
    )
    print(f"{tier.name}/{args.benchmark}: annotated {n} candidates")
    return 0


def _cmd_add_probes(args: argparse.Namespace) -> int:
    from .config import load_experiment
    from .generate import add_stdin_probes
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    n = add_stdin_probes(
        cfg["paths"]["pools"],
        args.benchmark,
        args.tier,
        tasks,
        cfg["probes"],
        cfg["execution"],
        args.workers,
        stress_root=cfg["paths"].get("stress", "data/lcb/stress") if args.stress else None,
    )
    print(f"{args.tier}/{args.benchmark}: probe signatures for {n} stdin candidates")
    return 0


def _cmd_baseline_gen(args: argparse.Namespace) -> int:
    """Generation passes for the Model Cascading for Code and self-escalation baselines."""
    from .baselines_gen import make_selftests, make_traces
    from .config import load_experiment, load_tiers
    from .live import generation_settings
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    gen = generation_settings(cfg, args.benchmark, args.tier)  # benchmark and tier overrides
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    kind = "selftests" if args.cmd == "make-selftests" else "traces"
    cache = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-{args.benchmark}-{kind}.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache))
    tasks = load_tasks(args.benchmark, args.limit)
    root = cfg["paths"].get("baselines", "data/baselines")
    if kind == "selftests":
        counts = make_selftests(tasks, client, root, tier.name, workers=args.workers)
    else:
        counts = make_traces(
            tasks, client, root, tier.name, max_tokens=gen["max_tokens"], workers=args.workers
        )
    print(f"{kind} by {tier.name} on {args.benchmark}: {counts}")
    return 0


def _cmd_baseline_exec(args: argparse.Namespace) -> int:
    """CPU: run generated self-tests against pool samples and label greedy traces."""
    from .baselines import exec_selftests, label_traces
    from .config import load_experiment
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    root = cfg["paths"].get("baselines", "data/baselines")
    n1 = exec_selftests(
        cfg["paths"]["pools"], root, args.benchmark, args.tier, workers=args.workers
    )
    tasks = {t.task_id: t for t in load_tasks(args.benchmark)}
    n2 = label_traces(root, args.benchmark, args.tier, tasks, cfg["execution"], args.workers)
    print(f"{args.tier}/{args.benchmark}: self-tests run for {n1} tasks, {n2} traces labelled")
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    """Replay Model Cascading for Code and Bayesian self-escalation on the same tasks."""
    import numpy as np

    from .baselines import bse_eval, mcc_matrices, split_eval
    from .config import load_experiment, load_tiers
    from .generate import load_pools

    cfg = load_experiment(args.experiment)
    ev = cfg["evaluation"]
    alphas = args.alphas or ev["alphas"]
    root = Path(cfg["paths"].get("baselines", "data/baselines"))
    params = {t.name: t.params_b for t in load_tiers(args.models)}
    p = [params[t] for t in args.tiers]

    def read(kind: str, bench: str, tier: str, tid: str):
        f = root / kind / bench / tier / f"{tid.replace('/', '__')}.json"
        return json.loads(f.read_text()) if f.exists() else None

    pools, execs, traces = {}, {}, {}
    for bench in args.benchmark:
        for tid, ps in load_pools(cfg["paths"]["pools"], bench, args.tiers).items():
            ex = [read("selftest_exec", bench, t, tid) for t in args.tiers]
            if all(ex):
                pools[tid], execs[tid] = ps, ex
        tids = {
            f.stem.replace("__", "/")
            for f in (root / "traces" / bench / args.tiers[0]).glob("*.json")
        }
        for tid in tids:
            tr = [read("traces", bench, t, tid) for t in args.tiers]
            if all(x and "hidden_pass" in x for x in tr):
                traces[tid] = tr
    out: dict[str, Any] = {"tiers": args.tiers, "benchmarks": args.benchmark}
    if execs:
        ids = sorted(execs)
        settings, loss, cost, corr = mcc_matrices(ids, pools, execs, p)
        best = int(corr.mean(axis=1).argmax())  # most accurate setting, no risk target
        out["mcc"] = {"n_tasks": len(ids), "calibrated_tuned": split_eval(
            loss, cost, corr, alphas, ev["n_splits"], ev["calib_frac"]),
            "most_accurate": {"setting": settings[best], "solved": float(corr[best].mean()),
                              "risk": float(loss[best].mean()), "cost": float(cost[best].mean())}}  # fmt: skip
    if traces:
        ids = sorted(traces)
        out["bse"] = {"n_tasks": len(ids), "calibrated_tuned": bse_eval(
            ids, traces, p, alphas, n_splits=ev["n_splits"], calib_frac=ev["calib_frac"])}  # fmt: skip
        # Greedy answers of each tier alone (the two ends of the self-escalation frontier).
        out["bse"]["greedy_per_tier"] = {
            t: {"solved": float(np.mean([traces[i][k]["hidden_pass"] for i in ids])),
                "cost": float(np.mean([(traces[i][k]["tokens_out"] + 0.1 * traces[i][k]["tokens_in"]) * p[k] for i in ids]))}
            for k, t in enumerate(args.tiers)
        }  # fmt: skip
    dst = result_path(
        cfg["paths"]["results"], f"baselines-{'+'.join(args.benchmark)}-{'-'.join(args.tiers)}.json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    for name in ("mcc", "bse"):
        if name in out:
            print(f"== {name} ({out[name]['n_tasks']} tasks)")
            extra = out[name].get("most_accurate") or out[name].get("greedy_per_tier")
            print(f"  untuned: {extra}")
            for r in out[name]["calibrated_tuned"]:
                print(f"  alpha {r['alpha']:.2f}: tuned in {r['tuned_splits']} splits, risk {r['mean_test_risk']:.3f}, "
                      f"true risk > alpha in {r['violation_rate']:.0%}, solved {r['mean_solved']:.3f}, cost {r['mean_test_cost']:.0f}")  # fmt: skip
    print(f"wrote {dst}")
    return 0


def _cmd_make_stress(args: argparse.Namespace) -> int:
    from .config import load_experiment, load_tiers
    from .llm import OpenAIChatClient, ResponseCache
    from .stress import make_stress_inputs
    from .tasks import load_tasks

    cfg = load_experiment(args.experiment)
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    cache_path = Path(cfg["paths"]["llm_cache"]) / f"{tier.name}-{args.benchmark}-stress.sqlite"
    client = OpenAIChatClient(tier, ResponseCache(cache_path))
    tasks = load_tasks(args.benchmark, args.limit)
    root = cfg["paths"].get("stress", "data/lcb/stress")
    counts = make_stress_inputs(
        tasks, client, root, tier.model, args.generators, args.keep, workers=args.workers
    )
    print(f"stress inputs by {tier.name} in {root}: {counts}")
    return 0


def _shift(args, cfg, pools, names, cost_fn, out) -> int:
    """E5: temporal shift on benchmarks with release dates (LiveCodeBench)."""
    from .controller import Policy, learn_action_table
    from .evaluate import replay, run_shift, threshold_grid
    from .tasks import load_tasks

    dates = {}
    for bench in args.benchmark:
        dates.update({t.task_id: t.release_date for t in load_tasks(bench)})
    ids = [tid for tid in pools if dates.get(tid)]
    if not ids:
        print("no release dates for these tasks (temporal shift needs LiveCodeBench)")
        return 1
    ctl, ev = cfg["controller"], cfg["evaluation"]
    task_pools = [pools[t] for t in ids]
    table = learn_action_table(task_pools) if ctl["action"] == "adaptive" else {}
    grid = threshold_grid(**ev["lambda_grid"])
    pols = [
        Policy(
            [lam] * len(names),
            ctl["n_init"],
            ctl["max_draws"],
            ctl["action"],
            table,
            score=ctl.get("score", "agreement"),
        )  # fmt: skip
        for lam in grid
    ]
    rep = replay(task_pools, pols, int(ctl.get("permutations", 1)), cost_fn)
    results = [
        run_shift(rep, [dates[t] for t in ids], a, ev["delta"], args.shift) for a in ev["alphas"]
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "shift.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(
            f"alpha={r['alpha']}: windows above alpha  fixed {r['fixed_violations']}  sliding {r['sliding_violations']}"
        )
    print(f"wrote {out}/shift.json")
    return 0


def _cmd_energy_log(args: argparse.Namespace) -> int:
    from .energy import log_energy

    log_energy(args.out, args.interval, args.duration)
    return 0


def _cmd_attach_energy(args: argparse.Namespace) -> int:
    """Attribute a run's measured GPU energy to the candidates generated during that run.

    Only candidates whose request started inside [start, end] count, so pools from earlier
    (resumed) runs keep their own energy figures. Cached responses (no new GPU work) are
    attributed zero energy.
    """
    from .energy import joules_per_token
    from .types import TierPool

    root = Path(args.pools) / args.benchmark / args.tier
    files = sorted(root.glob("*.json"))
    pools = [TierPool.from_dict(json.loads(f.read_text())) for f in files]

    def in_run(c) -> bool:
        t = c.extra.get("started_at") or 0.0
        return args.start <= t <= args.end

    def cands(p):
        return p.samples + [r for ch in p.repairs.values() for r in ch]

    tokens = sum(c.tokens_out for p in pools for c in cands(p) if in_run(c))
    if tokens == 0:
        print(f"{args.tier}: no candidates generated in this window; nothing to attach")
        return 0
    jpt = joules_per_token(args.log, args.start, args.end, tokens)
    touched = 0
    for f, p in zip(files, pools, strict=True):
        changed = False
        for c in cands(p):
            if in_run(c):
                c.extra["energy_j"] = c.tokens_out * jpt
                c.extra["joules_per_token"] = jpt
                changed = True
        if changed:
            f.write_text(json.dumps(p.to_dict()))
            touched += 1
    print(
        f"{args.tier}: {jpt:.4f} J per output token over {tokens} tokens; updated {touched} pools"
    )
    return 0


def _command_help() -> dict[str, str]:
    """One-line help per command, from the module docstring (``codeladder -h`` lists them)."""
    out = {}
    for ln in (__doc__ or "").split("Commands", 1)[1].splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def build_parser() -> argparse.ArgumentParser:
    """The full `codeladder` command line (also used to check the experiment batches)."""
    from .config import REPO_ROOT
    from .scoring import SCORES

    p = argparse.ArgumentParser(prog="codeladder")
    p.add_argument("--models", default=str(REPO_ROOT / "configs" / "models.yaml"))
    p.add_argument("--experiment", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    helps = _command_help()

    def cmd(name: str, **kw: Any) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=helps.get(name), **kw)

    cmd("check-env")

    g = cmd("generate")
    g.add_argument("--tier", required=True)
    g.add_argument(
        "--benchmark",
        required=True,
        choices=["toy", "humaneval", "mbpp", "livecodebench", "bigcodebench"],
    )
    g.add_argument("--limit", type=int)
    g.add_argument("--task-ids", help="JSON list of task ids to generate (a fixed subset)")
    g.add_argument("--n-samples", type=int, help="override generation.n_samples")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument(
        "--extend",
        action="store_true",
        help="top up existing pools to the current settings (more samples, longer repair chains)",
    )

    e = cmd("evaluate")
    e.add_argument("--benchmark", required=True, nargs="+", help="one or more (pooled)")
    e.add_argument("--tiers", nargs="*", help="subset of tier names (default: all enabled)")
    e.add_argument("--action", choices=["resample", "repair", "adaptive"])
    e.add_argument(
        "--score",
        choices=SCORES,
    )
    e.add_argument("--alphas", nargs="+", type=float)
    e.add_argument("--tag", default="", help="suffix for the results folder")
    e.add_argument("--shift", type=int, default=0, help="E5: number of time windows (needs dates)")
    e.add_argument("--family", choices=["shared", "two_level", "subsets", "singles"])
    e.add_argument(
        "--abstain-factor",
        type=float,
        help="price of abstaining when choosing among certified settings (x costliest setting)",
    )
    e.add_argument("--max-samples", type=int, default=0, help="ablation: use the first N samples")
    e.add_argument(
        "--calib-sizes", type=int, nargs="*", help="X5: also calibrate on this many tasks"
    )
    e.add_argument(
        "--calib-on", nargs="*", help="X6: calibrate on these benchmarks, test on the others"
    )
    e.add_argument("--lower-budget", type=int, help="draws allowed below the last ladder tier")
    e.add_argument(
        "--method", choices=["paths", "pareto"], help="calibration for multi-setting families"
    )
    e.add_argument(
        "--group-by",
        choices=["benchmark", "difficulty"],
        default="benchmark",
        help="groups for per-group (Mondrian) calibration and conditional risk",
    )
    e.add_argument(
        "--cost-metric", choices=["weighted_tokens", "tokens", "energy_j", "gpu_s"], help="E6"
    )
    e.add_argument(
        "--max-repair-depth", type=int, default=0, help="ablation: cut repair chains to N rounds"
    )
    e.add_argument("--task-ids", help="JSON list of task ids: evaluate only these tasks")

    ac = cmd("annotate-confidence")
    ac.add_argument("--tier", required=True)
    ac.add_argument("--benchmark", required=True)
    ac.add_argument("--workers", type=int, default=8)
    ac.add_argument(
        "--redo", action="store_true", help="re-ask ratings made without the tier's prefill"
    )

    pt = cmd("annotate-ptrue")
    pt.add_argument("--tier", required=True)
    pt.add_argument("--benchmark", required=True)
    pt.add_argument("--workers", type=int, default=8)

    ap = cmd("add-probes")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--stress", action="store_true", help="also run the stress inputs (make-stress)"
    )

    ms = cmd("make-stress")
    ms.add_argument("--tier", required=True, help="model that writes the input generators")
    ms.add_argument("--benchmark", default="livecodebench")
    ms.add_argument("--workers", type=int, default=8)
    ms.add_argument("--generators", type=int, default=3)
    ms.add_argument("--keep", type=int, default=2)
    ms.add_argument("--limit", type=int)

    for name in ("make-selftests", "make-traces"):
        bp = cmd(name)
        bp.add_argument("--tier", required=True)
        bp.add_argument("--benchmark", required=True)
        bp.add_argument("--workers", type=int, default=8)
        bp.add_argument("--limit", type=int)

    bx = cmd("baseline-exec")
    bx.add_argument("--tier", required=True)
    bx.add_argument("--benchmark", required=True)
    bx.add_argument("--workers", type=int, default=4)

    bl = cmd("baselines")
    bl.add_argument("--benchmark", required=True, nargs="+")
    bl.add_argument("--tiers", nargs="+", required=True)
    bl.add_argument("--alphas", nargs="+", type=float)

    dm = cmd("demo")
    dm.add_argument("--port", type=int, default=8765)
    dm.add_argument("--host", default="127.0.0.1")
    dm.add_argument("--bundle", help="folder made by demo-bundle (default data/demo)")

    db = cmd("demo-bundle")
    db.add_argument("--out", default="data/demo")
    db.add_argument(
        "--benchmark", nargs="+", default=["humaneval", "mbpp", "livecodebench", "bigcodebench"]
    )

    xp = cmd("export-policy")
    xp.add_argument("run", help="evaluate result folder, or its name (found under results/)")
    xp.add_argument("--alpha", type=float, required=True)
    xp.add_argument("--out", required=True)

    xs = cmd("solve")
    xs.add_argument("--policy", required=True, help="JSON from export-policy")
    xs.add_argument(
        "--benchmark",
        default="humaneval",
        choices=["humaneval", "mbpp", "livecodebench", "bigcodebench"],
    )
    xs.add_argument("--task-id", help="benchmark task id (e.g. HumanEval/0)")
    xs.add_argument("--problem", help="custom problem JSON: prompt, entry_point, tests")
    xs.add_argument("--replay", action="store_true", help="decisions from cached pools, no models")
    xs.add_argument("--perm", type=int, default=0, help="replay: sample order")

    bc = cmd("bcb-check")
    bc.add_argument("--workers", type=int, default=8)
    bc.add_argument("--limit", type=int)

    bp2 = cmd("make-bcb-probes")
    bp2.add_argument("--tier", required=True, help="model that writes the probes")
    bp2.add_argument("--workers", type=int, default=8)

    ho = cmd("hidden-outcomes")
    ho.add_argument("--tier", required=True)
    ho.add_argument("--benchmark", required=True)
    ho.add_argument("--workers", type=int, default=8)

    xo = cmd("oracle-shift")
    xo.add_argument("--benchmark", required=True, nargs="+")
    xo.add_argument("--tiers", nargs="+", required=True)
    xo.add_argument("--alphas", nargs="+", type=float)
    xo.add_argument("--abstain-factor", type=float)
    xo.add_argument(
        "--score",
        choices=SCORES,
    )
    xo.add_argument("--family", choices=["shared", "singles", "subsets"], default="singles")
    xo.add_argument("--tag", default="")

    xg = cmd("guaranteed")
    xg.add_argument("--benchmark", required=True, nargs="+")
    xg.add_argument("--tiers", nargs="+", required=True, help="cheapest first")
    xg.add_argument("--alphas", nargs="+", type=float)
    xg.add_argument(
        "--score",
        choices=SCORES,
    )
    xg.add_argument("--lean-budget", type=int, default=1, help="draws per lower rung (toe-lean)")
    xg.add_argument("--tag", default="")

    xc = cmd("compare")
    xc.add_argument("run_a", help="result folder, or its name (found under results/)")
    xc.add_argument("run_b")
    xc.add_argument("--alphas", nargs="+", type=float)
    xc.add_argument("--boot", type=int, default=200)
    xc.add_argument("--splits", type=int, default=20, help="calibration splits per resample")
    xc.add_argument("--out", help="write the comparison as JSON")

    xe = cmd("economics")
    xe.add_argument("--benchmark", required=True, nargs="+")
    xe.add_argument("--tiers", nargs="+", required=True, help="cheapest first")
    xe.add_argument("--alphas", nargs="+", type=float)
    xe.add_argument("--abstain-factor", type=float)
    xe.add_argument(
        "--score",
        choices=SCORES,
    )
    xe.add_argument(
        "--ratios", nargs="+", type=float, default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="price of the top tier per token, relative to the cheapest tier",
    )  # fmt: skip
    xe.add_argument("--lower-budget", type=int, help="draws allowed below the last ladder tier")
    xe.add_argument("--tag", default="")

    xa = cmd("actions")
    xa.add_argument("--benchmark", required=True, nargs="+")
    xa.add_argument("--tiers", nargs="+", required=True, help="cheapest first")
    xa.add_argument("--alphas", nargs="+", type=float)
    xa.add_argument("--abstain-factor", type=float)
    xa.add_argument(
        "--score",
        choices=SCORES,
    )
    xa.add_argument("--family", choices=["shared", "singles", "subsets"], default="singles")
    xa.add_argument("--train-frac", type=float, default=0.3, help="tasks used to learn it")
    xa.add_argument("--seeds", type=int, default=3)
    xa.add_argument("--tag", default="")

    rt = cmd("route")
    rt.add_argument("--benchmark", required=True, nargs="+")
    rt.add_argument("--tiers", nargs="+", required=True, help="cheapest first")
    rt.add_argument("--alphas", nargs="+", type=float)
    rt.add_argument("--abstain-factor", type=float)
    rt.add_argument(
        "--score",
        choices=SCORES,
    )
    rt.add_argument("--router-frac", type=float, default=0.3, help="tasks used to train it")
    rt.add_argument("--router-seeds", type=int, default=3)
    rt.add_argument("--subsets", action="store_true", help="also report sub-ladders")
    rt.add_argument("--tag", default="")

    el = cmd("energy-log")
    el.add_argument("--out", required=True)
    el.add_argument("--interval", type=float, default=1.0)
    el.add_argument("--duration", type=float)

    ae = cmd("attach-energy")
    ae.add_argument("--pools", required=True)
    ae.add_argument("--benchmark", required=True)
    ae.add_argument("--tier", required=True)
    ae.add_argument("--log", required=True)
    ae.add_argument(
        "--start", type=float, required=True, help="run start (unix time, see runs.jsonl)"
    )
    ae.add_argument("--end", type=float, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    handlers = {
        "check-env": _cmd_check_env,
        "generate": _cmd_generate,
        "evaluate": _cmd_evaluate,
        "annotate-confidence": _cmd_annotate_confidence,
        "annotate-ptrue": _cmd_annotate_ptrue,
        "add-probes": _cmd_add_probes,
        "make-stress": _cmd_make_stress,
        "make-selftests": _cmd_baseline_gen,
        "make-traces": _cmd_baseline_gen,
        "baseline-exec": _cmd_baseline_exec,
        "baselines": _cmd_baselines,
        "route": _cmd_route,
        "actions": _cmd_actions,
        "economics": _cmd_economics,
        "compare": _cmd_compare,
        "guaranteed": _cmd_guaranteed,
        "hidden-outcomes": _cmd_hidden_outcomes,
        "bcb-check": _cmd_bcb_check,
        "make-bcb-probes": _cmd_make_bcb_probes,
        "oracle-shift": _cmd_oracle_shift,
        "export-policy": _cmd_export_policy,
        "demo": _cmd_demo,
        "demo-bundle": _cmd_demo_bundle,
        "solve": _cmd_solve,
        "energy-log": _cmd_energy_log,
        "attach-energy": _cmd_attach_energy,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
