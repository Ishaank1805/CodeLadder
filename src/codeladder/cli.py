"""Command-line entry point: ``codeladder <command>``.

Commands
  check-env     print Python/package versions and test each enabled tier's endpoint
  generate      build cached candidate pools for one tier and benchmark
  evaluate      replay the controller over pools and run Learn-then-Test splits
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

from . import __version__


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
    from .llm import OpenAIChatClient, ResponseCache
    from .tasks import attach_expected_outputs, load_tasks

    cfg = load_experiment(args.experiment)
    if args.n_samples:
        cfg["generation"]["n_samples"] = args.n_samples
    tier = next(t for t in load_tiers(args.models) if t.name == args.tier)
    client = OpenAIChatClient(tier, ResponseCache(cfg["paths"]["llm_cache"]))
    tasks = [attach_expected_outputs(t) for t in load_tasks(args.benchmark, args.limit)]
    t0 = time.time()
    written = generate_pools(
        tasks, tier.name, client, cfg, workers=args.workers, overwrite=args.overwrite
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
    from .generate import load_pools

    cfg = load_experiment(args.experiment)
    if args.action:
        cfg["controller"]["action"] = args.action
    tiers = [
        t for t in enabled_tiers(load_tiers(args.models)) if not args.tiers or t.name in args.tiers
    ]
    names = [t.name for t in tiers]
    pools = load_pools(cfg["paths"]["pools"], args.benchmark, names)
    if not pools:
        print(
            f"no tasks with pools for all tiers {names} in {cfg['paths']['pools']}/{args.benchmark}"
        )
        return 1
    cost_fn = make_cost_fn(cfg["evaluation"]["cost_metric"], {t.name: t.params_b for t in tiers})
    out = (
        Path(cfg["paths"]["results"])
        / f"{args.benchmark}-{'-'.join(names)}-{cfg['controller']['action']}"
    )
    summary = run_experiment(list(pools.values()), names, cfg, cost_fn, out)
    brief = {k: v for k, v in summary.items() if k != "calibrated"}
    brief["calibrated"] = [
        {k: v for k, v in c.items() if k != "splits"} for c in summary["calibrated"]
    ]
    print(json.dumps(brief, indent=2))
    print(f"wrote {out}/summary.json")
    return 0


def _cmd_energy_log(args: argparse.Namespace) -> int:
    from .energy import log_energy

    log_energy(args.out, args.interval, args.duration)
    return 0


def _cmd_attach_energy(args: argparse.Namespace) -> int:
    from .energy import joules_per_token
    from .types import TierPool

    root = Path(args.pools) / args.benchmark / args.tier
    files = sorted(root.glob("*.json"))
    pools = [TierPool.from_dict(json.loads(f.read_text())) for f in files]
    tokens = sum(
        c.tokens_out
        for p in pools
        for c in p.samples + [r for ch in p.repairs.values() for r in ch]
    )
    jpt = joules_per_token(args.log, args.start, args.end, tokens)
    for f, p in zip(files, pools, strict=True):
        for c in p.samples + [r for ch in p.repairs.values() for r in ch]:
            c.extra["energy_j"] = c.tokens_out * jpt
        f.write_text(json.dumps(p.to_dict()))
    print(
        f"{args.tier}: {jpt:.4f} J per output token over {tokens} tokens; updated {len(files)} pools"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from .config import REPO_ROOT

    p = argparse.ArgumentParser(prog="codeladder")
    p.add_argument("--models", default=str(REPO_ROOT / "configs" / "models.yaml"))
    p.add_argument("--experiment", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-env")

    g = sub.add_parser("generate")
    g.add_argument("--tier", required=True)
    g.add_argument(
        "--benchmark", required=True, choices=["toy", "humaneval", "mbpp", "livecodebench"]
    )
    g.add_argument("--limit", type=int)
    g.add_argument("--n-samples", type=int, help="override generation.n_samples")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("evaluate")
    e.add_argument("--benchmark", required=True)
    e.add_argument("--tiers", nargs="*", help="subset of tier names (default: all enabled)")
    e.add_argument("--action", choices=["resample", "repair", "adaptive"])

    el = sub.add_parser("energy-log")
    el.add_argument("--out", required=True)
    el.add_argument("--interval", type=float, default=1.0)
    el.add_argument("--duration", type=float)

    ae = sub.add_parser("attach-energy")
    ae.add_argument("--pools", required=True)
    ae.add_argument("--benchmark", required=True)
    ae.add_argument("--tier", required=True)
    ae.add_argument("--log", required=True)
    ae.add_argument(
        "--start", type=float, required=True, help="run start (unix time, see runs.jsonl)"
    )
    ae.add_argument("--end", type=float, required=True)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    handlers = {
        "check-env": _cmd_check_env,
        "generate": _cmd_generate,
        "evaluate": _cmd_evaluate,
        "energy-log": _cmd_energy_log,
        "attach-energy": _cmd_attach_energy,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
