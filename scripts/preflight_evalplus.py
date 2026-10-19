"""Preflight: check that our labels agree with the official EvalPlus evaluator.

For every task we judge the reference solution and up to 3 AST mutants (small bugs
such as + -> -, < -> <=, constant + 1) with our harness, then run
``evalplus.evaluate`` on the same programs and compare pass/fail verdicts.

    python scripts/preflight_evalplus.py humaneval --out results/preflight --workers 8
    python scripts/preflight_evalplus.py mbpp --out results/preflight --workers 8

Needs ``pip install evalplus``. Uses up to ~4 GB per worker on a few tasks, so run
it on Ada (scripts/ada/preflight.sbatch) rather than a small laptop.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from codeladder.generate import evaluate_code
from codeladder.tasks import attach_expected_outputs, load_tasks

EXEC = {
    "per_call_timeout_s": 2.0,
    "total_timeout_cap_s": 60,
    "hidden_chunk": 250,
    "ref_time_factor": 4.0,
    "memory_mb": 2048,
    "memory_mb_hidden": 4096,
}
SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
    ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Mult: ast.FloorDiv,
}  # fmt: skip


def mutants(code: str, k: int, seed: int) -> list[str]:
    tree = ast.parse(code)
    sites = [
        n for n in ast.walk(tree)
        if (isinstance(n, ast.BinOp) and type(n.op) in SWAP)
        or (isinstance(n, ast.Compare) and type(n.ops[0]) in SWAP)
        or (isinstance(n, ast.Constant) and type(n.value) is int)
    ]  # fmt: skip
    random.Random(seed).shuffle(sites)
    out = []
    for node in sites[:k]:
        t = ast.parse(code)
        target = next(n for n in ast.walk(t) if ast.dump(n) == ast.dump(node))
        if isinstance(target, ast.BinOp):
            target.op = SWAP[type(target.op)]()
        elif isinstance(target, ast.Compare):
            target.ops[0] = SWAP[type(target.ops[0])]()
        else:
            target.value = target.value + 1
        out.append(ast.unparse(t))
    return out


def judge(task) -> list[dict]:
    try:
        task = attach_expected_outputs(task)
    except Exception as e:
        return [{"task_id": task.task_id, "variant": "ref", "harness_error": repr(e)[:200]}]
    rows = []
    variants = [("ref", task.reference)] + [
        (f"m{i}", m) for i, m in enumerate(mutants(task.reference, 3, 0))
    ]
    for name, code in variants:
        try:
            r = evaluate_code(task, code, [], EXEC)
            rows.append(
                {"task_id": task.task_id, "variant": name, "solution": code,
                 "ours_base": r["visible_pass_rate"] >= 1.0, "ours_plus": r["hidden_pass"]}
            )  # fmt: skip
        except Exception as e:
            rows.append({"task_id": task.task_id, "variant": name, "harness_error": repr(e)[:200]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["humaneval", "mbpp"])
    ap.add_argument("--out", default="results/preflight")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(judge, load_tasks(args.benchmark)) for r in rs]
    errors = [r for r in rows if "harness_error" in r]
    rows = [r for r in rows if "harness_error" not in r]
    samples = out / f"{args.benchmark}_samples.jsonl"
    samples.write_text("".join(json.dumps({"task_id": r["task_id"], "solution": r["solution"]}) + "\n" for r in rows))
    print(f"judged {len(rows)} programs in {time.time() - t0:.0f}s; running evalplus.evaluate ...")

    subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", args.benchmark, "--samples", str(samples),
         "--parallel", str(args.workers), "--i_just_wanna_run"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    ev = json.loads((out / f"{args.benchmark}_samples_eval_results.json").read_text())["eval"]

    seen: dict[str, int] = defaultdict(int)
    diffs = []
    for r in rows:
        e = ev[r["task_id"]][seen[r["task_id"]]]
        seen[r["task_id"]] += 1
        base = e["base_status"] == "pass"
        plus = base and e["plus_status"] == "pass"
        if (base, plus) != (r["ours_base"], r["ours_plus"]):
            diffs.append({"task_id": r["task_id"], "variant": r["variant"],
                          "ours": [r["ours_base"], r["ours_plus"]], "evalplus": [base, plus]})  # fmt: skip
    summary = {
        "benchmark": args.benchmark,
        "programs": len(rows),
        "agreement": (len(rows) - len(diffs)) / max(len(rows), 1),
        "disagreements": diffs,
        "harness_errors": errors,
    }
    (out / f"{args.benchmark}_preflight.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "harness_errors"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
