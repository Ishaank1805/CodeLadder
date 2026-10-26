"""Offline replay of two published baselines on the same tasks and costs as CodeLadder, over the
same number of random calibration/test halves (drawn independently of CodeLadder's halves).

Model Cascading for Code (MCC, arXiv 2405.15842). Per model: k solutions (our pool samples,
temperature 0.8 as in the reference baseline) and k self-generated test cases of up to l assert lines.
Solution s scores n_s x n_t, where n_s is the number of valid test lines it passes and n_t the
largest number of solutions passing one of those lines; the best solution is returned if its
score is at least theta x k x N_t (N_t: valid test lines), otherwise the next model is asked.
The last model always answers. theta is tuned on validation data (here: the calibration half
of each split, cheapest theta whose empirical risk is at most alpha; no guarantee).

Bayesian self-escalation (BSE, arXiv 2608.24087, its real-model protocol). Per model one greedy
generation with per-token log-probability, entropy and top-2 margin. A logistic regression on
the running means of the three signals (fit on the calibration half) gives a belief B_t of
eventual success; the model aborts and escalates at the first token with B_t < tau, otherwise
its answer is returned. The last model always answers. tau is tuned like theta.

Neither baseline abstains, so risk = error rate of the returned answers.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .executor import _limits, run_isolated
from .types import Task, TierPool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- running generated asserts

_ASSERT_RUNNER = r"""
import json, pickle, signal, sys
sys.setrecursionlimit(10000)
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
p = pickle.loads(sys.stdin.buffer.read())
def alarm(*_):
    raise TimeoutError()
signal.signal(signal.SIGALRM, alarm)
g = {"__name__": "__main__"}
res = []
try:
    signal.alarm(int(p["timeout"]) + 1)
    exec(p["code"], g)
    signal.alarm(0)
    ok = True
except BaseException:
    ok = False
for line in p["lines"]:
    if not ok:
        res.append(False); continue
    try:
        signal.setitimer(signal.ITIMER_REAL, p["timeout"])
        exec(line, dict(g))
        signal.setitimer(signal.ITIMER_REAL, 0)
        res.append(True)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        res.append(False)
sys.stdout.write("@@RES@@" + json.dumps(res))
"""


def run_asserts(
    code: str, lines: list[str], timeout: float = 2.0, memory_mb: int = 2048
) -> list[bool]:
    """Which assert lines a program passes (each line runs in a copy of the program's namespace)."""
    if not lines:
        return []
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    total = timeout * (len(lines) + 1) + 5
    with tempfile.TemporaryDirectory(prefix="codeladder-mcc-") as tmp:
        try:
            proc = run_isolated(
                [sys.executable, "-s", "-B", "-c", _ASSERT_RUNNER],
                input=pickle.dumps({"code": code, "lines": lines, "timeout": timeout}),
                timeout=total, cwd=tmp, env=env,
                preexec_fn=_limits(memory_mb, int(total) + 1),
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return [False] * len(lines)
    out = proc.stdout.decode("utf-8", errors="replace")
    i = out.rfind("@@RES@@")
    if i < 0:
        return [False] * len(lines)
    try:
        res = json.loads(out[i + 7 :])
    except json.JSONDecodeError:
        return [False] * len(lines)
    return [bool(x) for x in res] if len(res) == len(lines) else [False] * len(lines)


def _bpath(root: str | Path, kind: str, bench: str, tier: str, task_id: str) -> Path:
    return Path(root) / kind / bench / tier / f"{task_id.replace('/', '__')}.json"


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def exec_selftests(
    pools_root: str | Path,
    root: str | Path,
    bench: str,
    tier: str,
    n_solutions: int = 8,
    workers: int = 4,
) -> int:
    """Pass matrix of the first ``n_solutions`` pool samples on every generated test line."""
    files = sorted((Path(root) / "selftests" / bench / tier).glob("*.json"))

    def work(f: Path) -> int:
        st = json.loads(f.read_text())
        out = _bpath(root, "selftest_exec", bench, tier, st["task_id"])
        if out.exists():
            return 0
        pf = Path(pools_root) / bench / tier / f.name
        if not pf.exists():
            return 0
        pool = TierPool.from_dict(json.loads(pf.read_text()))
        lines = [ln for t in st["tests"] for ln in t["lines"]]
        test_of = [i for i, t in enumerate(st["tests"]) for _ in t["lines"]]
        passed = [run_asserts(c.code, lines) for c in pool.samples[:n_solutions]]
        _write(out, {"task_id": st["task_id"], "test_of_line": test_of, "passed": passed,
                     "test_tokens": [[t["tokens_in"], t["tokens_out"]] for t in st["tests"]]})  # fmt: skip
        return 1

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(work, files))


def label_traces(
    root: str | Path,
    bench: str,
    tier: str,
    tasks: dict[str, Task],
    exec_cfg: dict,
    workers: int = 4,
) -> int:
    """Visible and hidden labels for the greedy trace programs."""
    from .generate import evaluate_code
    from .tasks import prepare_task, release_tests

    files = sorted((Path(root) / "traces" / bench / tier).glob("*.json"))

    def work(f: Path) -> int:
        tr = json.loads(f.read_text())
        if "hidden_pass" in tr:
            return 0
        task = tasks[tr["task_id"]]
        try:
            prepare_task(task)
            res = evaluate_code(task, tr["code"], [], exec_cfg)
        except Exception:  # one task must not stop the run (rerun to retry it)
            log.exception("could not label %s", tr["task_id"])
            return 0
        release_tests(task)
        tr["visible_pass"] = res["visible_pass_rate"] >= 1.0
        tr["hidden_pass"] = bool(res["hidden_pass"])
        _write(f, tr)
        return 1

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(work, files))


# ---------------------------------------------------------------- MCC replay


def mcc_tier_outcome(
    passed: list[list[bool]], test_of: list[int], order: list[int], k: int
) -> tuple[int, float]:
    """For one tier: (index of the chosen solution, normalised score n_s x n_t / (k x N_t)).

    Uses the first k solutions in ``order`` and the test lines of the first k test cases."""
    sols = order[:k]
    keep = [j for j, t in enumerate(test_of) if t < k]
    if not keep or not sols:
        return (sols[0] if sols else 0), 0.0
    m = np.array([[passed[s][j] for j in keep] for s in sols], dtype=bool)  # (k, lines)
    per_line = m.sum(axis=0)
    best, best_score = sols[0], -1.0
    for a, s in enumerate(sols):
        n_s = int(m[a].sum())
        n_t = int(per_line[m[a]].max()) if n_s else 0
        score = n_s * n_t
        if score > best_score:
            best, best_score = s, score
    return best, best_score / (len(sols) * len(keep))


# ---------------------------------------------------------------- BSE replay


def running_means(stats: list[list[float]]) -> np.ndarray:
    a = np.asarray(stats, dtype=float).reshape(-1, 3)
    if not len(a):
        return np.zeros((1, 3))
    return np.cumsum(a, axis=0) / np.arange(1, len(a) + 1)[:, None]


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton steps; returns weights incl. intercept."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = l2 * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + reg @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def belief(w: np.ndarray, rm: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.hstack([(rm - mu) / sd, np.ones((len(rm), 1))]) @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- shared split evaluation


def _tune(loss_cal: np.ndarray, cost_cal: np.ndarray, alpha: float) -> int | None:
    """Cheapest setting with empirical calibration risk at most alpha (no guarantee)."""
    ok = np.where(loss_cal.mean(axis=1) <= alpha)[0]
    if not len(ok):
        return None
    return int(ok[np.argmin(cost_cal[ok].mean(axis=1))])


def split_eval(loss: np.ndarray, cost: np.ndarray, correct: np.ndarray, alphas: list[float],
               n_splits: int = 100, calib_frac: float = 0.5, seed: int = 0) -> list[dict[str, Any]]:  # fmt: skip
    """loss/cost/correct: (settings, tasks); random calibration/test halves as in run_splits."""
    rng = np.random.default_rng(seed)
    m, n = loss.shape
    rows = {a: [] for a in alphas}
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = int(round(calib_frac * n))
        cal, test = idx[:half], idx[half:]
        for a in alphas:
            j = _tune(loss[:, cal], cost[:, cal], a)
            if j is None:
                continue
            rows[a].append(
                (
                    loss[j, test].mean(),
                    loss[j].mean() > a,
                    cost[j, test].mean(),
                    correct[j, test].mean(),
                )
            )
    out = []
    for a in alphas:
        r = np.array(rows[a]) if rows[a] else np.zeros((0, 4))
        out.append({
            "alpha": a, "tuned_splits": len(r),
            "mean_test_risk": float(r[:, 0].mean()) if len(r) else float("nan"),
            "violation_rate": float(r[:, 1].mean()) if len(r) else float("nan"),
            "mean_test_cost": float(r[:, 2].mean()) if len(r) else float("nan"),
            "mean_solved": float(r[:, 3].mean()) if len(r) else float("nan"),
            "mean_coverage": 1.0,
        })  # fmt: skip
    return out


def _wcost(tokens_in: float, tokens_out: float, params_b: float) -> float:
    return (tokens_out + 0.1 * tokens_in) * params_b


def mcc_matrices(
    task_ids: list[str], pools: dict[str, list[TierPool]], execs: dict[str, list[dict]],
    params: list[float], ks: tuple[int, ...] = (3, 5, 8), thetas: np.ndarray | None = None,
) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Loss, cost and correct for every (k, theta) setting and task (tiers cheapest first)."""
    thetas = np.round(np.arange(0.0, 1.0001, 0.05), 4) if thetas is None else thetas
    settings = [(k, float(t)) for k in ks for t in thetas]
    T = len(params)
    loss = np.zeros((len(settings), len(task_ids)))
    cost, correct = np.zeros_like(loss), np.zeros_like(loss)
    for i, tid in enumerate(task_ids):
        for kk in ks:
            per_tier = []
            for t in range(T):
                ex, pool = execs[tid][t], pools[tid][t]
                order = list(range(min(len(pool.samples), len(ex["passed"]))))
                best, score = mcc_tier_outcome(ex["passed"], ex["test_of_line"], order, kk)
                c = sum(_wcost(s.tokens_in, s.tokens_out, params[t]) for s in pool.samples[:kk])
                c += sum(_wcost(ti, to, params[t]) for ti, to in ex["test_tokens"][:kk])
                per_tier.append((score, c, bool(pool.samples[best].hidden_pass)))
            for j, (k2, th) in enumerate(settings):
                if k2 != kk:
                    continue
                total = 0.0
                for t, (score, c, ok) in enumerate(per_tier):
                    total += c
                    if score >= th or t == T - 1:
                        loss[j, i], correct[j, i] = (not ok), ok
                        break
                cost[j, i] = total
    return settings, loss, cost, correct


def bse_eval(
    task_ids: list[str], traces: dict[str, list[dict]], params: list[float], alphas: list[float],
    taus: np.ndarray | None = None, n_splits: int = 100, calib_frac: float = 0.5, seed: int = 0,
    stride: int = 4,
) -> list[dict[str, Any]]:  # fmt: skip
    """Streaming escalation with a per-split logistic belief per tier (last tier always answers)."""
    taus = np.round(np.arange(0.0, 1.0001, 0.05), 4) if taus is None else taus
    T, n = len(params), len(task_ids)
    rms = [[running_means(traces[tid][t]["token_stats"]) for tid in task_ids] for t in range(T)]
    ok = np.array([[bool(traces[tid][t]["hidden_pass"]) for tid in task_ids] for t in range(T)])
    tin = np.array([[traces[tid][t]["tokens_in"] for tid in task_ids] for t in range(T)], float)
    tout = np.array([[traces[tid][t]["tokens_out"] for tid in task_ids] for t in range(T)], float)
    rng = np.random.default_rng(seed)
    rows: dict[float, list] = {a: [] for a in alphas}
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = int(round(calib_frac * n))
        cal, test = idx[:half], idx[half:]
        # escalation token (or None) per tier, task and tau
        first_below = np.full((T, len(taus), n), -1, dtype=int)
        for t in range(T - 1):
            X = np.vstack([rms[t][i][::stride] for i in cal])
            y = np.concatenate([np.full(len(rms[t][i][::stride]), ok[t, i], float) for i in cal])
            mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
            w = fit_logistic((X - mu) / sd, y)
            for i in range(n):
                b = belief(w, rms[t][i], mu, sd)
                for a_, tau in enumerate(taus):
                    below = np.nonzero(b < tau)[0]
                    first_below[t, a_, i] = int(below[0]) if len(below) else -1
        loss = np.zeros((len(taus), n))
        cost, corr = np.zeros_like(loss), np.zeros_like(loss)
        for a_ in range(len(taus)):
            for i in range(n):
                total = 0.0
                for t in range(T):
                    fb = first_below[t, a_, i] if t < T - 1 else -1
                    if fb >= 0:  # abort after fb + 1 tokens and escalate
                        total += _wcost(tin[t, i], fb + 1, params[t])
                        continue
                    total += _wcost(tin[t, i], tout[t, i], params[t])
                    loss[a_, i], corr[a_, i] = (not ok[t, i]), ok[t, i]
                    break
                cost[a_, i] = total
        for a in alphas:
            j = _tune(loss[:, cal], cost[:, cal], a)
            if j is None:
                continue
            rows[a].append(
                (
                    loss[j, test].mean(),
                    loss[j].mean() > a,
                    cost[j, test].mean(),
                    corr[j, test].mean(),
                )
            )
    out = []
    for a in alphas:
        r = np.array(rows[a]) if rows[a] else np.zeros((0, 4))
        out.append({
            "alpha": a, "tuned_splits": len(r),
            "mean_test_risk": float(r[:, 0].mean()) if len(r) else float("nan"),
            "violation_rate": float(r[:, 1].mean()) if len(r) else float("nan"),
            "mean_test_cost": float(r[:, 2].mean()) if len(r) else float("nan"),
            "mean_solved": float(r[:, 3].mean()) if len(r) else float("nan"),
            "mean_coverage": 1.0,
        })  # fmt: skip
    return out
