"""X15: a calibrated CodeLadder policy as a tool.

``export_policy`` turns an evaluate run into a deployable policy: Learn-then-Test on all of
the run's tasks (each task once) certifies settings at the target α, the usual selection rule
(cost plus a price for abstaining) picks one, and the ladder (tiers, per-tier thresholds,
score, draw budgets, action) is written as JSON with its calibration statistics. The
guarantee then covers new tasks exchangeable with the calibration tasks.

``solve`` runs that policy online on one problem: it asks the tiers' models for candidates
one at a time (same prompts, seeds and checks as pool generation), rates them when the score
needs it, and accepts, repairs, resamples, escalates or abstains exactly as the offline
controller does. ``replay`` shows the same decisions from cached pools, without any model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import bonferroni_paths, learn_then_test
from .controller import Policy, _choose_action, run_task
from .evaluate import family_policies, selection_cost
from .generate import make_candidate, task_probes
from .prompts import (
    confidence_messages,
    initial_messages,
    p_true,
    parse_confidence,
    ptrue_messages,
    repair_messages,
)
from .scoring import checks_ok, phi_scores
from .types import Candidate, Task, TierPool

Log = Callable[[str], None]


# ---------------------------------------------------------------- export


def export_policy(run_dir: str | Path, cfg: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Certify on every task of an evaluate run and return the chosen ladder as a dict."""
    run = Path(run_dir)
    s = json.loads((run / "summary.json").read_text())
    z = np.load(run / "replay.npz")
    loss, cost, acc, cor = (z[k][:, :, 0] for k in ("loss", "cost", "accepted", "correct"))
    tiers, family = s["tiers"], s.get("family", "shared")
    lower = s.get("lower_budget")
    if lower is None and (m := re.search(r"-lb(\d+)", run.name)):
        lower = int(m.group(1))  # runs written before the summary recorded it
    ctl = {**cfg["controller"], "score": s.get("score", "agreement"), "lower_budget": lower}
    ctl["action"] = s.get("action", ctl["action"])
    ev = {**cfg["evaluation"], "abstain_factor": s.get("abstain_factor", 0.0)}
    fam = family if len(tiers) > 1 else "shared"
    configs, policies, method, paths = family_policies(fam, tiers, ctl, ev, {})
    if len(configs) != loss.shape[0]:
        raise ValueError(f"{run}: settings do not match the current configuration")
    sel = selection_cost(cost, acc, float(ev["abstain_factor"]))
    if fam == "shared":
        res = learn_then_test(loss, sel, alpha, ev["delta"])
    else:
        res = bonferroni_paths(loss, sel, alpha, ev["delta"], paths)
    if res.chosen is None:
        raise ValueError(f"nothing certified at alpha={alpha} on {run.name}")
    j = res.chosen
    pol = policies[j]
    return {
        "source_run": run.name,
        "alpha": alpha,
        "delta": ev["delta"],
        "calibration_tasks": int(loss.shape[1]),
        "setting": configs[j] if not isinstance(configs[j], float) else f"lambda={configs[j]}",
        "tiers": tiers,
        "thresholds": pol.thresholds,
        "score": pol.score,
        "action": pol.action,
        "n_init": pol.n_init,
        "max_draws": pol.max_draws,
        "lower_budget": pol.lower_budget,
        "calibration": {
            "risk": float(loss[j].mean()),
            "coverage": float(acc[j].mean()),
            "solved": float(cor[j].mean()),
            "mean_cost": float(cost[j].mean()),
            "certified_settings": len(res.valid),
        },
        "guarantee": (
            f"With probability at least {1 - ev['delta']:.2f} over the {loss.shape[1]} "
            f"calibration tasks, the share of new tasks (exchangeable with them) that end with "
            f"accepted but wrong code is at most {alpha}."
        ),
    }


def load_policy(path: str | Path) -> tuple[Policy, dict[str, Any]]:
    meta = json.loads(Path(path).read_text())
    pol = Policy(
        meta["thresholds"],
        meta["n_init"],
        meta["max_draws"],
        meta["action"],
        {},
        score=meta["score"],
        name=meta.get("setting", ""),
    )
    pol.lower_budget = meta.get("lower_budget")
    return pol, meta


# ---------------------------------------------------------------- online controller


def generation_settings(cfg: dict[str, Any], benchmark: str, tier: str) -> dict[str, Any]:
    gen = dict(cfg["generation"])
    for key, val in (cfg.get("generation_overrides") or {}).get(benchmark, {}).items():
        gen[key] = val
    for key, val in (cfg.get("tier_overrides") or {}).get(tier, {}).items():
        gen[key] = val
    return gen


def _rate(c: Candidate, task: Task, client, score: str, gen: dict[str, Any]) -> None:
    """Ask the candidate's model for the rating the score needs (visible-passing only)."""
    if score in ("selfconf", "agreement_selfconf") and c.extra.get("self_conf") is None:
        prefill = bool(gen.get("conf_prefill"))
        kw = {"continue_final": True} if prefill else {}
        comp = client.complete(
            confidence_messages(task, c.code, prefill=prefill),
            temperature=0.0, top_p=1.0, max_tokens=int(gen.get("conf_max_tokens", 8)), seed=0, **kw,
        )  # fmt: skip
        c.extra["self_conf"] = parse_confidence(comp.text, prefilled=prefill)
        c.extra["self_conf_tokens"] = comp.tokens_in + comp.tokens_out
    if score in ("ptrue", "agreement_ptrue") and c.extra.get("p_true") is None:
        comp = client.complete(
            ptrue_messages(task, c.code), temperature=0.0, top_p=1.0, max_tokens=1, seed=0,
            top_logprobs=10, continue_final=True,
        )  # fmt: skip
        c.extra["p_true"] = p_true(comp.first_top)
        c.extra["p_true_tokens"] = comp.tokens_in + comp.tokens_out


def solve(
    task: Task,
    policy: Policy,
    tiers: list[str],
    clients: dict[str, Any],
    cfg: dict[str, Any],
    log: Log = print,
) -> dict[str, Any]:
    """Run the policy online on one prepared task. Returns the accepted code (or None), the
    tier that answered, every candidate drawn and the decision trace."""
    probes = task_probes(task, cfg["probes"])
    stress: list[str] = []
    if task.mode == "stdin" and cfg["paths"].get("stress"):
        from .stress import load_stress_inputs

        stress = load_stress_inputs(cfg["paths"]["stress"], task)
    exec_cfg = cfg["execution"]
    last = max(i for i, th in enumerate(policy.thresholds) if th is not None)
    trace: list[dict[str, Any]] = []
    all_drawn: list[Candidate] = []
    for t, tier in enumerate(tiers):
        lam = policy.thresholds[t]
        if lam is None:
            continue
        gen = generation_settings(cfg, task.benchmark, tier)
        client = clients[tier]
        max_draws, n_init = policy.max_draws, policy.n_init
        if policy.lower_budget is not None and t < last:
            max_draws, n_init = policy.lower_budget, min(n_init, policy.lower_budget)
        sampling = {k: gen[k] for k in ("temperature", "top_p", "max_tokens")}
        base = int(gen.get("seed", 0)) * 100_003
        drawn: list[Candidate] = []
        next_idx = 0
        repairs: dict[int, list[Candidate]] = {}

        def sample() -> Candidate:
            nonlocal next_idx
            comp = client.complete(initial_messages(task), seed=base + next_idx, **sampling)  # noqa: B023
            c = make_candidate(task, tier, "sample", next_idx, 0, comp, probes, stress, exec_cfg)  # noqa: B023
            next_idx += 1
            return c

        log(f"[{tier}] threshold {lam:g}, up to {max_draws} draws")
        for _ in range(n_init):
            drawn.append(sample())
        while True:
            for c in drawn:
                if checks_ok(c):
                    _rate(c, task, client, policy.score, gen)
            scores = phi_scores(drawn, policy.score)
            i = max(range(len(drawn)), key=lambda k: (scores[k], -k))  # noqa: B023
            best = drawn[i]
            log(
                f"[{tier}] {len(drawn)} drawn, {sum(map(checks_ok, drawn))} pass the checks;"
                f" best score {scores[i]:.2f}"
            )
            if scores[i] >= lam and checks_ok(best):
                trace.append(
                    {"tier": tier, "step": "accept", "score": scores[i], "drawn": len(drawn)}
                )
                log(f"[{tier}] accept (score {scores[i]:.2f} >= {lam:g})")
                return {"accepted": best, "tier": tier, "drawn": all_drawn + drawn, "trace": trace}
            if len(drawn) >= max_draws:
                break
            action = _choose_action(policy, t, drawn)
            nxt = None
            if action == "repair":
                failing = [c for c in drawn if not c.passes_visible]
                target = (
                    max(failing, key=lambda c: (c.visible_pass_rate, c.depth)) if failing else None
                )
                if target is not None and target.depth < int(gen.get("repair", {}).get("depth", 2)):
                    d = target.depth + 1
                    comp = client.complete(
                        repair_messages(task, target.code, target.feedback),
                        seed=base + 10_000 + target.sample_idx * 100 + d, **sampling,
                    )  # fmt: skip
                    nxt = make_candidate(task, tier, "repair", target.sample_idx, d, comp,
                                         probes, stress, exec_cfg)  # fmt: skip
                    repairs.setdefault(target.sample_idx, []).append(nxt)
            if nxt is None:
                action, nxt = "resample", sample()
            trace.append({"tier": tier, "step": action, "score": scores[i], "drawn": len(drawn)})
            log(f"[{tier}] {action}")
            drawn.append(nxt)
        trace.append({"tier": tier, "step": "escalate", "drawn": len(drawn)})
        log(f"[{tier}] escalate")
        all_drawn.extend(drawn)
    trace.append({"step": "abstain"})
    log("abstain: no candidate certified at this risk level")
    return {"accepted": None, "tier": None, "drawn": all_drawn, "trace": trace}


def replay(pools: list[TierPool], policy: Policy, perm_seed: int = 0) -> dict[str, Any]:
    """The same decisions from cached pools (no model calls)."""
    o = run_task(pools, policy, perm_seed)
    names = [p.tier for p in pools]
    trace = [
        {"tier": names[t], "step": step, "score": round(score, 3)} for t, step, score in o.trace
    ]
    if o.accepted is None:
        trace.append({"step": "abstain"})
    return {
        "accepted": o.accepted,
        "tier": names[o.tier_index] if o.tier_index is not None else None,
        "drawn": o.drawn,
        "trace": trace,
    }
