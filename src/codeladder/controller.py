"""The CodeLadder controller, run as an offline replay over cached pools.

At each tier the controller draws ``n_init`` samples, then loops:
  - accept the best candidate if its score phi >= the tier's threshold;
  - otherwise repair a failing candidate or resample, until ``max_draws``;
  - then escalate to the next tier. After the last tier it abstains.

The controller never reads ``Candidate.hidden_pass``; only the evaluator does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .scoring import checks_ok, phi_scores
from .types import Candidate, TierPool


def error_class(err: str) -> str:
    if err == "pass":
        return "pass"
    if err.startswith("exception:"):
        return "runtime"
    if err in ("syntax_error", "load_error", "missing_entry_point", "crashed"):
        return "broken"
    return err  # wrong_answer, timeout


@dataclass
class Policy:
    thresholds: list[float | None]  # one per tier; None skips the tier (sub-ladders)
    n_init: int = 3
    max_draws: int = 8
    action: str = "resample"  # resample | repair | adaptive
    action_table: dict[str, str] = field(
        default_factory=dict
    )  # context key -> action (action_keys)
    score: str = "agreement"  # one of scoring.SCORES (see scoring.phi_scores)
    name: str = ""
    # Draws allowed at every tier below the last one used (None: max_draws everywhere);
    # small values make cheap rungs give up early instead of spending a full budget.
    lower_budget: int | None = None


@dataclass
class Outcome:
    accepted: Candidate | None
    tier_index: int | None
    drawn: list[Candidate]
    repair_fallbacks: int = 0  # times a repair was wanted but no cached chain existed
    trace: list[tuple[int, str, float]] = field(default_factory=list)  # (tier, action, best phi)


def _best(cands: list[Candidate], score: str = "agreement") -> tuple[int, float]:
    scores = phi_scores(cands, score)
    i = max(range(len(cands)), key=lambda k: (scores[k], -k))
    return i, scores[i]


def _repair_target(drawn: list[Candidate], pool: TierPool, used: set[str]) -> Candidate | None:
    """The most promising failing candidate that still has an unused cached repair."""
    options = []
    for c in drawn:
        if c.passes_visible:
            continue
        chain = pool.repairs.get(c.sample_idx, [])
        if len(chain) > c.depth and chain[c.depth].cand_id not in used:
            options.append(c)
    if not options:
        return None
    return max(options, key=lambda c: (c.visible_pass_rate, c.depth))


def visible_bucket(rate: float) -> str:
    return "0" if rate <= 0 else ("low" if rate < 0.5 else "high")


def action_keys(tier_idx: int, target: Candidate) -> list[str]:
    """Context keys for the action table, most specific first: tier, error class, visible pass
    rate bucket and repair depth of the candidate that would be repaired; then tier and error
    class; then tier. A table may define any of these levels (X4 learns all three)."""
    err = error_class(target.error_type)
    return [
        f"{tier_idx}|{err}|{visible_bucket(target.visible_pass_rate)}|{target.depth}",
        f"{tier_idx}|{err}",
        f"{tier_idx}",
    ]


def _choose_action(policy: Policy, tier_idx: int, drawn: list[Candidate]) -> str:
    if policy.action != "adaptive":
        return policy.action
    failing = [c for c in drawn if not c.passes_visible]
    if not failing:
        return "resample"
    target = max(failing, key=lambda c: (c.visible_pass_rate, c.depth))
    for key in action_keys(tier_idx, target):
        if key in policy.action_table:
            return policy.action_table[key]
    return "resample"


def run_task(pools: list[TierPool], policy: Policy, perm_seed: int = 0, on_event=None) -> Outcome:
    """Replay the controller on one task. ``pools`` are ordered cheapest tier first.

    ``on_event`` (optional, for the demo) receives a dict per step: tier, draw, scores,
    accept, resample / repair, escalate, abstain. Without it nothing extra is computed."""
    emit = on_event or (lambda e: None)
    all_drawn: list[Candidate] = []
    trace: list[tuple[int, str, float]] = []
    fallbacks = 0
    last = max(i for i, th in enumerate(policy.thresholds) if th is not None)
    for t, pool in enumerate(pools):
        lam = policy.thresholds[t]
        if lam is None:
            continue  # tier not part of this ladder
        max_draws, n_init = policy.max_draws, policy.n_init
        if policy.lower_budget is not None and t < last:
            max_draws, n_init = policy.lower_budget, min(policy.n_init, policy.lower_budget)
        order = list(range(len(pool.samples)))
        random.Random(perm_seed * 1009 + t).shuffle(order)
        fresh = iter([pool.samples[k] for k in order])  # samples in this replay's order
        drawn: list[Candidate] = []
        used: set[str] = set()
        if on_event:
            emit({"type": "tier", "tier": t, "threshold": lam, "max_draws": max_draws})

        for _ in range(n_init):
            c = next(fresh, None)
            if c is None:
                break
            drawn.append(c)
            used.add(c.cand_id)
            if on_event:
                emit({"type": "draw", "tier": t, "cand": c})

        while drawn:
            i, score = _best(drawn, policy.score)
            if on_event:
                emit({"type": "scores", "tier": t, "scores": phi_scores(drawn, policy.score),
                      "best": i, "threshold": lam})  # fmt: skip
            if score >= lam and checks_ok(drawn[i]):
                trace.append((t, "accept", score))
                if on_event:
                    emit({"type": "accept", "tier": t, "cand": drawn[i], "score": score})
                return Outcome(drawn[i], t, all_drawn + drawn, fallbacks, trace)
            if len(drawn) >= max_draws:
                break
            action = _choose_action(policy, t, drawn)
            nxt: Candidate | None = None
            if action == "repair":
                target = _repair_target(drawn, pool, used)
                if target is not None:
                    nxt = pool.repairs[target.sample_idx][target.depth]
                else:
                    fallbacks += 1
            if nxt is None:
                nxt = next(fresh, None)
                action = "resample"
            if nxt is None:
                break  # pool exhausted
            trace.append((t, action, score))
            if on_event:
                emit({"type": action, "tier": t})
                emit({"type": "draw", "tier": t, "cand": nxt})
            drawn.append(nxt)
            used.add(nxt.cand_id)
        trace.append((t, "escalate", _best(drawn, policy.score)[1] if drawn else 0.0))
        if on_event:
            emit({"type": "escalate", "tier": t})
        all_drawn.extend(drawn)
    if on_event:
        emit({"type": "abstain"})
    return Outcome(None, None, all_drawn, fallbacks, trace)


def oracle_outcome(pools: list[TierPool], max_draws: int, perm_seed: int = 0) -> Outcome:
    """Cheapest path to a hidden-correct sample (uses labels: a lower bound on cost only)."""
    all_drawn: list[Candidate] = []
    for t, pool in enumerate(pools):
        order = list(range(len(pool.samples)))
        random.Random(perm_seed * 1009 + t).shuffle(order)
        for k in order[:max_draws]:
            c = pool.samples[k]
            all_drawn.append(c)
            if c.hidden_pass:
                return Outcome(c, t, all_drawn)
    return Outcome(None, None, all_drawn)


def learn_action_table(task_pools: list[list[TierPool]]) -> dict[str, str]:
    """Label-free repair-vs-resample table.

    For each (tier, error class) compare how often one repair round makes a failing
    candidate pass the *visible* tests with how often a fresh sample passes them.
    Hidden labels are not used, so the table can be fit on any split.
    """
    from collections import defaultdict

    rep_ok: dict[str, list[int]] = defaultdict(list)
    fresh_ok: dict[int, list[int]] = defaultdict(list)
    for pools in task_pools:
        for t, pool in enumerate(pools):
            fresh_ok[t].extend(int(c.passes_visible) for c in pool.samples)
            for s, chain in pool.repairs.items():
                parent = pool.samples[s]
                prev = parent
                for c in chain:
                    rep_ok[f"{t}|{error_class(prev.error_type)}"].append(int(c.passes_visible))
                    prev = c
    table = {}
    for key, outcomes in rep_ok.items():
        t = int(key.split("|")[0])
        fresh = fresh_ok.get(t) or [0]
        table[key] = (
            "repair" if sum(outcomes) / len(outcomes) > sum(fresh) / len(fresh) else "resample"
        )
    return table
