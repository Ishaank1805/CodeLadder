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

from .scoring import phi_scores
from .types import Candidate, TierPool

ACTIONS = ("resample", "repair", "adaptive")


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
    thresholds: list[float]  # one per tier
    n_init: int = 3
    max_draws: int = 8
    action: str = "resample"  # resample | repair | adaptive
    action_table: dict[str, str] = field(default_factory=dict)  # "tier_idx|error_class" -> action
    name: str = ""


@dataclass
class Outcome:
    accepted: Candidate | None
    tier_index: int | None
    drawn: list[Candidate]
    repair_fallbacks: int = 0  # times a repair was wanted but no cached chain existed
    trace: list[tuple[int, str, float]] = field(default_factory=list)  # (tier, action, best phi)


def _best(cands: list[Candidate]) -> tuple[int, float]:
    scores = phi_scores(cands)
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


def _choose_action(policy: Policy, tier_idx: int, drawn: list[Candidate]) -> str:
    if policy.action != "adaptive":
        return policy.action
    failing = [c for c in drawn if not c.passes_visible]
    if not failing:
        return "resample"
    target = max(failing, key=lambda c: (c.visible_pass_rate, c.depth))
    return policy.action_table.get(f"{tier_idx}|{error_class(target.error_type)}", "resample")


def run_task(pools: list[TierPool], policy: Policy, perm_seed: int = 0) -> Outcome:
    """Replay the controller on one task. ``pools`` are ordered cheapest tier first."""
    all_drawn: list[Candidate] = []
    trace: list[tuple[int, str, float]] = []
    fallbacks = 0
    for t, pool in enumerate(pools):
        lam = policy.thresholds[t]
        order = list(range(len(pool.samples)))
        random.Random(perm_seed * 1009 + t).shuffle(order)
        fresh = iter([pool.samples[k] for k in order])  # samples in this replay's order
        drawn: list[Candidate] = []
        used: set[str] = set()

        for _ in range(policy.n_init):
            c = next(fresh, None)
            if c is None:
                break
            drawn.append(c)
            used.add(c.cand_id)

        while drawn:
            i, score = _best(drawn)
            if score >= lam and drawn[i].passes_visible:
                trace.append((t, "accept", score))
                return Outcome(drawn[i], t, all_drawn + drawn, fallbacks, trace)
            if len(drawn) >= policy.max_draws:
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
            drawn.append(nxt)
            used.add(nxt.cand_id)
        trace.append((t, "escalate", _best(drawn)[1] if drawn else 0.0))
        all_drawn.extend(drawn)
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
