"""Probe inputs for behavioural agreement.

Probes are produced by type-aware mutation of the *visible* inputs (in the spirit
of EvalPlus). They have no expected outputs: we only check whether different
candidates behave the same way on them. Hidden tests are never used here.
"""

from __future__ import annotations

import copy
import json
import random
import string
from typing import Any


def _clamp_int(x: int, ref: int) -> int:
    bound = 2 * abs(ref) + 10
    return max(-bound, min(bound, x))


def mutate_value(v: Any, rng: random.Random, depth: int = 0) -> Any:
    if depth > 4:
        return copy.deepcopy(v)
    if isinstance(v, bool):
        return (not v) if rng.random() < 0.5 else v
    if isinstance(v, int):
        options = [v + 1, v - 1, v * 2, -v, 0, rng.randint(-10, 10)]
        return _clamp_int(rng.choice(options), v)
    if isinstance(v, float):
        return v + rng.uniform(-1.0, 1.0) * max(1.0, abs(v)) * 0.5
    if isinstance(v, str):
        alphabet = (v or "") + string.ascii_letters + string.digits + " "
        s = list(v)
        op = rng.choice(["insert", "delete", "replace", "swap"])
        if op == "insert" or not s:
            s.insert(rng.randint(0, len(s)), rng.choice(alphabet))
        elif op == "delete":
            del s[rng.randrange(len(s))]
        elif op == "replace":
            s[rng.randrange(len(s))] = rng.choice(alphabet)
        elif len(s) > 1:
            i, j = rng.randrange(len(s)), rng.randrange(len(s))
            s[i], s[j] = s[j], s[i]
        return "".join(s)
    if isinstance(v, (list, tuple)):
        items = list(copy.deepcopy(v))
        if items:
            op = rng.choice(["mutate", "append", "remove", "shuffle"])
            if op == "mutate":
                i = rng.randrange(len(items))
                items[i] = mutate_value(items[i], rng, depth + 1)
            elif op == "append":
                items.append(mutate_value(rng.choice(items), rng, depth + 1))
            elif op == "remove":
                del items[rng.randrange(len(items))]
            else:
                rng.shuffle(items)
        return tuple(items) if isinstance(v, tuple) else items
    if isinstance(v, dict):
        out = copy.deepcopy(v)
        if out:
            k = rng.choice(list(out.keys()))
            out[k] = mutate_value(out[k], rng, depth + 1)
        return out
    return copy.deepcopy(v)


def make_probes(visible_inputs: list[list[Any]], n: int, seed: int = 0) -> list[list[Any]]:
    """Return up to ``n`` distinct mutated argument lists derived from the visible inputs."""
    if not visible_inputs or n <= 0:
        return []
    rng = random.Random(seed)
    seen: set[str] = {json.dumps(x, sort_keys=True, default=repr) for x in visible_inputs}
    probes: list[list[Any]] = []
    attempts = 0
    while len(probes) < n and attempts < n * 20:
        attempts += 1
        base = visible_inputs[attempts % len(visible_inputs)]
        args = list(copy.deepcopy(base))
        if not args:
            continue
        mutated_any = False
        for i in range(len(args)):
            if rng.random() < 0.5:
                args[i] = mutate_value(args[i], rng)
                mutated_any = True
        if not mutated_any:
            i = rng.randrange(len(args))
            args[i] = mutate_value(args[i], rng)
        key = json.dumps(args, sort_keys=True, default=repr)
        if key in seen:
            continue
        seen.add(key)
        probes.append(args)
    return probes
