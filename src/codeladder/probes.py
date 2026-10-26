"""Probe inputs for behavioural agreement.

Probes are produced by type-aware mutation of the *visible* inputs (in the spirit
of EvalPlus). They have no expected outputs: we only check whether different
candidates behave the same way on them. Hidden tests are never used here.
"""

from __future__ import annotations

import copy
import json
import random
import re
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


# Whole-program (stdin) tasks: inputs are text in a problem-specific format, so mutations must
# keep that format. The first line (usually sizes and counts) and short all-number lines (often
# counts, edges or queries) are kept as they are, unless the input is a single line. Mutations: shuffle or redraw values within a
# number array (sorted arrays stay sorted, permutations are only shuffled), shuffle or replace
# characters of a string using the input's own string alphabet (grids stay grids, binary strings
# stay binary), and swap two lines within a run of same-shape lines (edges, queries, grid rows).
_INT = re.compile(r"-?\d+")


def _is_int(tok: str) -> bool:
    # "00011" or very long digit strings are strings, not numbers.
    return (
        bool(_INT.fullmatch(tok))
        and not (len(tok) > 1 and tok.lstrip("-")[0] == "0")
        and len(tok) <= 18
    )


def _shape(tokens: list[str]) -> tuple:
    return tuple("i" if _is_int(t) else ("s", len(t)) for t in tokens)


def _mutate_array(vals: list[int], rng: random.Random) -> list[int]:
    out = list(vals)
    distinct = len(set(vals)) == len(vals)
    perm = sorted(vals) in (list(range(1, len(vals) + 1)), list(range(len(vals))))
    op = "shuffle" if perm else rng.choice(["shuffle", "redraw", "range"])
    i = rng.randrange(len(out))
    if op == "shuffle":
        rng.shuffle(out)
    elif op == "redraw" and not distinct:
        out[i] = rng.choice(vals)
    else:
        lo, hi = min(vals), max(vals)
        choices = (
            [v for v in range(lo, hi + 1) if not distinct or v not in vals]
            if hi - lo < 10_000
            else []
        )
        if choices:
            out[i] = rng.choice(choices)
        elif not distinct:
            out[i] = rng.randint(lo, hi)
    if vals == sorted(vals):
        out.sort()
    return out


def _mutate_scalar(v: int, rng: random.Random) -> int:
    out = rng.choice(
        [v + 1, v - 1, 2 * v, v // 2, v + rng.randint(1, 10), rng.randint(0, 2 * abs(v) + 10)]
    )
    return max(out, 1) if v >= 1 else (abs(out) if v == 0 else -abs(out) or -1)


def _mutate_string(tok: str, alphabet: list[str], rng: random.Random) -> str:
    s = list(tok)
    if rng.random() < 0.5:
        rng.shuffle(s)
    else:
        s[rng.randrange(len(s))] = rng.choice(alphabet)
    return "".join(s)


def _mutate_text(text: str, alphabet: list[str], rng: random.Random) -> str:
    lines = text.split("\n")
    toks = [ln.split() for ln in lines]
    units: list[tuple[str, int, int]] = []  # (kind, line, token)
    # A one-line input has no counts that later lines depend on: its numbers are the data.
    one_line = sum(1 for tk in toks if tk) == 1
    for li, tk in enumerate(toks):
        for ti, t in enumerate(tk):
            if not _is_int(t) and len(t) >= 2 and len(alphabet) >= 2:
                units.append(("str", li, ti))
            elif one_line and _is_int(t) and len(tk) < 3:
                units.append(("num", li, ti))
        if (li > 0 or one_line) and len(tk) >= 3 and all(_is_int(t) for t in tk):
            units.append(("arr", li, -1))
    # Swaps stay inside a run of consecutive same-shape lines, so input sections never mix.
    pairs = [
        (a, b)
        for a in range(1, len(toks))
        for b in range(a + 1, len(toks))
        if toks[a]
        and toks[a] != toks[b]
        and all(_shape(toks[k]) == _shape(toks[a]) for k in range(a, b + 1))
    ]
    if pairs:
        units.append(("swap", -1, -1))
    if not units:
        return text
    changed: set[int] = set()
    for _ in range(rng.randint(1, 3)):
        kind, li, ti = rng.choice(units)
        if kind == "str":
            toks[li][ti] = _mutate_string(toks[li][ti], alphabet, rng)
            changed.add(li)
        elif kind == "num":
            toks[li][ti] = str(_mutate_scalar(int(toks[li][ti]), rng))
            changed.add(li)
        elif kind == "arr":
            toks[li] = [str(v) for v in _mutate_array([int(t) for t in toks[li]], rng)]
            changed.add(li)
        else:
            a, b = rng.choice(pairs)
            toks[a], toks[b] = toks[b], toks[a]
            changed.update((a, b))
    return "\n".join(" ".join(toks[i]) if i in changed else ln for i, ln in enumerate(lines))


def make_stdin_probes(visible_inputs: list[str], n: int, seed: int = 0) -> list[str]:
    """Return up to ``n`` distinct format-preserving mutations of the visible stdin inputs."""
    texts = [x for x in visible_inputs if isinstance(x, str) and x.strip()]
    if not texts or n <= 0:
        return []
    rng = random.Random(seed)
    alphabet = sorted({ch for x in texts for t in x.split() if not _is_int(t) for ch in t})
    seen = set(texts)
    probes: list[str] = []
    attempts = 0
    while len(probes) < n and attempts < n * 20:
        attempts += 1
        p = _mutate_text(texts[attempts % len(texts)], alphabet, rng)
        if p not in seen:
            seen.add(p)
            probes.append(p)
    return probes
