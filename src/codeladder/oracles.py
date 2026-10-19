"""Task-specific output checkers that mirror the official EvalPlus evaluator.

Most tasks compare outputs for equality (see executor.outputs_match). A few EvalPlus
tasks use special oracles, for example HumanEval/32 accepts any root of the
polynomial and several MBPP+ tasks compare outputs as sets. We import EvalPlus's
own oracle helpers so our labels match the official ones exactly.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .executor import outputs_match

# entry_point -> oracle name, per benchmark (from evalplus.eval.__init__ / _special_oracle)
_MBPP_SET_EQ = {
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
}  # fmt: skip
_MBPP_NOT_NONE = {"check_str", "text_match_three", "text_starta_endb"}
_MBPP_OTHER = {
    "are_equivalent": "always",
    "sum_div": "sum_div",
    "surface_Area": "surface_area",
    "digit_distance_nums": "digit_distance",
}


def oracle_for(benchmark: str, entry_point: str) -> str | None:
    if benchmark == "humaneval" and entry_point == "find_zero":
        return "find_zero"
    if benchmark == "mbpp":
        if entry_point in _MBPP_SET_EQ:
            return "set_eq"
        if entry_point in _MBPP_NOT_NONE:
            return "not_none"
        return _MBPP_OTHER.get(entry_point)
    return None


def from_canon(c: list[Any]) -> Any:
    """Rebuild a Python value from the executor's canonical form (best effort)."""
    tag = c[0]
    if tag == "n":
        return None
    if tag in ("b", "s"):
        return c[1]
    if tag == "i":
        return int(c[1])
    if tag == "f":
        return float(c[1])
    if tag == "c":
        return complex(*c[1])
    if tag == "l":
        return [from_canon(x) for x in c[1]]
    if tag == "t":
        return tuple(from_canon(x) for x in c[1])
    if tag == "S":
        return {from_canon(x) for x in c[1]}
    if tag == "d":
        return {from_canon(k): from_canon(v) for k, v in c[1]}
    return c[1]


def _items_as_set(c: list[Any]) -> set[str] | None:
    if c[0] in ("l", "t", "S"):
        return {json.dumps(x, sort_keys=True) for x in c[1]}
    if c[0] == "s":
        return {json.dumps(["s", ch]) for ch in c[1]}
    return None


def check(oracle: str | None, args: list[Any], out: list[Any], exp: list[Any], atol: float) -> bool:
    """True if canonical output ``out`` is acceptable for inputs ``args``."""
    if oracle is None:
        return outputs_match(out, exp, atol)
    exact = outputs_match(out, exp, atol)
    try:
        if oracle == "always":
            return True
        if oracle == "set_eq":
            a, b = _items_as_set(out), _items_as_set(exp)
            return a is not None and a == b
        if oracle == "not_none":
            # EvalPlus turns the expected output into "a match was found" (not None).
            e = exp[1] if exp[0] == "b" else exp[0] != "n"
            return out[1] == e if out[0] == "b" else e == (out[0] != "n")
        if oracle == "sum_div":
            return exact or from_canon(out) == 0
        from evalplus.eval._special_oracle import (  # only needed for these three tasks
            _digit_distance_nums,
            _poly,
            _surface_Area,
        )

        if oracle == "surface_area":
            return exact or abs(from_canon(out) - _surface_Area(*args)) <= atol
        if oracle == "digit_distance":
            return exact or from_canon(out) == _digit_distance_nums(*args)
        if oracle == "find_zero":
            x = from_canon(out)
            return (
                isinstance(x, (int, float))
                and math.isfinite(_poly(*args, x))
                and abs(_poly(*args, x)) <= atol
            )
    except Exception:
        return False
    raise ValueError(f"unknown oracle {oracle}")
