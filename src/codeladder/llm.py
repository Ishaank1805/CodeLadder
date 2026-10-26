"""OpenAI-compatible chat client with an on-disk cache and simple rate limiting.

Every request is keyed by (model, messages, sampling params, sample index). A
cached response is never re-requested, which keeps free-API quotas safe and makes
the generation pass resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import TierConfig


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    mean_logprob: float | None = None
    started_at: float = 0.0
    cached: bool = False
    # Per generated token: [logprob, entropy over the top-k alternatives, top-1 minus top-2
    # logprob]; only when requested with top_logprobs (Bayesian self-escalation baseline).
    token_stats: list[list[float]] | None = None
    # Top alternatives of the first generated token, token text -> logprob (P(True) ratings).
    first_top: dict[str, float] | None = None


_CONTEXT = re.compile(
    r"maximum context length is (\d+) tokens.*?\((\d+) in the messages, (\d+) in the completion\)",
    re.S,
)


def _fitting_max_tokens(err: Exception, max_tokens: int, floor: int = 256) -> int | None:
    """A smaller answer budget when a local server rejects the request as longer than its
    context window (vLLM's 400 message); None for other errors or when too little room is left."""
    if getattr(err, "status_code", None) != 400:
        return None
    m = _CONTEXT.search(str(err))
    if not m:
        return None
    room = int(m.group(1)) - int(m.group(2)) - 16
    return room if floor <= room < max_tokens else None


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        top_logprobs: int = 0,
        continue_final: bool = False,
    ) -> Completion: ...


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class QuotaExhausted(RuntimeError):
    """Every key of a tier is rate-limited for hours (daily quota spent): resume later."""


class _RateLimiter:
    def __init__(self, per_minute: float | None):
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def _token_stats(tok: Any) -> list[float]:
    """[logprob, entropy of the renormalised top-k distribution, top-1 minus top-2 logprob]."""
    import math

    lps = sorted(
        (a.logprob for a in (tok.top_logprobs or []) if a.logprob is not None), reverse=True
    )
    if not lps:
        lps = [tok.logprob or 0.0]
    m = max(lps)
    ps = [math.exp(x - m) for x in lps]
    z = sum(ps)
    ent = -sum(p / z * math.log(p / z) for p in ps if p > 0)
    margin = lps[0] - lps[1] if len(lps) > 1 else 10.0
    return [round(tok.logprob or 0.0, 4), round(ent, 4), round(margin, 4)]


_CONTINUE = {"continue_final_message": True, "add_generation_prompt": False}


class OpenAIChatClient:
    """Chat client for one tier (local vLLM/llama.cpp server or a free API)."""

    def __init__(self, tier: TierConfig, cache: ResponseCache | None = None, retries: int = 5):
        from openai import OpenAI  # imported lazily so tests do not need the package

        self.tier = tier
        self.cache = cache
        self.retries = retries
        url = tier.resolve_base_url()
        # One client and rate limiter per key; a key that hits a limit rests until the server
        # says it may retry, and requests move on to the next key.
        # Local servers: no SDK-level retries, so a 500 reaches the loop in complete() at once
        # (the SDK would otherwise repeat a failing full-length generation twice before the
        # retry without logprobs). Free APIs keep the SDK's retries for transient errors.
        sdk_retries = 0 if tier.api_key_env is None else 2
        self._clients = [
            OpenAI(base_url=url, api_key=k, timeout=600, max_retries=sdk_retries)
            for k in tier.api_keys()
        ]
        self._limiters = [_RateLimiter(tier.requests_per_minute) for _ in self._clients]
        self._rest_until = [0.0] * len(self._clients)
        self._next_key = 0
        self._key_lock = threading.Lock()

    def _pick_key(self, max_wait_s: float = 3 * 3600) -> int:
        """Next key that is not resting (round robin); waits for the earliest one if all are."""
        while True:
            with self._key_lock:
                now = time.time()
                n = len(self._clients)
                for d in range(n):
                    i = (self._next_key + d) % n
                    if self._rest_until[i] <= now:
                        self._next_key = (i + 1) % n
                        return i
                wait = min(self._rest_until) - now
            if wait > max_wait_s:
                raise QuotaExhausted(
                    f"tier {self.tier.name}: every key is rate-limited for {wait / 3600:.1f} h"
                )
            time.sleep(min(wait, 60))

    def _rest(self, i: int, err: Exception) -> None:
        resp = getattr(err, "response", None)
        headers = getattr(resp, "headers", None) or {}
        try:
            secs = float(headers.get("retry-after") or 30)
        except ValueError:
            secs = 30.0
        with self._key_lock:
            self._rest_until[i] = time.time() + secs + 1

    def complete(
        self,
        messages,
        *,
        temperature,
        top_p,
        max_tokens,
        seed,
        top_logprobs: int = 0,
        continue_final: bool = False,
    ) -> Completion:
        """``continue_final``: the last message is a started assistant reply that the server
        continues (vLLM ``continue_final_message``); local servers only."""
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if top_logprobs:
            params["top_logprobs"] = top_logprobs  # separate cache entries
        if continue_final:
            params["continue_final"] = True
        key = ResponseCache.key(self.tier.model, messages, params)
        if self.cache and (hit := self.cache.get(key)):
            return Completion(**{**hit, "cached": True})
        last_err: Exception | None = None
        attempt = rate_waits = 0
        # Dropped after a server-side JSON failure (see below), or from the start with
        # CODELADDER_NO_LOGPROBS=1 (re-runs of tasks known to hit that failure).
        want_logprobs = os.environ.get("CODELADDER_NO_LOGPROBS") != "1"
        while attempt < self.retries:
            k = self._pick_key()
            self._limiters[k].wait()
            started = time.time()
            # Local servers support seeds and logprobs; free APIs vary, so we only send the
            # basic parameters to them. The cache key still includes the seed, so
            # repeated samples stay distinct.
            local = self.tier.api_key_env is None
            extra = {"seed": seed, "logprobs": want_logprobs} if local else {}
            if local and top_logprobs and want_logprobs:
                extra["top_logprobs"] = top_logprobs
            try:
                resp = self._clients[k].chat.completions.create(
                    model=self.tier.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=(
                        {**(self.tier.extra_params or {}), **_CONTINUE}
                        if continue_final
                        else self.tier.extra_params or None
                    ),
                    **extra,
                )
            except Exception as e:  # network errors, 429s, server restarts
                last_err = e
                if getattr(e, "status_code", None) == 429:
                    self._rest(k, e)  # rate limit: not a failure, try another key
                    rate_waits += 1
                    if rate_waits > 500:  # hours of waiting: the request itself is refused
                        raise
                    continue
                fit = _fitting_max_tokens(e, max_tokens)
                if fit is not None:
                    # The prompt plus the answer budget exceeds the server's context window
                    # (a few BigCodeBench prompts carry a very long visible test): answer
                    # within what is left. The response is still cached under the request's
                    # original parameters, so reruns reuse it.
                    max_tokens = fit
                    attempt += 1
                    continue
                if local and want_logprobs and getattr(e, "status_code", None) == 500:
                    # A -inf token logprob (fp16 underflow) makes the server's JSON invalid
                    # and the same seeded request fails every time: retry without logprobs.
                    want_logprobs = False
                    attempt += 1
                    continue
                attempt += 1
                time.sleep(min(60, 2**attempt))
                continue
            choice = resp.choices[0]
            mean_lp, stats, first = None, None, None
            lp = getattr(choice, "logprobs", None)
            if lp is not None and getattr(lp, "content", None):
                vals = [t.logprob for t in lp.content if t.logprob is not None]
                mean_lp = sum(vals) / len(vals) if vals else None
                if top_logprobs:
                    stats = [_token_stats(t) for t in lp.content]
                    first = {
                        a.token: a.logprob
                        for a in (lp.content[0].top_logprobs or [])
                        if a.logprob is not None
                    }
            usage = resp.usage
            comp = Completion(
                text=choice.message.content or "",
                tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                tokens_out=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.time() - started,
                mean_logprob=mean_lp,
                started_at=started,
                token_stats=stats,
                first_top=first,
            )
            if self.cache:
                self.cache.put(key, {k: v for k, v in comp.__dict__.items() if k != "cached"})
            return comp
        raise RuntimeError(
            f"tier {self.tier.name}: request failed after {self.retries} tries: {last_err}"
        )
