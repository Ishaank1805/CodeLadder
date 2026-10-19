"""OpenAI-compatible chat client with an on-disk cache and simple rate limiting.

Every request is keyed by (model, messages, sampling params, sample index). A
cached response is never re-requested, which keeps free-API quotas safe and makes
the generation pass resumable.
"""

from __future__ import annotations

import hashlib
import json
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


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
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


class OpenAIChatClient:
    """Chat client for one tier (local vLLM/llama.cpp server or a free API)."""

    def __init__(self, tier: TierConfig, cache: ResponseCache | None = None, retries: int = 5):
        from openai import OpenAI  # imported lazily so tests do not need the package

        self.tier = tier
        self.cache = cache
        self.retries = retries
        self._client = OpenAI(base_url=tier.resolve_base_url(), api_key=tier.api_key(), timeout=600)
        self._limiter = _RateLimiter(tier.requests_per_minute)

    def complete(self, messages, *, temperature, top_p, max_tokens, seed) -> Completion:
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        key = ResponseCache.key(self.tier.model, messages, params)
        if self.cache and (hit := self.cache.get(key)):
            return Completion(**{**hit, "cached": True})
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._limiter.wait()
            started = time.time()
            # Local servers support seeds and logprobs; free APIs vary, so we only send the
            # basic parameters to them. The cache key still includes the seed, so
            # repeated samples stay distinct.
            local = self.tier.api_key_env is None
            extra = {"seed": seed, "logprobs": True} if local else {}
            try:
                resp = self._client.chat.completions.create(
                    model=self.tier.model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **extra,
                )
            except Exception as e:  # network errors, 429s, server restarts
                last_err = e
                time.sleep(min(60, 2**attempt))
                continue
            choice = resp.choices[0]
            mean_lp = None
            lp = getattr(choice, "logprobs", None)
            if lp is not None and getattr(lp, "content", None):
                vals = [t.logprob for t in lp.content if t.logprob is not None]
                mean_lp = sum(vals) / len(vals) if vals else None
            usage = resp.usage
            comp = Completion(
                text=choice.message.content or "",
                tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                tokens_out=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.time() - started,
                mean_logprob=mean_lp,
                started_at=started,
            )
            if self.cache:
                self.cache.put(key, {k: v for k, v in comp.__dict__.items() if k != "cached"})
            return comp
        raise RuntimeError(
            f"tier {self.tier.name}: request failed after {self.retries} tries: {last_err}"
        )
