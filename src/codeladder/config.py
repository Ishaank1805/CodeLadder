"""Loading YAML configs and resolving model endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TierConfig:
    name: str
    model: str
    params_b: float
    max_concurrency: int = 8
    enabled: bool = True
    endpoint_file: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None  # a variable name, or a prefix ending in "*" for a key pool
    requests_per_minute: float | None = None  # per key
    extra_params: dict[str, Any] | None = None  # sent with every request (e.g. reasoning_effort)

    def resolve_base_url(self) -> str:
        """Return the OpenAI-compatible base URL for this tier."""
        if self.base_url:
            return self.base_url
        if self.endpoint_file:
            path = Path(os.path.expanduser(self.endpoint_file))
            if not path.exists():
                raise FileNotFoundError(
                    f"tier {self.name}: {path} not found. Is the server job running? "
                    "(scripts/ada/serve_vllm.sbatch writes host:port there)"
                )
            hostport = path.read_text().strip()
            return f"http://{hostport}/v1"
        raise ValueError(f"tier {self.name}: set base_url or endpoint_file")

    def api_key(self) -> str:
        return self.api_keys()[0]

    def api_keys(self) -> list[str]:
        """Keys from the environment, else from ~/.codeladder_keys (KEY=VALUE lines, mode 600,
        never in the repository). A name ending in "*" collects every matching variable."""
        if not self.api_key_env:
            return ["EMPTY"]  # local vLLM / llama.cpp servers accept any key
        env = dict(_key_file()) | dict(os.environ)
        if self.api_key_env.endswith("*"):
            prefix = self.api_key_env[:-1]
            keys = [env[k] for k in sorted(env) if k.startswith(prefix) and env[k]]
        else:
            keys = [env[self.api_key_env]] if env.get(self.api_key_env) else []
        if not keys:
            raise RuntimeError(f"tier {self.name}: no key found for {self.api_key_env}")
        return keys


def _key_file(path: str = "~/.codeladder_keys") -> list[tuple[str, str]]:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out.append((k.strip(), v.strip().strip("'\"")))
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tiers(path: str | Path = REPO_ROOT / "configs" / "models.yaml") -> list[TierConfig]:
    raw = load_yaml(path)
    return [TierConfig(**t) for t in raw["tiers"]]


def enabled_tiers(tiers: list[TierConfig]) -> list[TierConfig]:
    return [t for t in tiers if t.enabled]


def load_experiment(path: str | Path = REPO_ROOT / "configs" / "experiment.yaml") -> dict[str, Any]:
    cfg = load_yaml(path)
    for key, rel in cfg.get("paths", {}).items():
        p = Path(rel)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    return cfg
