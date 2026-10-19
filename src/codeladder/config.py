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
    api_key_env: str | None = None
    requests_per_minute: float | None = None

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
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"tier {self.name}: environment variable {self.api_key_env} is not set"
                )
            return key
        return "EMPTY"  # local vLLM / llama.cpp servers accept any key


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
