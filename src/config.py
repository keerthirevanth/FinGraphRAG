"""Central configuration for the project.

All environment-specific values are read from a local .env file through this
single module, so the rest of the code never hard-codes a server address or a
key and the same source runs unchanged on another machine.

The LLM configuration is a pool of endpoints rather than a single one. Free
API tiers impose daily caps that are scoped per provider and per model, so
spreading one extraction run across several provider/model combinations
multiplies the usable daily budget. The client works through the pool in
order and moves on when an endpoint is exhausted.
"""

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present. Real
# environment variables take precedence, which is convenient on servers where
# values are injected directly rather than through a file.
load_dotenv()


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible API target: where, with which key, which model."""

    base_url: str
    api_key: str
    model: str

    def describe(self) -> str:
        """Short human-readable label used in logs and error messages."""
        host = self.base_url.split("//")[-1].split("/")[0]
        return f"{self.model} @ {host}"


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for the language-model endpoint pool."""

    endpoints: Tuple[Endpoint, ...]
    timeout: float


def _parse_endpoints() -> Tuple[Endpoint, ...]:
    """Read the endpoint pool from the environment.

    Preferred form is LLM_ENDPOINTS: entries separated by semicolons, fields
    within an entry separated by pipes:

        LLM_ENDPOINTS=<base_url>|<api_key>|<model>;<base_url>|<api_key>|<model>

    For convenience the single-provider form is still accepted and expands to
    one endpoint per key:

        LLM_BASE_URL=...  LLM_API_KEYS=key1,key2  LLM_MODEL=...
    """
    raw = os.getenv("LLM_ENDPOINTS", "").strip()
    if raw:
        endpoints = []
        for entry in raw.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = [field.strip() for field in entry.split("|")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(
                    "Malformed LLM_ENDPOINTS entry (expected "
                    f"'base_url|api_key|model'): {entry[:60]}"
                )
            endpoints.append(Endpoint(*parts))
        return tuple(endpoints)

    # Legacy single-provider form.
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    raw_keys = os.getenv("LLM_API_KEYS", "") or os.getenv("LLM_API_KEY", "")
    keys = tuple(k.strip() for k in raw_keys.split(",") if k.strip())
    if base_url and model and keys:
        return tuple(Endpoint(base_url, key, model) for key in keys)
    return ()


def load_llm_config() -> LLMConfig:
    """Build the language-model configuration from environment variables."""
    return LLMConfig(
        endpoints=_parse_endpoints(),
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
    )
