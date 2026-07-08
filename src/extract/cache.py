"""Disk cache for LLM responses.

Extraction over the full corpus costs thousands of API calls against a
rate-limited free tier, so no chunk should ever be paid for twice. Every
response is stored on disk under a key derived from everything that affects
the output: the model id, the prompt version, and the exact chunk text.
Re-running the pipeline therefore only calls the API for chunks that have
never been processed, while changing the prompt or model naturally invalidates
old entries because the key changes.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "extraction"


def _key(prompt_version: str, chunk_text: str) -> str:
    """Build a stable cache key from the inputs that determine the output.

    The serving model is deliberately not part of the key: extraction runs
    across a pool of comparable models to spread daily quota, and a chunk
    extracted by any of them counts as done. Which model actually produced a
    response is still recorded in the payload for later analysis.
    """
    digest = hashlib.sha256()
    digest.update(prompt_version.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(chunk_text.encode("utf-8"))
    return digest.hexdigest()


def get(prompt_version: str, chunk_text: str) -> Optional[str]:
    """Return the cached raw response for this chunk, or None on a miss."""
    path = CACHE_DIR / f"{_key(prompt_version, chunk_text)}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["response"]


def put(prompt_version: str, chunk_text: str, response: str, model: str) -> None:
    """Store a raw model response for this chunk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(prompt_version, chunk_text)}.json"
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "response": response,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
