"""Client for a pool of OpenAI-compatible LLM endpoints.

Free API tiers cap usage per day, and the caps are scoped per provider and
per model. This client therefore treats (provider, key, model) combinations
as an endpoint pool: requests go to the current endpoint, a rate-limited
endpoint is skipped in favour of the next, and an endpoint that reports a
daily cap is disabled for the rest of the session so long runs stop wasting
attempts on it. The pool is exhausted only when every endpoint is disabled,
which callers can catch to stop gracefully.

The rest of the project depends only on this class, so providers and models
can be added or removed in .env without touching any calling code.
"""

import time
from typing import List, Dict, Optional, Set

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from src.config import LLMConfig, load_llm_config


class BudgetExhaustedError(RuntimeError):
    """Raised when every endpoint in the pool hit a daily usage cap."""


def _is_daily_cap(error_text: str) -> bool:
    """Recognise rate-limit errors that will not clear within minutes.

    Groq reports daily caps as "tokens per day (TPD)"; Google reports quota
    ids containing "PerDay". Anything else (per-minute or burst limits) is
    treated as transient.
    """
    lowered = error_text.lower()
    return "per day" in lowered or "perday" in lowered or "tpd" in lowered


class LLMClient:
    """Chat client that spreads requests across an endpoint pool."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or load_llm_config()

        if not self.config.endpoints:
            raise ValueError(
                "No endpoints configured. Set LLM_ENDPOINTS (or the legacy "
                "LLM_BASE_URL / LLM_API_KEYS / LLM_MODEL) in your .env file."
            )

        # One transport client per endpoint.
        self._clients = [
            OpenAI(
                base_url=endpoint.base_url,
                api_key=endpoint.api_key,
                timeout=self.config.timeout,
            )
            for endpoint in self.config.endpoints
        ]
        self._current = 0
        # Endpoints that reported a daily cap this session; skipped until the
        # process restarts (daily budgets do not recover mid-run).
        self._disabled: Set[int] = set()

    @property
    def num_endpoints(self) -> int:
        return len(self._clients)

    @property
    def num_active(self) -> int:
        return self.num_endpoints - len(self._disabled)

    def _advance(self) -> None:
        """Move to the next non-disabled endpoint, wrapping around."""
        if self.num_active == 0:
            return
        self._current = (self._current + 1) % self.num_endpoints
        while self._current in self._disabled:
            self._current = (self._current + 1) % self.num_endpoints

    def _ensure_active(self) -> None:
        """Make sure the current endpoint is usable, or raise."""
        if self.num_active == 0:
            raise BudgetExhaustedError(
                "All endpoints have hit their daily usage caps. Progress is "
                "cached; re-run after the budgets reset."
            )
        if self._current in self._disabled:
            self._advance()

    def list_models(self) -> List[str]:
        """Return the model ids the current endpoint's provider reports."""
        self._ensure_active()
        response = self._clients[self._current].models.list()
        return [model.id for model in response.data]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_attempts: Optional[int] = None,
    ) -> str:
        """Send a chat request, working through the endpoint pool as needed.

        Temperature defaults to 0.0 so extraction output is deterministic.
        On a transient rate limit the client moves to the next endpoint; on a
        daily-cap rate limit the endpoint is disabled for the session first.
        When a full cycle over the active endpoints has been tried, a short
        backoff precedes the next cycle. Raises BudgetExhaustedError once no
        active endpoint remains.
        """
        attempts_limit = max_attempts or (self.num_endpoints * 2)
        last_error: Optional[Exception] = None
        tried_in_cycle = 0

        for _ in range(attempts_limit):
            self._ensure_active()
            client = self._clients[self._current]
            endpoint = self.config.endpoints[self._current]
            try:
                response = client.chat.completions.create(
                    model=endpoint.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""

            except RateLimitError as error:
                last_error = error
                if _is_daily_cap(str(error)):
                    # This endpoint is done for the day; stop cycling to it.
                    self._disabled.add(self._current)
                self._advance()
                tried_in_cycle += 1
                if tried_in_cycle >= self.num_active > 0:
                    # Every active endpoint was throttled in this cycle; wait
                    # before the next pass instead of hammering them.
                    time.sleep(10.0)
                    tried_in_cycle = 0

            except (APITimeoutError, APIConnectionError) as error:
                # Transient network issue: short wait, same endpoint.
                last_error = error
                time.sleep(1.5)

            except Exception as error:  # noqa: BLE001 - surfaced after attempts
                # Provider-side or protocol errors: try the next endpoint.
                last_error = error
                self._advance()
                time.sleep(1.0)

        if self.num_active == 0:
            raise BudgetExhaustedError(
                "All endpoints have hit their daily usage caps. Progress is "
                "cached; re-run after the budgets reset."
            )
        raise RuntimeError(
            f"Chat request failed after {attempts_limit} attempts across "
            f"{self.num_endpoints} endpoint(s): {last_error}"
        )

    def complete(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        """Convenience wrapper for a single-turn user prompt."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
