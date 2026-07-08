"""Health check for the configured LLM endpoint pool.

Sends one tiny chat request to every endpoint in LLM_ENDPOINTS and reports
which are usable right now. Free-tier daily caps mean an endpoint that worked
this morning can be exhausted by evening, so this probes with a real request
rather than trusting the provider's model listing.

Usage (from the project root):
    python -m scripts.check_llm_connection
"""

import sys
import time

from src.config import load_llm_config, LLMConfig
from src.llm_client import LLMClient


def main() -> int:
    config = load_llm_config()
    if not config.endpoints:
        print("No endpoints configured. Set LLM_ENDPOINTS in your .env file.")
        return 1

    print(f"Probing {len(config.endpoints)} endpoint(s)")
    print("-" * 72)

    usable = 0
    for endpoint in config.endpoints:
        single = LLMConfig(endpoints=(endpoint,), timeout=30)
        client = LLMClient(single)
        start = time.perf_counter()
        try:
            reply = client.chat(
                [{"role": "user", "content": "Reply with the word: ok"}],
                max_tokens=5,
                max_attempts=1,
            )
            elapsed = time.perf_counter() - start
            usable += 1
            print(f"  OK    {endpoint.describe():<50} "
                  f"{elapsed:.1f}s  ({reply.strip()[:12]})")
        except Exception as error:  # noqa: BLE001 - reported per endpoint
            text = str(error)
            if "per day" in text.lower() or "perday" in text.lower() or "tpd" in text.lower():
                reason = "daily cap exhausted"
            elif "429" in text:
                reason = "rate limited"
            elif "401" in text or "403" in text:
                reason = "auth failed (check key)"
            else:
                reason = text[-60:]
            print(f"  FAIL  {endpoint.describe():<50} {reason}")

    print("-" * 72)
    print(f"{usable} of {len(config.endpoints)} endpoints usable right now.")
    return 0 if usable > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
