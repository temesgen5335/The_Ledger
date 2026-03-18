"""Multi-provider LLM text generation with ordered fallback and per-provider circuit-breakers.

Provider selection is driven entirely by environment variables:

    LLM_PROVIDER_ORDER=gemini,openai,anthropic   # try in this order
    GEMINI_LLM_MODEL=gemini-3.0-flash
    OPENAI_LLM_MODEL=gpt-4o-mini
    ANTHROPIC_LLM_MODEL=claude-haiku-4-5-20251001

A provider is skipped if:
  - its API key env var is empty / unset, or
  - its circuit-breaker has tripped (quota exhausted during this process run).

On a rate-limit error (429 / RESOURCE_EXHAUSTED) the provider gets one retry
after _RETRY_DELAY seconds.  A second rate-limit failure trips its circuit-breaker
and the next provider in the order is tried immediately.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Callable


_RATE_LIMIT_SIGNALS = ("429", "resource_exhausted", "quota", "rate limit", "too many requests")
_RETRY_DELAY = 2  # seconds before the single per-provider retry


def _short_err(exc: Exception) -> str:
    """Extract the first meaningful line from an exception, stripping JSON blobs."""
    first_line = str(exc).splitlines()[0].strip()
    # Trim at the first '{' to drop embedded JSON payloads
    brace = first_line.find("{")
    if brace > 0:
        first_line = first_line[:brace].strip().rstrip(".,—-")
    return first_line[:120]

# Per-provider circuit-breakers — reset when the process restarts.
_provider_exhausted: dict[str, bool] = {}

_FALLBACK_GEMINI_MODEL = "gemini-3.0-flash"
_FALLBACK_OPENAI_MODEL = "gpt-4o-mini"
_FALLBACK_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        base = Path(__file__).resolve().parent.parent.parent
        load_dotenv(base / ".env")
    except ImportError:
        pass


def _llm_config() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (provider_order, models_by_provider, api_keys_by_provider) from env."""
    _load_dotenv()
    order_raw = os.environ.get("LLM_PROVIDER_ORDER", "gemini,openai,anthropic")
    provider_order = [p.strip().lower() for p in order_raw.split(",") if p.strip()]

    models: dict[str, str] = {
        "gemini": os.environ.get("GEMINI_LLM_MODEL", _FALLBACK_GEMINI_MODEL).strip(),
        "openai": os.environ.get("OPENAI_LLM_MODEL", _FALLBACK_OPENAI_MODEL).strip(),
        "anthropic": os.environ.get("ANTHROPIC_LLM_MODEL", _FALLBACK_ANTHROPIC_MODEL).strip(),
    }

    keys: dict[str, str] = {
        "gemini": (os.environ.get("GEMINI_API_KEY") or "").strip(),
        "openai": (os.environ.get("OPENAI_API_KEY") or "").strip(),
        "anthropic": (os.environ.get("ANTHROPIC_API_KEY") or "").strip(),
    }

    return provider_order, models, keys


# ---------------------------------------------------------------------------
# Provider-specific callers
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=[prompt])
    return (response.text or "").strip()


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return (message.content[0].text or "").strip()


_CALLERS: dict[str, Callable[[str, str, str], str]] = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_generate(prompt: str) -> str | None:
    """Generate text using the first available LLM provider.

    Iterates through LLM_PROVIDER_ORDER, skipping providers with no API key
    or a tripped circuit-breaker.  Returns None when no provider succeeds.
    """
    provider_order, models, keys = _llm_config()

    for provider in provider_order:
        api_key = keys.get(provider, "")
        if not api_key:
            continue
        if _provider_exhausted.get(provider):
            continue

        model = models[provider]
        caller = _CALLERS.get(provider)
        if caller is None:
            warnings.warn(f"Unknown LLM provider {provider!r} — skipping.", RuntimeWarning, stacklevel=2)
            continue

        for attempt in range(2):
            if attempt == 1:
                time.sleep(_RETRY_DELAY)
            try:
                return caller(prompt, model, api_key)
            except Exception as exc:
                err_lower = str(exc).lower()
                if any(sig in err_lower for sig in _RATE_LIMIT_SIGNALS):
                    if attempt == 0:
                        warnings.warn(
                            f"[{provider}/{model}] Rate limit — retrying in {_RETRY_DELAY}s",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    _provider_exhausted[provider] = True
                    warnings.warn(
                        f"[{provider}/{model}] Quota exhausted — falling back to next provider",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"[{provider}/{model}] LLM call failed: {_short_err(exc)}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                break

    return None


def active_provider() -> str | None:
    """Return the name of the first provider that has a key and is not exhausted."""
    provider_order, _, keys = _llm_config()
    for provider in provider_order:
        if keys.get(provider) and not _provider_exhausted.get(provider):
            return provider
    return None
