"""Pre-run checks that fail fast and loud before any tokens are spent.

Without this, a missing ``OPENAI_API_KEY`` surfaces as a cryptic litellm
exception deep inside GEPA — after dataset generation has already burned
budget. We resolve the provider for each configured model and verify the
corresponding credential is present up front.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

# Map a litellm provider prefix to the env var(s) that satisfy its auth.
# A provider is satisfied if ANY of its listed env vars is set.
PROVIDER_ENV_VARS: Dict[str, List[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
}

# Providers that run locally and need no credential.
KEYLESS_PROVIDERS = {"ollama", "local", "hosted_vllm", "vllm"}


def provider_of(model: str) -> str:
    """Return the litellm provider prefix for ``model`` (defaults to openai)."""
    if "/" in model:
        return model.split("/", 1)[0].lower()
    return "openai"


def missing_api_keys(
    models: Iterable[str],
    env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return human-readable messages for each model whose credential is absent.

    Unknown providers (not in :data:`PROVIDER_ENV_VARS` and not keyless) are not
    flagged — we only assert credentials we actually know how to check, to avoid
    false positives that would block valid custom endpoints.
    """
    environ = os.environ if env is None else env
    messages: List[str] = []
    seen: set[tuple[str, str]] = set()
    for model in models:
        if not model:
            continue
        provider = provider_of(model)
        if provider in KEYLESS_PROVIDERS:
            continue
        env_vars = PROVIDER_ENV_VARS.get(provider)
        if not env_vars:
            continue  # unknown provider — don't guess
        if any(environ.get(var) for var in env_vars):
            continue
        key = (provider, env_vars[0])
        if key in seen:
            continue
        seen.add(key)
        joined = " or ".join(env_vars)
        messages.append(
            f"Missing credential for provider '{provider}' (model '{model}'): set {joined}."
        )
    return messages
