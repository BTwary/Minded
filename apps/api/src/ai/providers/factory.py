"""Universal AI Provider Factory supporting Dual-Mode (Mode 1: Free/Zero-API vs Mode 2: Optional AI Augmentation)."""
import logging
import os
from typing import Any, Dict, Optional
from apps.api.src.ai.providers.base import BaseAIProvider
from apps.api.src.ai.providers.claude import ClaudeProvider
from apps.api.src.ai.providers.gemini import GeminiProvider
from apps.api.src.ai.providers.mock import DeterministicMockAIProvider, NoneAIProvider
from apps.api.src.ai.providers.ollama import OllamaProvider
from apps.api.src.ai.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Determine initial AI enabled status:
# An explicit AI_ENABLED always wins; a named provider only opts in when AI_ENABLED is unset.
from packages.analytics_core.src.platform_local_first import resolve_ai_enabled

from packages.analytics_core.src.security.ssrf import assert_safe_endpoint, SSRFSecurityError

_env_ai_enabled = resolve_ai_enabled()
_env_provider = os.getenv("AI_PROVIDER", "none").lower()

def _base_default_config() -> Dict[str, Any]:
    return {
        "enabled": _env_ai_enabled,
        "provider": _env_provider,
        "model": os.getenv("AI_MODEL", "auto"),
        "api_key": os.getenv("AI_API_KEY", ""),
        "base_url": os.getenv("AI_BASE_URL", ""),
    }

# Scoped runtime configuration store by user_id
_USER_AI_CONFIGS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_AI_CONFIG: Dict[str, Any] = _base_default_config()


def is_ai_enabled(user_id: Optional[str] = None) -> bool:
    """Check if AI augmentation layer is enabled for a given user or globally."""
    return bool(get_ai_config(user_id=user_id).get("enabled", False))


def get_ai_config(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get active AI runtime configuration for a specific user, or default."""
    if user_id and user_id in _USER_AI_CONFIGS:
        return dict(_USER_AI_CONFIGS[user_id])
    return dict(_DEFAULT_AI_CONFIG)


def set_ai_config(
    user_id: Optional[str] = None,
    enabled: Optional[bool] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Update active AI runtime configuration for a user (or default)."""
    ptype = (provider or "").lower() if provider is not None else ""
    if base_url:
        # Pre-flight only: gives the user an immediate, specific error at
        # settings-save time for an obviously unsafe URL. This is not the sole
        # protection -- each provider (claude.py / openai_compat.py / ollama.py)
        # re-validates and DNS-pins the endpoint again immediately before every
        # outbound request via resolve_pinned_target(), which is what actually
        # closes the DNS-rebinding window between save and use.
        allow_loopback = ptype == "ollama" or (not ptype and "ollama" in base_url)
        assert_safe_endpoint(base_url, allow_loopback=allow_loopback, label="AI provider base URL")

    target = _USER_AI_CONFIGS.setdefault(user_id, _base_default_config()) if user_id else _DEFAULT_AI_CONFIG
    if enabled is not None:
        target["enabled"] = bool(enabled)
    if provider is not None:
        target["provider"] = provider.lower()
        if provider.lower() in ["none", "mock"]:
            target["enabled"] = False
    if model is not None:
        target["model"] = model
    if api_key is not None:
        target["api_key"] = api_key
    if base_url is not None:
        target["base_url"] = base_url


def get_ai_provider(provider_type: Optional[str] = None, user_id: Optional[str] = None) -> BaseAIProvider:
    """
    Instantiate and return the configured Universal AI provider for the given user context.
    If AI is disabled or provider is 'none'/'mock', returns Mode 1 (Deterministic / Free).
    """
    cfg = get_ai_config(user_id=user_id)
    enabled = cfg.get("enabled", False) if provider_type is None else True
    ptype = (provider_type or cfg.get("provider") or os.getenv("AI_PROVIDER", "none")).lower()
    api_key = cfg.get("api_key") or os.getenv("AI_API_KEY", "")
    model = cfg.get("model") or os.getenv("AI_MODEL", "")
    base_url = cfg.get("base_url") or os.getenv("AI_BASE_URL", "")

    # Mode 1: Free / Zero AI API
    if not enabled or ptype in ["none", "mock", "offline", "deterministic"]:
        return NoneAIProvider()


    try:
        if ptype in ["claude", "anthropic"]:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            m = model if model and model != "auto" else "claude-3-7-sonnet-20250219"
            url = base_url if base_url else "https://api.anthropic.com/v1"
            return ClaudeProvider(api_key=key, model=m, base_url=url)

        elif ptype == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            m = model if model and model != "auto" else "gpt-4o"
            url = base_url if base_url else "https://api.openai.com/v1"
            return OpenAICompatibleProvider(api_key=key, base_url=url, model=m)

        elif ptype in ["gemini", "google"]:
            key = api_key or os.getenv("GEMINI_API_KEY", "")
            m = model if model and model != "auto" else "gemini-2.0-flash"
            return GeminiProvider(api_key=key, model=m)

        elif ptype == "groq":
            key = api_key or os.getenv("GROQ_API_KEY", "")
            m = model if model and model != "auto" else "llama-3.3-70b-versatile"
            url = base_url if base_url else "https://api.groq.com/openai/v1"
            return OpenAICompatibleProvider(api_key=key, base_url=url, model=m)

        elif ptype == "ollama":
            url = base_url if base_url else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            m = model if model and model != "auto" else os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
            return OllamaProvider(base_url=url, model=m)

        elif ptype in ["custom", "openrouter", "deepseek", "mistral", "together", "perplexity", "vllm"]:
            key = api_key or "local-token"
            m = model if model and model != "auto" else "deepseek-chat"
            url = base_url if base_url else "https://api.deepseek.com/v1"
            return OpenAICompatibleProvider(api_key=key, base_url=url, model=m)

        else:
            logger.warning("Unrecognized AI provider '%s'; defaulting to deterministic Free mode.", ptype)
            return NoneAIProvider()

    except Exception as exc:
        logger.warning("Failed to initialize AI provider '%s' (%s); falling back to deterministic Free mode.", ptype, str(exc))
        return NoneAIProvider()

