"""Фабрика провайдеров по конфигу."""

from __future__ import annotations

import os

from harness.providers.base import Provider, ToolCall, ToolOutcome, ToolSpec, Turn

__all__ = ["Provider", "ToolCall", "ToolOutcome", "ToolSpec", "Turn", "build_provider"]

# Дефолты на провайдер: (base_url, env-переменная с ключом, модель по умолчанию)
_PRESETS = {
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-opus-5"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4.1-mini"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
}


def build_provider(cfg: "Config") -> Provider:  # noqa: F821 (см. harness.config)
    provider = cfg.provider.lower()
    if provider not in _PRESETS:
        raise ValueError(
            f"Неизвестный провайдер '{provider}'. Доступны: {', '.join(_PRESETS)}"
        )

    default_url, env_key, default_model = _PRESETS[provider]
    base_url = cfg.base_url or default_url
    api_key = cfg.api_key or os.environ.get(env_key)
    model = cfg.model or default_model

    if provider == "anthropic":
        from harness.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model, cfg.max_tokens, api_key=api_key, base_url=base_url
        )

    # openai и deepseek — один класс, разные пресеты
    from harness.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(
        model,
        cfg.max_tokens,
        api_key=api_key,
        base_url=base_url,
        thinking=cfg.thinking,
        reasoning_effort=cfg.reasoning_effort,
    )
