"""Builds providers from configuration and resolves which one an agent should use."""

from __future__ import annotations

import os
import time

from prometheus_client import Counter, Histogram

from ai_eos.config import ProviderCfg, Settings
from ai_eos.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse
from ai_eos.llm.providers import AnthropicProvider, GeminiProvider, OfflineProvider, OpenAICompatibleProvider

LLM_CALLS = Counter("eos_llm_calls_total", "LLM calls", ["provider", "purpose", "outcome"])
LLM_TOKENS = Counter("eos_llm_tokens_total", "LLM tokens", ["provider", "direction"])
LLM_LATENCY = Histogram("eos_llm_latency_seconds", "LLM latency", ["provider"])

_KINDS = {
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def build_provider(name: str, cfg: ProviderCfg, timeout: float, model: str | None = None) -> LLMProvider:
    if cfg.kind == "offline":
        return OfflineProvider(name, model or cfg.model)
    cls = _KINDS.get(cfg.kind)
    if cls is None:
        raise LLMError(f"unknown provider kind '{cfg.kind}' for '{name}'")
    key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    if cfg.api_key_env and not key:
        raise LLMError(f"provider '{name}' needs the environment variable {cfg.api_key_env}")
    return cls(name, model or cfg.model, cfg.base_url, key, timeout=timeout)


class LLMRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[tuple[str, str | None], LLMProvider] = {}
        self._overrides: dict[tuple[str, str | None], LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider, model: str | None = None) -> None:
        """Inject a provider instance (used by tests and plugins)."""
        self._overrides[(name, model)] = provider

    def available(self) -> list[dict[str, str | bool]]:
        out = []
        for name, cfg in self.settings.llm.providers.items():
            configured = cfg.kind == "offline" or not cfg.api_key_env or bool(os.environ.get(cfg.api_key_env))
            out.append({"name": name, "kind": cfg.kind, "model": cfg.model, "configured": configured})
        return out

    def get(self, name: str | None = None, model: str | None = None) -> LLMProvider:
        name = name or self.settings.llm.default_provider
        if (name, model) in self._overrides:
            return self._overrides[(name, model)]
        if (name, None) in self._overrides:
            return self._overrides[(name, None)]
        if (name, model) not in self._cache:
            cfg = self.settings.llm.providers.get(name)
            if cfg is None:
                raise LLMError(f"provider '{name}' is not configured")
            self._cache[(name, model)] = build_provider(name, cfg, self.settings.llm.timeout_seconds, model)
        return self._cache[(name, model)]

    async def complete(self, request: LLMRequest, provider: str | None = None, model: str | None = None) -> LLMResponse:
        llm = self.get(provider, model)
        start = time.perf_counter()
        try:
            resp = await llm.complete(request)
        except Exception:
            LLM_CALLS.labels(llm.name, request.purpose, "error").inc()
            raise
        LLM_LATENCY.labels(llm.name).observe(time.perf_counter() - start)
        LLM_CALLS.labels(llm.name, request.purpose, "ok").inc()
        LLM_TOKENS.labels(llm.name, "input").inc(resp.usage.get("input", 0))
        LLM_TOKENS.labels(llm.name, "output").inc(resp.usage.get("output", 0))
        return resp
