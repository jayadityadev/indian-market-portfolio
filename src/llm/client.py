"""Waterfall LLM Client orchestration layer.

Manages provider availability detection, cascade execution (Gemini -> Groq -> NVIDIA NIM -> OpenRouter -> Mock),
graceful failover, and fallback history tracking.
"""
from __future__ import annotations

import logging
from typing import Any

from llm.prompts import generate_mock_report
from llm.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    MockProvider,
    NvidiaProvider,
    OpenRouterProvider,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Waterfall LLM client with multi-provider cascade and deterministic fallback."""

    DEFAULT_CASCADE = ["gemini", "groq", "nvidia", "openrouter", "mock"]

    def __init__(self, providers: dict[str, BaseProvider] | None = None):
        if providers is None:
            self.providers: dict[str, BaseProvider] = {
                "gemini": GeminiProvider(),
                "groq": GroqProvider(),
                "nvidia": NvidiaProvider(),
                "openrouter": OpenRouterProvider(),
                "mock": MockProvider(),
            }
        else:
            self.providers = dict(providers)

    def get_available_providers(self) -> list[str]:
        """Return a list of registered provider names that are currently available."""
        return [name for name, p in self.providers.items() if p.is_available()]

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        provider_override: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Execute text generation through the provider waterfall cascade.

        Args:
            system_prompt: High-level institutional persona & constraints.
            user_prompt: Structured input data and context.
            provider_override: Optional specific provider name (e.g. 'groq' or 'groq:llama-3.1-8b-instant').
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature.

        Returns:
            dict with:
                - content: generated markdown string
                - provider_used: identifier of provider that produced output
                - model_used: model identifier
                - fallback_history: list of provider attempt logs
        """
        fallback_history: list[str] = []
        attempted: set[str] = set()

        # Handle explicit provider override if specified
        if provider_override and provider_override.lower() != "auto":
            override_str = provider_override.strip()
            custom_model: str | None = None
            p_name = override_str.lower()

            if ":" in override_str:
                parts = override_str.split(":", 1)
                p_name = parts[0].strip().lower()
                custom_model = parts[1].strip()

            if p_name in self.providers:
                target_provider = self.providers[p_name]
                attempted.add(p_name)

                if not target_provider.is_available():
                    fallback_history.append(f"{p_name}: skipped (no API key configured)")
                else:
                    try:
                        logger.info(f"Attempting LLM generation with overridden provider: {p_name} (model: {custom_model or target_provider.default_model})")
                        content = target_provider.generate(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            model=custom_model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        fallback_history.append(f"{p_name}: success")
                        return {
                            "content": content,
                            "provider_used": p_name,
                            "model_used": custom_model or target_provider.default_model,
                            "fallback_history": fallback_history,
                        }
                    except Exception as exc:
                        logger.warning(f"Overridden provider {p_name} failed: {exc}. Cascading to remaining providers.")
                        fallback_history.append(f"{p_name}: failed ({exc})")
            else:
                fallback_history.append(f"{p_name}: unknown provider override requested")

        # Execute standard waterfall cascade
        for name in self.DEFAULT_CASCADE:
            if name in attempted:
                continue

            provider = self.providers.get(name)
            if not provider:
                continue

            attempted.add(name)

            if not provider.is_available():
                fallback_history.append(f"{name}: skipped (no API key configured)")
                continue

            try:
                logger.info(f"Cascading to provider: {name} (model: {provider.default_model})")
                content = provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                fallback_history.append(f"{name}: success")
                return {
                    "content": content,
                    "provider_used": name,
                    "model_used": provider.default_model,
                    "fallback_history": fallback_history,
                }
            except Exception as exc:
                logger.warning(f"Provider {name} failed in cascade: {exc}. Trying next provider.")
                fallback_history.append(f"{name}: failed ({exc})")

        # Ultimate fallback guarantee to deterministic mock engine
        fallback_history.append("mock_emergency_fallback: success")
        mock_content = generate_mock_report(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return {
            "content": mock_content,
            "provider_used": "mock",
            "model_used": "mock-deterministic-v1",
            "fallback_history": fallback_history,
        }
