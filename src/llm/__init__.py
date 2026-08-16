"""AI Market Analyst Subsystem for Indian Market Portfolio Intelligence."""
from __future__ import annotations

from llm.analyst import generate_analyst_report
from llm.client import LLMClient
from llm.prompts import (
    build_analyst_user_prompt,
    format_backtest_metrics_table,
    format_regime_context,
    format_risk_forecast_context,
    generate_mock_report,
    get_analyst_system_prompt,
)
from llm.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    LLMAuthenticationError,
    LLMError,
    LLMProviderError,
    LLMQuotaError,
    LLMTimeoutError,
    MockProvider,
    NvidiaProvider,
    OpenRouterProvider,
    get_env_var,
)

__all__ = [
    "generate_analyst_report",
    "LLMClient",
    "BaseProvider",
    "GeminiProvider",
    "GroqProvider",
    "NvidiaProvider",
    "OpenRouterProvider",
    "MockProvider",
    "LLMError",
    "LLMProviderError",
    "LLMAuthenticationError",
    "LLMQuotaError",
    "LLMTimeoutError",
    "get_env_var",
    "get_analyst_system_prompt",
    "build_analyst_user_prompt",
    "format_backtest_metrics_table",
    "format_regime_context",
    "format_risk_forecast_context",
    "generate_mock_report",
]
