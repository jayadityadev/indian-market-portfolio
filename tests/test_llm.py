"""Unit and integration test suite for LLM multi-provider analyst subsystem.

100% hermetic and offline-capable using unittest.mock.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from src.llm.analyst import generate_analyst_report
from src.llm.client import LLMClient
from src.llm.prompts import (
    build_analyst_user_prompt,
    format_backtest_metrics_table,
    format_regime_context,
    format_risk_forecast_context,
    generate_mock_report,
    get_analyst_system_prompt,
)
from src.llm.providers import (
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


# ===========================================================================
# 1. Provider Isolation & Error Handling Tests
# ===========================================================================

class TestProvidersIsolation:
    """Test each provider class in isolation with mocked network calls."""

    def test_mock_provider_generates_five_sections(self):
        """Verify MockProvider generates all 5 required markdown section headers."""
        provider = MockProvider()
        assert provider.is_available() is True
        assert provider.name == "mock"

        report = provider.generate(
            system_prompt="system",
            user_prompt="Asset: ^NSEI\nRegime: Bull",
        )

        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in report
        assert "## 🔄 2. Regime Shift Dynamics & Transition Probability" in report
        assert "## 📊 3. Quantitative Strategy Evaluation & Justification" in report
        assert "## 🛡️ 4. Risk Budgeting & Drawdown Guardrails" in report
        assert "## 🚀 5. Tactical Capital Allocation & Action Plan" in report

    def test_gemini_provider_success(self):
        """Verify GeminiProvider makes correct REST call and extracts response text."""
        provider = GeminiProvider(api_key="fake_gemini_key")
        assert provider.is_available() is True

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini generated market commentary."}]
                    }
                }
            ]
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            output = provider.generate("sys", "user")
            assert output == "Gemini generated market commentary."
            mock_post.assert_called_once()
            called_url = mock_post.call_args[0][0]
            assert "gemini-2.5-flash:generateContent" in called_url
            assert "key=fake_gemini_key" in called_url

    def test_gemini_provider_model_fallback(self):
        """Verify GeminiProvider falls back to next candidate model on 404."""
        provider = GeminiProvider(api_key="fake_gemini_key")

        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404
        mock_resp_404.text = "Model deprecated / not found"

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Fallback candidate output."}]
                    }
                }
            ]
        }

        with patch("httpx.Client.post", side_effect=[mock_resp_404, mock_resp_200]) as mock_post:
            output = provider.generate("sys", "user")
            assert output == "Fallback candidate output."
            assert mock_post.call_count == 2

    def test_gemini_provider_authentication_error(self):
        """Verify GeminiProvider raises LLMAuthenticationError on 401/403."""
        provider = GeminiProvider(api_key="invalid_key")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(LLMAuthenticationError):
                provider.generate("sys", "user")

    def test_gemini_provider_quota_error(self):
        """Verify GeminiProvider raises LLMQuotaError on 429."""
        provider = GeminiProvider(api_key="rate_limited_key")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Resource exhausted"

        with patch("httpx.Client.post", return_value=mock_resp):
            with pytest.raises(LLMQuotaError):
                provider.generate("sys", "user")

    def test_gemini_provider_timeout_error(self):
        """Verify GeminiProvider raises LLMTimeoutError on network timeout."""
        provider = GeminiProvider(api_key="timeout_key")

        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
            with pytest.raises(LLMTimeoutError):
                provider.generate("sys", "user")

    def test_gemini_provider_missing_key(self):
        """Verify GeminiProvider without API key is not available and raises LLMAuthenticationError."""
        provider = GeminiProvider(api_key=None)
        with patch.object(provider, "api_key", None):
            assert provider.is_available() is False
            with pytest.raises(LLMAuthenticationError):
                provider.generate("sys", "user")

    def test_groq_provider_success(self):
        """Verify GroqProvider executes chat completions via OpenAI interface."""
        provider = GroqProvider(api_key="fake_groq_key")
        assert provider.is_available() is True

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq institutional commentary."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai_cls.return_value = mock_client

            output = provider.generate("sys", "user")
            assert output == "Groq institutional commentary."
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_groq_provider_auth_error(self):
        """Verify GroqProvider raises LLMAuthenticationError on auth failure."""
        provider = GroqProvider(api_key="fake_groq_key")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
                "Invalid API key", response=MagicMock(status_code=401), body=None
            )
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMAuthenticationError):
                provider.generate("sys", "user")

    def test_groq_provider_quota_error(self):
        """Verify GroqProvider raises LLMQuotaError on rate limits."""
        provider = GroqProvider(api_key="fake_groq_key")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = openai.RateLimitError(
                "Rate limit exceeded", response=MagicMock(status_code=429), body=None
            )
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMQuotaError):
                provider.generate("sys", "user")

    def test_groq_provider_timeout_error(self):
        """Verify GroqProvider raises LLMTimeoutError on timeout."""
        provider = GroqProvider(api_key="fake_groq_key")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
                request=MagicMock()
            )
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMTimeoutError):
                provider.generate("sys", "user")

    def test_groq_provider_missing_key(self):
        """Verify GroqProvider without key is not available."""
        provider = GroqProvider(api_key=None)
        with patch.object(provider, "api_key", None):
            assert provider.is_available() is False
            with pytest.raises(LLMAuthenticationError):
                provider.generate("sys", "user")

    def test_nvidia_provider_success(self):
        """Verify NvidiaProvider executes chat completions."""
        provider = NvidiaProvider(api_key="fake_nv_key")
        assert provider.is_available() is True

        mock_choice = MagicMock()
        mock_choice.message.content = "NVIDIA NIM commentary."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai_cls.return_value = mock_client

            output = provider.generate("sys", "user")
            assert output == "NVIDIA NIM commentary."
            assert mock_openai_cls.call_args[1]["base_url"] == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_provider_fallback_to_second_model(self):
        """Verify NvidiaProvider tries next candidate model if first fails."""
        provider = NvidiaProvider(api_key="fake_nv_key")

        mock_choice = MagicMock()
        mock_choice.message.content = "NVIDIA second model output."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                Exception("First model unavailable"),
                mock_completion,
            ]
            mock_openai_cls.return_value = mock_client

            output = provider.generate("sys", "user")
            assert output == "NVIDIA second model output."
            assert mock_client.chat.completions.create.call_count == 2

    def test_openrouter_provider_success(self):
        """Verify OpenRouterProvider executes chat completions."""
        provider = OpenRouterProvider(api_key="fake_or_key")
        assert provider.is_available() is True

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenRouter output."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai_cls.return_value = mock_client

            output = provider.generate("sys", "user")
            assert output == "OpenRouter output."
            assert mock_openai_cls.call_args[1]["base_url"] == "https://openrouter.ai/api/v1"

    def test_openrouter_provider_error(self):
        """Verify OpenRouterProvider wraps generic failure in LLMProviderError."""
        provider = OpenRouterProvider(api_key="fake_or_key")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("General network drop")
            mock_openai_cls.return_value = mock_client

            with pytest.raises(LLMProviderError):
                provider.generate("sys", "user")

    def test_get_env_var(self):
        """Verify get_env_var returns environment variable or default."""
        with patch.dict("os.environ", {"TEST_LLM_KEY": "secret_value"}):
            assert get_env_var("TEST_LLM_KEY") == "secret_value"
        assert get_env_var("NON_EXISTENT_KEY_XYZ_123", default="def_val") == "def_val"


# ===========================================================================
# 2. LLMClient Waterfall & Cascade Tests
# ===========================================================================

class TestLLMClientWaterfall:
    """Test LLMClient cascade, fallback tracking, and provider overrides."""

    def test_waterfall_first_succeeds(self):
        """Verify waterfall returns Gemini result immediately when Gemini succeeds."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.name = "gemini"
        mock_gemini.default_model = "gemini-2.5-flash"
        mock_gemini.is_available.return_value = True
        mock_gemini.generate.return_value = "Gemini generated commentary"

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.name = "groq"
        mock_groq.is_available.return_value = True

        client = LLMClient(providers={"gemini": mock_gemini, "groq": mock_groq, "mock": MockProvider()})
        res = client.generate("sys", "user")

        assert res["provider_used"] == "gemini"
        assert res["model_used"] == "gemini-2.5-flash"
        assert res["content"] == "Gemini generated commentary"
        assert "gemini: success" in res["fallback_history"]
        mock_groq.generate.assert_not_called()

    def test_waterfall_failover_to_groq(self):
        """Verify waterfall cascades to Groq when Gemini fails."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.name = "gemini"
        mock_gemini.default_model = "gemini-2.5-flash"
        mock_gemini.is_available.return_value = True
        mock_gemini.generate.side_effect = LLMQuotaError("Gemini quota exceeded")

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.name = "groq"
        mock_groq.default_model = "llama-3.3-70b-versatile"
        mock_groq.is_available.return_value = True
        mock_groq.generate.return_value = "Groq commentary"

        client = LLMClient(providers={"gemini": mock_gemini, "groq": mock_groq, "mock": MockProvider()})
        res = client.generate("sys", "user")

        assert res["provider_used"] == "groq"
        assert res["model_used"] == "llama-3.3-70b-versatile"
        assert res["content"] == "Groq commentary"
        assert any("gemini: failed" in h for h in res["fallback_history"])
        assert "groq: success" in res["fallback_history"]

    def test_waterfall_failover_to_nvidia(self):
        """Verify waterfall cascades to NVIDIA NIM when Gemini and Groq fail."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.is_available.return_value = False

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.is_available.return_value = True
        mock_groq.generate.side_effect = LLMAuthenticationError("Groq key invalid")

        mock_nvidia = MagicMock(spec=BaseProvider)
        mock_nvidia.name = "nvidia"
        mock_nvidia.default_model = "meta/llama-3.1-70b-instruct"
        mock_nvidia.is_available.return_value = True
        mock_nvidia.generate.return_value = "NVIDIA NIM commentary"

        client = LLMClient(providers={
            "gemini": mock_gemini,
            "groq": mock_groq,
            "nvidia": mock_nvidia,
            "mock": MockProvider()
        })
        res = client.generate("sys", "user")

        assert res["provider_used"] == "nvidia"
        assert res["model_used"] == "meta/llama-3.1-70b-instruct"
        assert res["content"] == "NVIDIA NIM commentary"

    def test_waterfall_all_remote_fail_to_mock(self):
        """Verify waterfall seamlessly reaches MockProvider when all external APIs fail."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.is_available.return_value = False

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.is_available.return_value = True
        mock_groq.generate.side_effect = LLMTimeoutError("Groq timeout")

        mock_nvidia = MagicMock(spec=BaseProvider)
        mock_nvidia.is_available.return_value = False

        mock_openrouter = MagicMock(spec=BaseProvider)
        mock_openrouter.is_available.return_value = True
        mock_openrouter.generate.side_effect = LLMProviderError("OpenRouter 502 Bad Gateway")

        client = LLMClient(providers={
            "gemini": mock_gemini,
            "groq": mock_groq,
            "nvidia": mock_nvidia,
            "openrouter": mock_openrouter,
            "mock": MockProvider(),
        })

        res = client.generate("sys", "Asset: ^NSEI\nRegime: Bull")

        assert res["provider_used"] == "mock"
        assert res["model_used"] == "mock-deterministic-v1"
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in res["content"]
        assert len(res["fallback_history"]) >= 4

    def test_provider_override_success(self):
        """Verify provider_override routes directly to specified provider."""
        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.name = "groq"
        mock_groq.default_model = "llama-3.3-70b-versatile"
        mock_groq.is_available.return_value = True
        mock_groq.generate.return_value = "Overridden Groq commentary"

        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.is_available.return_value = True

        client = LLMClient(providers={"gemini": mock_gemini, "groq": mock_groq, "mock": MockProvider()})
        res = client.generate("sys", "user", provider_override="groq")

        assert res["provider_used"] == "groq"
        assert res["content"] == "Overridden Groq commentary"
        mock_gemini.generate.assert_not_called()

    def test_provider_override_with_custom_model(self):
        """Verify provider_override parses custom model syntax (e.g. 'groq:llama-3.1-8b-instant')."""
        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.name = "groq"
        mock_groq.default_model = "llama-3.3-70b-versatile"
        mock_groq.is_available.return_value = True
        mock_groq.generate.return_value = "Groq 8B commentary"

        client = LLMClient(providers={"groq": mock_groq, "mock": MockProvider()})
        res = client.generate("sys", "user", provider_override="groq:llama-3.1-8b-instant")

        assert res["provider_used"] == "groq"
        assert res["model_used"] == "llama-3.1-8b-instant"
        mock_groq.generate.assert_called_once_with(
            system_prompt="sys",
            user_prompt="user",
            model="llama-3.1-8b-instant",
            max_tokens=2048,
            temperature=0.3,
        )

    def test_waterfall_failover_to_openrouter(self):
        """Verify waterfall cascades to OpenRouter when Gemini, Groq, and NVIDIA fail."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.is_available.return_value = False

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.is_available.return_value = False

        mock_nvidia = MagicMock(spec=BaseProvider)
        mock_nvidia.is_available.return_value = True
        mock_nvidia.generate.side_effect = LLMQuotaError("NVIDIA quota exceeded")

        mock_openrouter = MagicMock(spec=BaseProvider)
        mock_openrouter.name = "openrouter"
        mock_openrouter.default_model = "meta-llama/llama-3.3-70b-instruct"
        mock_openrouter.is_available.return_value = True
        mock_openrouter.generate.return_value = "OpenRouter cascade output"

        client = LLMClient(providers={
            "gemini": mock_gemini,
            "groq": mock_groq,
            "nvidia": mock_nvidia,
            "openrouter": mock_openrouter,
            "mock": MockProvider(),
        })
        res = client.generate("sys", "user")

        assert res["provider_used"] == "openrouter"
        assert res["model_used"] == "meta-llama/llama-3.3-70b-instruct"
        assert res["content"] == "OpenRouter cascade output"

    def test_provider_availability_and_discovery(self):
        """Verify get_available_providers returns only ready providers."""
        mock_gemini = MagicMock(spec=BaseProvider)
        mock_gemini.is_available.return_value = True

        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.is_available.return_value = False

        client = LLMClient(providers={"gemini": mock_gemini, "groq": mock_groq, "mock": MockProvider()})
        available = client.get_available_providers()
        assert "gemini" in available
        assert "mock" in available
        assert "groq" not in available

    def test_provider_override_failure_cascades(self):
        """Verify failing provider_override logs error and cascades gracefully."""
        mock_groq = MagicMock(spec=BaseProvider)
        mock_groq.name = "groq"
        mock_groq.is_available.return_value = True
        mock_groq.generate.side_effect = LLMProviderError("Groq crashed")

        client = LLMClient(providers={"groq": mock_groq, "mock": MockProvider()})
        res = client.generate("sys", "Asset: ^NSEI", provider_override="groq")

        assert res["provider_used"] == "mock"
        assert any("groq: failed" in h for h in res["fallback_history"])



# ===========================================================================
# 3. Prompt Formatting & Context Building Tests
# ===========================================================================

class TestPromptFormatting:
    """Test system prompt, context formatters, and user prompt generator."""

    def test_system_prompt_domain_keywords(self):
        """Verify system prompt contains institutional Indian equities domain concepts."""
        sys_prompt = get_analyst_system_prompt()
        assert "NIFTY 50" in sys_prompt
        assert "BFSI" in sys_prompt
        assert "Reserve Bank of India (RBI)" in sys_prompt
        assert "Monetary Policy Committee (MPC)" in sys_prompt
        assert "FII" in sys_prompt
        assert "DII" in sys_prompt
        assert "India VIX" in sys_prompt
        assert "Gaussian Hidden Markov Model" in sys_prompt
        assert "Sharpe Ratio" in sys_prompt
        assert "Monte Carlo" in sys_prompt

        # Verify all 5 section headers are strictly declared
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in sys_prompt
        assert "## 🔄 2. Regime Shift Dynamics & Transition Probability" in sys_prompt
        assert "## 📊 3. Quantitative Strategy Evaluation & Justification" in sys_prompt
        assert "## 🛡️ 4. Risk Budgeting & Drawdown Guardrails" in sys_prompt
        assert "## 🚀 5. Tactical Capital Allocation & Action Plan" in sys_prompt

    def test_format_backtest_metrics_table(self):
        """Verify format_backtest_metrics_table produces valid markdown table."""
        metrics = {
            "Momentum": {
                "CAGR": 0.198,
                "Sharpe": 1.42,
                "Sortino": 2.05,
                "MaxDrawdown": -0.142,
                "Calmar": 1.39,
                "Volatility": 0.142,
            },
            "Buy & Hold": {
                "CAGR": 0.125,
                "Sharpe": 0.85,
                "Sortino": 1.10,
                "MaxDrawdown": -0.284,
                "Calmar": 0.44,
                "Volatility": 0.165,
            },
        }

        table = format_backtest_metrics_table(metrics)
        assert "| Strategy | CAGR | Sharpe | Sortino | Max Drawdown | Calmar | Volatility |" in table
        assert "| Momentum | 19.8% | 1.42 | 2.05 | -14.2% | 1.39 | 14.2% |" in table
        assert "| Buy & Hold | 12.5% | 0.85 | 1.10 | -28.4% | 0.44 | 16.5% |" in table

    def test_format_backtest_metrics_table_empty(self):
        """Verify format_backtest_metrics_table handles empty/None gracefully."""
        assert format_backtest_metrics_table(None) == "*No backtest metrics provided.*"
        assert format_backtest_metrics_table({}) == "*No backtest metrics provided.*"

    def test_format_regime_context(self):
        """Verify format_regime_context produces clean bullet points."""
        regime_data = {
            "current_regime": "Bull",
            "regime_distribution": {"Bull": 1200, "Bear": 450, "Sideways": 800},
            "transition_matrix": [[0.88, 0.05, 0.07], [0.06, 0.82, 0.12], [0.10, 0.08, 0.82]],
            "probabilities": {"Momentum": 0.76, "Dual Momentum": 0.14, "Buy & Hold": 0.10},
            "recommended_strategy": "Momentum",
        }

        context = format_regime_context(regime_data)
        assert "**Current Detected Regime (HMM)**: `Bull`" in context
        assert "Historical Regime Distribution" in context
        assert "Bull: 1200 days" in context
        assert "ML Strategy Suitability Probabilities" in context
        assert "Momentum: 76.0%" in context
        assert "**Recommended Strategy**: `Momentum`" in context

    def test_format_regime_context_empty(self):
        """Verify format_regime_context handles None gracefully."""
        context = format_regime_context(None)
        assert "Current Market Regime: **Bull**" in context

    def test_format_risk_forecast_context(self):
        """Verify format_risk_forecast_context formats Monte Carlo drawdown statistics."""
        risk_metrics = {
            "worst_case_10": -0.124,
            "median_50": -0.058,
            "best_case_90": -0.018,
            "recommended_exposure": "100% (Normal Risk)",
        }

        context = format_risk_forecast_context(risk_metrics)
        assert "**10th Percentile (Tail Risk / Adverse Scenario)**: -12.4%" in context
        assert "**50th Percentile (Median Expected Drawdown)**: -5.8%" in context
        assert "**90th Percentile (Favorable Path Drawdown)**: -1.8%" in context
        assert "**Suggested Exposure Limit**: `100% (Normal Risk)`" in context

    def test_format_risk_forecast_context_empty(self):
        """Verify format_risk_forecast_context handles None gracefully."""
        context = format_risk_forecast_context(None)
        assert "Monte Carlo 63-Day Forward Drawdown Simulation" in context

    def test_build_analyst_user_prompt(self):
        """Verify build_analyst_user_prompt combines all sections and custom prompt."""
        prompt = build_analyst_user_prompt(
            ticker="RELIANCE.NS",
            custom_prompt="Focus on refining margin safety around crude volatility.",
        )
        assert "Asset: RELIANCE.NS" in prompt
        assert "Focus on refining margin safety around crude volatility." in prompt
        assert "1. Gaussian HMM Regime Detection & ML Strategy Suitability" in prompt
        assert "2. Backtest Performance Metrics Across 6 Quantitative Strategies" in prompt
        assert "3. Quantitative Risk Forecast & Monte Carlo Simulation" in prompt


# ===========================================================================
# 4. Service Integration & Analyst Facade Tests
# ===========================================================================

class TestAnalystReportIntegration:
    """Test high-level generate_analyst_report() facade."""

    def test_generate_analyst_report_schema(self):
        """Verify generate_analyst_report returns dict matching LLMReportResponse."""
        result = generate_analyst_report(
            backtest_metrics={
                "Momentum": {"CAGR": 0.198, "Sharpe": 1.42, "Sortino": 2.05, "MaxDrawdown": -0.14, "Calmar": 1.41, "Volatility": 0.142},
                "Buy & Hold": {"CAGR": 0.125, "Sharpe": 0.85, "Sortino": 1.10, "MaxDrawdown": -0.28, "Calmar": 0.44, "Volatility": 0.165},
            },
            regime_data={"current_regime": "Bull", "probabilities": {"Momentum": 0.80}},
            risk_metrics={"worst_case_10": -0.124, "median_50": -0.058, "best_case_90": -0.018, "recommended_exposure": "100%"},
            ticker="^NSEI",
            provider_override="mock",
        )

        assert isinstance(result, dict)
        assert result["ticker"] == "^NSEI"
        assert result["current_regime"] == "Bull"
        assert result["recommended_strategy"] == "Momentum"
        assert result["provider_used"] == "mock"
        assert result["model_used"] == "mock-deterministic-v1"
        assert isinstance(result["report_markdown"], str)
        assert isinstance(result["generated_at"], str)
        assert isinstance(result["fallback_history"], list)

    def test_generate_analyst_report_five_sections(self):
        """Verify generated report markdown contains all 5 required section headers."""
        result = generate_analyst_report(
            ticker="^NSEI",
            provider_override="mock",
        )

        report = result["report_markdown"]
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in report
        assert "## 🔄 2. Regime Shift Dynamics & Transition Probability" in report
        assert "## 📊 3. Quantitative Strategy Evaluation & Justification" in report
        assert "## 🛡️ 4. Risk Budgeting & Drawdown Guardrails" in report
        assert "## 🚀 5. Tactical Capital Allocation & Action Plan" in report

    def test_generate_analyst_report_timestamp_iso(self):
        """Verify generated_at timestamp is valid ISO-8601 UTC string."""
        result = generate_analyst_report(provider_override="mock")
        ts = result["generated_at"]
        dt = datetime.fromisoformat(ts)
        assert dt is not None

    def test_generate_analyst_report_resilience_none_inputs(self):
        """Verify generate_analyst_report executes safely when all arguments are None/empty."""
        result = generate_analyst_report(
            backtest_metrics=None,
            regime_data=None,
            risk_metrics=None,
            ticker="",
            provider_override="mock",
        )

        assert result["ticker"] == "^NSEI"
        assert result["current_regime"] == "Bull"
        assert result["recommended_strategy"] == "Momentum"
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in result["report_markdown"]

    def test_generate_analyst_report_bear_and_sideways_regimes(self):
        """Verify analyst report adapts commentary for Bear and Sideways regimes."""
        bear_result = generate_analyst_report(
            regime_data={"current_regime": "Bear", "recommended_strategy": "RSI Mean Reversion"},
            provider_override="mock",
        )
        assert bear_result["current_regime"] == "Bear"
        assert "Bear" in bear_result["report_markdown"]
        assert "RSI Mean Reversion" in bear_result["report_markdown"]

        sideways_result = generate_analyst_report(
            regime_data={"current_regime": "Sideways", "recommended_strategy": "Bollinger Bands"},
            provider_override="mock",
        )
        assert sideways_result["current_regime"] == "Sideways"
        assert "Sideways" in sideways_result["report_markdown"]
        assert "Bollinger Bands" in sideways_result["report_markdown"]

    def test_generate_analyst_report_custom_prompt(self):
        """Verify custom user prompt is incorporated when requested."""
        result = generate_analyst_report(
            ticker="TCS.NS",
            custom_prompt="Analyze IT sector margin compression from US macroeconomic headwinds.",
            provider_override="mock",
        )
        assert result["ticker"] == "TCS.NS"
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in result["report_markdown"]

    def test_generate_mock_report_direct_call(self):
        """Verify direct invocation of generate_mock_report produces high fidelity markdown."""
        report = generate_mock_report(
            user_prompt="Asset: INFY.NS\nRegime: Bull",
            backtest_metrics={"Momentum": {"CAGR": 0.22, "Sharpe": 1.6, "Sortino": 2.2, "MaxDrawdown": -0.11, "Calmar": 2.0}},
            regime_data={"current_regime": "Bull", "recommended_strategy": "Momentum"},
            risk_metrics={"worst_case_10": -0.09, "median_50": -0.04, "best_case_90": -0.01, "recommended_exposure": "95%"},
            ticker="INFY.NS",
        )
        assert "INFY.NS" in report
        assert "## 🏛️ 1. Executive Market & Regime Diagnosis" in report
        assert "## 🚀 5. Tactical Capital Allocation & Action Plan" in report
        assert "22.0%" in report

