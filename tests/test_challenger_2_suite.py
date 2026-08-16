"""Empirical Adversarial Test Suite for Challenger 2 (Milestone M3).

Focus areas:
1. Prompt formatting & context builder boundary edge cases (None, empty dict, missing keys, NaN/Inf, negative CAGR, extreme -100% drawdown)
2. Ticker format robustness (^BSESN, RELIANCE.NS, UNKNOWN, symbols with punctuation, emojis, quotes)
3. Custom prompts with special characters, markdown tags, XML/HTML, unclosed code blocks, quotes, newlines
4. Deterministic Mock fidelity & dynamic domain narrative tone verification across Bull, Bear, and Sideways regimes
5. Precision and formatting of metrics inside the generated mock report table and prose
"""
from __future__ import annotations

import math
from typing import Any
import pytest

from src.llm.analyst import generate_analyst_report
from src.llm.prompts import (
    build_analyst_user_prompt,
    format_backtest_metrics_table,
    format_regime_context,
    format_risk_forecast_context,
    generate_mock_report,
    get_analyst_system_prompt,
)
from src.llm.client import LLMClient
from src.llm.providers import MockProvider


# ===========================================================================
# 1. Boundary Edge Cases: None, Empty Dicts, Missing Keys, NaN/Inf, Extremes
# ===========================================================================

class TestBoundaryEdgeCases:
    """Test boundary conditions, numeric edge cases, and malformed structures."""

    def test_missing_and_partial_keys_in_metrics(self):
        """Test with partial metrics missing several standard keys (e.g. only CAGR, or only Sharpe)."""
        metrics = {
            "PartialStrat1": {"CAGR": 0.15},
            "PartialStrat2": {"Sharpe": 1.25, "MaxDrawdown": -0.08},
            "EmptyDictStrat": {},
        }
        table = format_backtest_metrics_table(metrics)
        assert "| PartialStrat1 | 15.0% | 0.00 | 0.00 | 0.0% | 0.00 | 0.0% |" in table
        assert "| PartialStrat2 | 0.0% | 1.25 | 0.00 | -8.0% | 0.00 | 0.0% |" in table
        assert "| EmptyDictStrat | 0.0% | 0.00 | 0.00 | 0.0% | 0.00 | 0.0% |" in table

    def test_negative_cagr_and_extreme_100_percent_drawdown(self):
        """Test negative CAGR (-85.4%) and catastrophic -100% drawdown."""
        metrics = {
            "DisasterStrat": {
                "CAGR": -0.854,
                "Sharpe": -2.45,
                "Sortino": -3.12,
                "MaxDrawdown": -1.00,  # 100% loss of capital
                "Calmar": -0.85,
                "Volatility": 0.95,
            },
            "Buy & Hold": {
                "CAGR": -0.15,
                "Sharpe": -0.2,
                "Sortino": -0.3,
                "MaxDrawdown": -0.55,
                "Calmar": -0.27,
                "Volatility": 0.25,
            }
        }
        table = format_backtest_metrics_table(metrics)
        assert "-85.4%" in table
        assert "-100.0%" in table
        assert "-2.45" in table

        # Also test that generate_analyst_report and mock generator dynamically reflect these extreme values
        res = generate_analyst_report(
            backtest_metrics=metrics,
            regime_data={"current_regime": "Bear", "recommended_strategy": "DisasterStrat"},
            risk_metrics={"worst_case_10": -1.0, "median_50": -0.75, "best_case_90": -0.30},
            ticker="^NSEI",
            provider_override="mock",
        )
        assert res["current_regime"] == "Bear"
        assert res["recommended_strategy"] == "DisasterStrat"
        report = res["report_markdown"]
        assert "-100.0%" in report
        assert "-85.4%" in report

    def test_nan_and_inf_handling_in_formatters(self):
        """Test NaN and positive/negative infinity values in floating point fields."""
        metrics = {
            "NaNStrat": {
                "CAGR": float("nan"),
                "Sharpe": float("inf"),
                "Sortino": float("-inf"),
                "MaxDrawdown": float("nan"),
                "Calmar": float("nan"),
                "Volatility": float("inf"),
            }
        }
        table = format_backtest_metrics_table(metrics)
        # Verify table string formats without throwing unhandled exceptions
        assert "| NaNStrat |" in table
        assert "nan" in table or "inf" in table

        risk_metrics = {
            "worst_case_10": float("nan"),
            "median_50": float("inf"),
            "best_case_90": float("-inf"),
        }
        risk_str = format_risk_forecast_context(risk_metrics)
        assert "10th Percentile" in risk_str
        assert "50th Percentile" in risk_str

        # Ensure report generation doesn't crash
        res = generate_analyst_report(
            backtest_metrics=metrics,
            risk_metrics=risk_metrics,
            ticker="^NSEI",
            provider_override="mock",
        )
        assert isinstance(res["report_markdown"], str)

    def test_regime_data_probabilities_formats(self):
        """Test probability parsing when values are > 1.0 (e.g. 75.0 instead of 0.75) or zero."""
        regime_data_pct = {
            "current_regime": "Bull",
            "probabilities": {"Momentum": 85.5, "Buy & Hold": 14.5},
            "recommended_strategy": "Momentum",
        }
        context = format_regime_context(regime_data_pct)
        assert "Momentum: 85.5%" in context
        assert "Buy & Hold: 14.5%" in context

        # Zero probabilities
        regime_data_zero = {
            "current_regime": "Sideways",
            "probabilities": {"Momentum": 0.0, "Bollinger Bands": 1.0},
        }
        context_zero = format_regime_context(regime_data_zero)
        assert "Momentum: 0.0%" in context_zero
        assert "Bollinger Bands: 100.0%" in context_zero


# ===========================================================================
# 2. Non-Standard Tickers Robustness
# ===========================================================================

class TestNonStandardTickers:
    """Test resolution and formatting with various ticker formats."""

    @pytest.mark.parametrize("ticker_input,expected_resolved", [
        ("^NSEI", "^NSEI"),
        ("^BSESN", "^BSESN"),
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("TCS.BO", "TCS.BO"),
        ("UNKNOWN", "UNKNOWN"),
        ("NIFTY_BANK", "NIFTY_BANK"),
        ("HDFCBANK.NS", "HDFCBANK.NS"),
        ("GOLD-BEES.NS", "GOLD-BEES.NS"),
        ("  INFY.NS  ", "INFY.NS"),
        ("", "^NSEI"),  # Empty defaults to ^NSEI
        (None, "^NSEI"),  # None defaults to ^NSEI
        ("   ", "^NSEI"),  # Whitespace only defaults to ^NSEI
    ])
    def test_ticker_resolution_in_analyst_report(self, ticker_input, expected_resolved):
        """Verify ticker symbols are preserved or normalized cleanly."""
        res = generate_analyst_report(
            ticker=ticker_input,
            provider_override="mock",
        )
        assert res["ticker"] == expected_resolved
        # Mock report should reference resolved ticker
        assert expected_resolved in res["report_markdown"]

    def test_complex_ticker_in_mock_report_extraction(self):
        """Test regex ticker extraction from user prompt in generate_mock_report directly."""
        prompt = "### Market Analysis Request for Asset: TATAMOTORS.NS\n\nPlease generate an institutional report..."
        report = generate_mock_report(user_prompt=prompt)
        assert "TATAMOTORS.NS" in report
        assert "- **Primary Ticker / Asset**: `TATAMOTORS.NS`" in report

    def test_bse_sensex_ticker_in_mock_report(self):
        """Test BSE Sensex ticker `^BSESN` in prompt extraction."""
        prompt = "### Market Analysis Request for Asset: ^BSESN\n\nPlease generate an institutional report..."
        report = generate_mock_report(user_prompt=prompt)
        assert "^BSESN" in report
        assert "- **Primary Ticker / Asset**: `^BSESN`" in report


# ===========================================================================
# 3. Custom Prompts with Special Characters, Markdown Tags, and Quotes
# ===========================================================================

class TestCustomPromptsFormatting:
    """Test custom prompt injection with markdown, codeblocks, HTML tags, quotes."""

    @pytest.mark.parametrize("custom_prompt_text", [
        'Focus on "defensive rotation" and "FII debt-equity rebalancing".',
        "Analyze `NIFTY_AUTO` vs `NIFTY_IT` with 2x leverage stop-loss: ```python\nsl = 0.05\n```",
        "<b>Institutional Alert:</b> Check MPC stance on <i>liquidity deficit</i> & <a href='https://rbi.org.in'>RBI</a>.",
        "Special symbols: ₹ 1,50,000 Cr, % allocation, @portfolio_mgr, #RiskManagement, &amp; ~0.85 Sharpe.",
        "Line 1 with text.\n\nLine 2 with bullet point:\n- Bullet A\n- Bullet B\n\nLine 3 with quote:\n> Institutional risk ceiling.",
        "Single 'quotes', double \"quotes\", backticks `code`, and triple ```code```.",
    ])
    def test_custom_prompt_integration(self, custom_prompt_text):
        """Verify custom prompt is cleanly appended into build_analyst_user_prompt."""
        user_prompt = build_analyst_user_prompt(
            ticker="^NSEI",
            custom_prompt=custom_prompt_text,
        )
        assert "### User Custom Request / Specific Focus:" in user_prompt
        assert custom_prompt_text.strip() in user_prompt

        # Verify analyst report runs cleanly
        res = generate_analyst_report(
            ticker="^NSEI",
            custom_prompt=custom_prompt_text,
            provider_override="mock",
        )
        assert res["ticker"] == "^NSEI"
        assert len(res["report_markdown"]) > 500

    def test_custom_prompt_whitespace_only(self):
        """Verify whitespace-only custom prompt is omitted from the prompt."""
        user_prompt = build_analyst_user_prompt(
            ticker="^NSEI",
            custom_prompt="   \n\t  \n  ",
        )
        assert "### User Custom Request / Specific Focus:" not in user_prompt


# ===========================================================================
# 4. Deterministic Mock Fidelity & Narrative Tone by Regime
# ===========================================================================

class TestMockFidelityAndRegimeNarrative:
    """Verify mock report dynamically reflects input metrics and adapts narrative tone."""

    def test_bull_regime_narrative_and_tone(self):
        """Verify Bull regime report tone: momentum expansion, high equity allocation (85-95%), BFSI leadership."""
        regime_data = {
            "current_regime": "Bull",
            "recommended_strategy": "Momentum",
            "probabilities": {"Momentum": 0.78, "Dual Momentum": 0.15, "Buy & Hold": 0.07},
        }
        backtest_metrics = {
            "Momentum": {"CAGR": 0.245, "Sharpe": 1.65, "Sortino": 2.30, "MaxDrawdown": -0.115, "Calmar": 2.13},
            "Buy & Hold": {"CAGR": 0.130, "Sharpe": 0.88, "Sortino": 1.15, "MaxDrawdown": -0.270, "Calmar": 0.48},
        }
        risk_metrics = {
            "worst_case_10": -0.095,
            "median_50": -0.042,
            "best_case_90": -0.010,
            "recommended_exposure": "90% - 100% Equity Exposure",
        }

        res = generate_analyst_report(
            backtest_metrics=backtest_metrics,
            regime_data=regime_data,
            risk_metrics=risk_metrics,
            ticker="^NSEI",
            provider_override="mock",
        )

        report = res["report_markdown"]
        # Regime & Strategy Assertions
        assert "**Bull Regime**" in report
        assert "**Momentum**" in report
        assert "90% - 100% Equity Exposure" in report

        # Dynamic metric insertion verification
        assert "24.5%" in report  # Top CAGR
        assert "1.65" in report   # Top Sharpe
        assert "2.30" in report   # Top Sortino
        assert "-11.5%" in report # Top MaxDD
        assert "2.13" in report   # Top Calmar
        assert "13.0%" in report  # BH CAGR
        assert "0.88" in report   # BH Sharpe
        assert "-27.0%" in report # BH MaxDD

        # Risk metrics dynamic insertion
        assert "-9.5%" in report  # 10th percentile
        assert "-4.2%" in report  # 50th percentile
        assert "-1.0%" in report  # 90th percentile

        # Bull tone assertions
        assert "SIP inflows" in report
        assert "85% - 95% actively deployed" in report
        assert "P(S_{t+1}=\\text{Bull} | S_t=\\text{Bull})" in report

    def test_bear_regime_narrative_and_tone(self):
        """Verify Bear regime report tone: capital preservation, defensive allocation (30-45%), FII distribution."""
        regime_data = {
            "current_regime": "Bear",
            "recommended_strategy": "RSI Mean Reversion",
            "probabilities": {"RSI Mean Reversion": 0.72, "Bollinger Bands": 0.18, "Buy & Hold": 0.10},
        }
        backtest_metrics = {
            "RSI Mean Reversion": {"CAGR": 0.082, "Sharpe": 0.95, "Sortino": 1.40, "MaxDrawdown": -0.098, "Calmar": 0.84},
            "Buy & Hold": {"CAGR": -0.145, "Sharpe": -0.35, "Sortino": -0.45, "MaxDrawdown": -0.385, "Calmar": -0.38},
        }
        risk_metrics = {
            "worst_case_10": -0.245,
            "median_50": -0.140,
            "best_case_90": -0.045,
            "recommended_exposure": "35% (Defensive Mode)",
        }

        res = generate_analyst_report(
            backtest_metrics=backtest_metrics,
            regime_data=regime_data,
            risk_metrics=risk_metrics,
            ticker="^NSEI",
            provider_override="mock",
        )

        report = res["report_markdown"]
        # Regime & Strategy Assertions
        assert "**Bear Regime**" in report
        assert "**RSI Mean Reversion**" in report
        assert "35% (Defensive Mode)" in report

        # Dynamic metric insertion verification
        assert "8.2%" in report   # RSI CAGR
        assert "0.95" in report   # RSI Sharpe
        assert "1.40" in report   # RSI Sortino
        assert "-9.8%" in report  # RSI MaxDD
        assert "0.84" in report   # RSI Calmar
        assert "-14.5%" in report # BH CAGR
        assert "-0.35" in report  # BH Sharpe
        assert "-38.5%" in report # BH MaxDD

        # Risk metrics dynamic insertion
        assert "-24.5%" in report # worst 10
        assert "-14.0%" in report # med 50
        assert "-4.5%" in report  # best 90

        # Bear tone assertions
        assert "capital preservation" in report.lower()
        assert "Foreign Institutional Investors (FII)" in report
        assert "30% - 45%" in report  # Defensive allocation
        assert "P(S_{t+1}=\\text{Bear} | S_t=\\text{Bear})" in report

    def test_sideways_regime_narrative_and_tone(self):
        """Verify Sideways regime report tone: range-bound consolidation, mean-reversion swing baskets (60-70%), options overlay."""
        regime_data = {
            "current_regime": "Sideways",
            "recommended_strategy": "Bollinger Bands",
            "probabilities": {"Bollinger Bands": 0.68, "RSI Mean Reversion": 0.22, "Momentum": 0.10},
        }
        backtest_metrics = {
            "Bollinger Bands": {"CAGR": 0.142, "Sharpe": 1.28, "Sortino": 1.82, "MaxDrawdown": -0.102, "Calmar": 1.39},
            "Buy & Hold": {"CAGR": 0.055, "Sharpe": 0.45, "Sortino": 0.60, "MaxDrawdown": -0.180, "Calmar": 0.31},
        }
        risk_metrics = {
            "worst_case_10": -0.110,
            "median_50": -0.050,
            "best_case_90": -0.015,
            "recommended_exposure": "65% Equity / 35% Cash",
        }

        res = generate_analyst_report(
            backtest_metrics=backtest_metrics,
            regime_data=regime_data,
            risk_metrics=risk_metrics,
            ticker="^NSEI",
            provider_override="mock",
        )

        report = res["report_markdown"]
        # Regime & Strategy Assertions
        assert "**Sideways / Mean-Reverting Regime**" in report
        assert "**Bollinger Bands**" in report
        assert "65% Equity / 35% Cash" in report

        # Dynamic metric insertion verification
        assert "14.2%" in report  # BB CAGR
        assert "1.28" in report   # BB Sharpe
        assert "1.82" in report   # BB Sortino
        assert "-10.2%" in report # BB MaxDD
        assert "1.39" in report   # BB Calmar
        assert "5.5%" in report   # BH CAGR
        assert "0.45" in report   # BH Sharpe
        assert "-18.0%" in report # BH MaxDD

        # Sideways tone assertions
        assert "range-bound" in report.lower()
        assert "60% - 70% deployed in mean-reversion swing baskets" in report
        assert "Iron Condors" in report
        assert "P(S_{t+1}=\\text{Sideways} | S_t=\\text{Sideways})" in report

    def test_default_fallbacks_when_metrics_empty(self):
        """Verify that when no metrics are passed, default institutional baseline values are rendered safely."""
        res = generate_analyst_report(
            backtest_metrics=None,
            regime_data=None,
            risk_metrics=None,
            ticker="^NSEI",
            provider_override="mock",
        )
        report = res["report_markdown"]
        # Default baseline values in generate_mock_report
        assert "18.6%" in report  # top cagr default
        assert "1.38" in report   # top sharpe default
        assert "12.4%" in report  # bh cagr default
        assert "-12.4%" in report # worst case 10 default


# ===========================================================================
# 5. Full Pipeline & Contract Compliance
# ===========================================================================

class TestFullContractCompliance:
    """Test full schema validation and output structure compliance."""

    def test_llm_report_response_keys_and_types(self):
        """Verify return dictionary strictly matches the LLMReportResponse schema requirements."""
        res = generate_analyst_report(ticker="TCS.NS", provider_override="mock")

        assert isinstance(res, dict)
        expected_keys = {
            "ticker",
            "current_regime",
            "recommended_strategy",
            "provider_used",
            "model_used",
            "report_markdown",
            "generated_at",
            "fallback_history",
        }
        assert set(res.keys()) == expected_keys
        assert isinstance(res["ticker"], str)
        assert isinstance(res["current_regime"], str)
        assert isinstance(res["recommended_strategy"], str)
        assert isinstance(res["provider_used"], str)
        assert isinstance(res["model_used"], str)
        assert isinstance(res["report_markdown"], str)
        assert isinstance(res["generated_at"], str)
        assert isinstance(res["fallback_history"], list)
