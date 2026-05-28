"""Minimal test suite for core modules.

Covers: backtester, strategies, regime_detector, risk_forecaster.
Run: uv run pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtester import run_backtest
from strategies import (
    buy_and_hold, ma_crossover, rsi_strategy, momentum_strategy,
    bollinger_bands, dual_momentum,
)
from risk_forecaster import simulate_drawdowns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """~500 trading days of synthetic OHLCV data with features."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))
    df = pd.DataFrame({
        "Open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "Low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df["Close"].pct_change(60)
    df["drawdown"] = df["Close"] / df["Close"].cummax() - 1.0
    return df.dropna()


# ---------------------------------------------------------------------------
# Backtester Tests
# ---------------------------------------------------------------------------

class TestBacktester:
    def test_metrics_keys(self, sample_df: pd.DataFrame):
        signals = pd.Series(1, index=sample_df.index, dtype=int)
        result = run_backtest(sample_df["Close"], signals)
        assert "metrics" in result
        assert "equity_curve" in result
        for key in ("CAGR", "Sharpe", "MaxDrawdown", "Volatility"):
            assert key in result["metrics"], f"Missing metric: {key}"

    def test_signal_lag(self, sample_df: pd.DataFrame):
        """Signal at T should affect position at T+1 (no lookahead)."""
        signals = pd.Series(0, index=sample_df.index, dtype=int)
        # Set signal to 1 only on first day
        signals.iloc[0] = 1
        result = run_backtest(sample_df["Close"], signals)
        # Equity curve should barely move (only 1 day of exposure)
        equity = result["equity_curve"]
        assert abs(equity.iloc[-1] / equity.iloc[0] - 1.0) < 0.05

    def test_all_flat_returns_zero(self, sample_df: pd.DataFrame):
        """All-zero signals → flat → near-zero return."""
        signals = pd.Series(0, index=sample_df.index, dtype=int)
        result = run_backtest(sample_df["Close"], signals)
        assert abs(result["metrics"]["CAGR"]) < 0.01


# ---------------------------------------------------------------------------
# Strategy Tests
# ---------------------------------------------------------------------------

class TestStrategies:
    ALL_STRATEGIES = [
        ("buy_and_hold", buy_and_hold),
        ("ma_crossover", ma_crossover),
        ("rsi_strategy", rsi_strategy),
        ("momentum_strategy", momentum_strategy),
        ("bollinger_bands", bollinger_bands),
        ("dual_momentum", dual_momentum),
    ]

    @pytest.mark.parametrize("name,func", ALL_STRATEGIES)
    def test_signal_values_binary(self, name, func, sample_df):
        """All strategy signals must be 0 or 1."""
        signals = func(sample_df)
        unique = set(signals.unique())
        assert unique <= {0, 1}, f"{name} returned non-binary signals: {unique}"

    @pytest.mark.parametrize("name,func", ALL_STRATEGIES)
    def test_signal_length_matches_input(self, name, func, sample_df):
        """Signal length must equal input length."""
        signals = func(sample_df)
        assert len(signals) == len(sample_df), f"{name} length mismatch"

    def test_buy_and_hold_always_one(self, sample_df):
        signals = buy_and_hold(sample_df)
        assert signals.sum() == len(sample_df)


# ---------------------------------------------------------------------------
# Risk Forecaster Tests
# ---------------------------------------------------------------------------

class TestRiskForecaster:
    def test_output_keys(self):
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        result = simulate_drawdowns(returns, n_simulations=100, horizon=63)
        for key in ("worst_case_10", "median_50", "best_case_90"):
            assert key in result

    def test_drawdowns_negative(self):
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        result = simulate_drawdowns(returns, n_simulations=100, horizon=63)
        assert result["worst_case_10"] <= 0
        assert result["median_50"] <= 0
        assert result["best_case_90"] <= 0

    def test_reproducible_with_seed(self):
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        r1 = simulate_drawdowns(returns, random_seed=42)
        r2 = simulate_drawdowns(returns, random_seed=42)
        assert r1 == r2

    def test_empty_returns(self):
        returns = pd.Series([], dtype=float)
        result = simulate_drawdowns(returns)
        assert result["worst_case_10"] == 0.0
