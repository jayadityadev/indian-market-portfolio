"""Strategy Library — Technical analysis trading signals.

Purpose:
	Implements 6 trading strategies for backtesting. Each generates buy/sell signals
	based on technical indicators. Used by backtester to simulate performance.

Strategies Implemented:
	1. Buy & Hold: Always long (baseline benchmark)
	2. MA Crossover: Long when 50-day SMA > 200-day SMA
	3. RSI: Long in oversold (<30), flat in overbought (>70), hold in between
	4. Momentum: Long when 252-day momentum is positive
	5. Bollinger Bands: Long when price < lower band (mean - 2σ), flat when > upper band
	6. Dual Momentum: Long when both 12-month and 6-month momentum are positive

Inputs:
	- DataFrame with 'Close' column and DatetimeIndex

Outputs:
	- pd.Series of 0/1 signals with DatetimeIndex

Key Functions:
	- buy_and_hold(df), ma_crossover(df), rsi_strategy(df), momentum_strategy(df)
	- bollinger_bands(df), dual_momentum(df)
	- run_all_strategies(df) → dict with all 6 strategy results

Note:
	All signals computed without lookahead bias. Signal at time t uses only data up to t.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from backtester import run_backtest


def _require_close(df: pd.DataFrame) -> pd.Series:
	if "Close" not in df.columns:
		raise KeyError("df must contain a 'Close' column.")
	return df["Close"].astype(float)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
	"""Always-long baseline benchmark."""
	close = _require_close(df)
	return pd.Series(1, index=close.index, dtype=int, name="buy_and_hold")


def ma_crossover(df: pd.DataFrame) -> pd.Series:
	"""Long when 50-day SMA is above 200-day SMA."""
	close = _require_close(df)
	sma_50 = close.rolling(window=50, min_periods=50).mean()
	sma_200 = close.rolling(window=200, min_periods=200).mean()
	signal = (sma_50 > sma_200).fillna(False).astype(int)
	signal.name = "ma_crossover"
	return signal


def rsi_strategy(df: pd.DataFrame) -> pd.Series:
	"""Long in oversold zones, flat in overbought zones, hold in between."""
	close = _require_close(df)
	delta = close.diff()
	gain = delta.clip(lower=0.0)
	loss = -delta.clip(upper=0.0)
	period = 14
	alpha = 1.0 / period
	avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
	avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
	rs = avg_gain / avg_loss
	rsi = 100.0 - (100.0 / (1.0 + rs))

	signal = pd.Series(np.nan, index=close.index, dtype=float)
	signal[rsi < 30.0] = 1.0
	signal[rsi > 70.0] = 0.0
	signal = signal.ffill().fillna(0.0).astype(int)
	signal.name = "rsi_strategy"
	return signal


def momentum_strategy(df: pd.DataFrame) -> pd.Series:
	"""Long when 252-day momentum is positive."""
	close = _require_close(df)
	momentum = close / close.shift(252) - 1.0
	signal = (momentum > 0.0).fillna(False).astype(int)
	signal.name = "momentum_strategy"
	return signal


def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
	"""Long when price drops below lower Bollinger Band, flat above upper band.

	Mean-reversion strategy. Works best in sideways / range-bound markets.
	Buys dips (price < mean - 2σ), sells rips (price > mean + 2σ), holds in between.
	"""
	close = _require_close(df)
	rolling_mean = close.rolling(window=window, min_periods=window).mean()
	rolling_std = close.rolling(window=window, min_periods=window).std()
	lower_band = rolling_mean - num_std * rolling_std
	upper_band = rolling_mean + num_std * rolling_std

	signal = pd.Series(np.nan, index=close.index, dtype=float)
	signal[close < lower_band] = 1.0  # oversold → buy
	signal[close > upper_band] = 0.0  # overbought → sell
	signal = signal.ffill().fillna(0.0).astype(int)
	signal.name = "bollinger_bands"
	return signal


def dual_momentum(df: pd.DataFrame) -> pd.Series:
	"""Long when both 12-month and 6-month momentum are positive.

	Combines absolute momentum (is the trend up?) with intermediate momentum
	(is acceleration positive?) for crash protection. Goes flat if either
	momentum measure is negative — avoids holding through drawdowns.
	"""
	close = _require_close(df)
	mom_12m = close / close.shift(252) - 1.0  # 12-month absolute momentum
	mom_6m = close / close.shift(126) - 1.0   # 6-month intermediate momentum
	# Long only when BOTH are positive
	signal = ((mom_12m > 0.0) & (mom_6m > 0.0)).fillna(False).astype(int)
	signal.name = "dual_momentum"
	return signal


def run_all_strategies(df: pd.DataFrame) -> dict[str, dict[str, object]]:
	"""Run all strategies through backtester and return results by name."""
	strategy_functions: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
		"Buy & Hold": buy_and_hold,
		"MA Crossover": ma_crossover,
		"RSI": rsi_strategy,
		"Momentum": momentum_strategy,
		"Bollinger Bands": bollinger_bands,
		"Dual Momentum": dual_momentum,
	}

	results: dict[str, dict[str, object]] = {}
	for strategy_name, strategy_function in strategy_functions.items():
		signals = strategy_function(df)
		results[strategy_name] = run_backtest(_require_close(df), signals)

	return results


def format_strategy_comparison(results: dict[str, dict[str, object]]) -> pd.DataFrame:
	"""Build a compact comparison table for strategy outputs."""
	rows: list[dict[str, float | str]] = []
	for strategy_name, result in results.items():
		metrics = result["metrics"]
		rows.append(
			{
				"Strategy": strategy_name,
				"CAGR": float(metrics["CAGR"]),
				"Sharpe": float(metrics["Sharpe"]),
				"MaxDrawdown": float(metrics["MaxDrawdown"]),
				"Volatility": float(metrics["Volatility"]),
			}
		)

	comparison = pd.DataFrame(rows).set_index("Strategy")
	return comparison.sort_index()