"""Backtesting Engine — Portfolio simulation and metric computation.

Purpose:
	Core engine that converts strategy signals into portfolio equity curves and computes
	performance metrics (CAGR, Sharpe, Max Drawdown, Volatility). Handles signal lag
	(entry at next day's open, not same candle) to avoid look-ahead bias.

Inputs:
	- Prices: pd.Series of daily close prices with DatetimeIndex
	- Signals: pd.Series of 0/1 trading signals with matching DatetimeIndex

Outputs:
	- Dictionary with:
		- 'metrics': dict of CAGR, Sharpe, MaxDrawdown, Volatility
		- 'equity_curve': pd.Series of normalized portfolio value over time

Key Functions:
	- run_backtest(prices, signals) → dict with metrics and equity_curve

Assumptions:
	- Long-only portfolio (signals are 1=long, 0=flat, not -1=short)
	- No transaction costs modeled (MVP scope)
	- No leverage
	- Daily rebalancing based on signal changes
"""
from __future__ import annotations

import pandas as pd

from data_pipeline import run_pipeline
from utils import compute_backtest_metrics


def _validate_backtest_inputs(prices: pd.Series, signals: pd.Series) -> None:
	if not isinstance(prices, pd.Series):
		raise TypeError("prices must be a pandas Series.")
	if not isinstance(signals, pd.Series):
		raise TypeError("signals must be a pandas Series.")
	if len(prices) != len(signals):
		raise ValueError(
			f"prices and signals must have same length, got {len(prices)} and {len(signals)}."
		)
	if not prices.index.equals(signals.index):
		raise ValueError("prices and signals must share same DatetimeIndex in same order.")
	if prices.empty:
		raise ValueError("prices cannot be empty.")
	if prices.isnull().any():
		raise ValueError("prices cannot contain nulls.")
	if signals.isnull().any():
		raise ValueError("signals cannot contain nulls.")

	unique_values = set(pd.Series(signals).astype(float).unique())
	if not unique_values.issubset({0.0, 1.0}):
		raise ValueError("signals must contain only 0/1 values.")


def run_backtest(
	prices: pd.Series,
	signals: pd.Series,
	commission_pct: float = 0.0,
	slippage_pct: float = 0.0,
) -> dict[str, object]:
	"""Run long-only backtest with one-day signal lag and transaction costs."""
	_validate_backtest_inputs(prices, signals)

	price_series = prices.astype(float)
	signal_series = signals.astype(float)
	daily_returns = price_series.pct_change().fillna(0.0)

	# 1-day lag: signals generated at close of day t-1 are applied to return of day t
	shifted_signals = signal_series.shift(1).fillna(0.0)

	# Transaction costs: applied on entry (0->1) and exit (1->0)
	# total_cost is the sum of commission and slippage
	total_cost = commission_pct + slippage_pct
	trades = shifted_signals.diff().fillna(0.0).abs()

	# Net returns = (Exposure * Market Return) - (Turnover * Cost)
	portfolio_returns = (shifted_signals * daily_returns) - (trades * total_cost)

	equity_curve = (1.0 + portfolio_returns).cumprod()
	equity_curve.name = "equity_curve"
	portfolio_returns.name = "portfolio_returns"

	metrics = compute_backtest_metrics(equity_curve, portfolio_returns)
	return {"metrics": metrics, "equity_curve": equity_curve}
