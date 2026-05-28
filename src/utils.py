"""Utility Functions — Portfolio metrics and helpers.

Purpose:
	Shared utility functions for computing portfolio performance metrics (CAGR, Sharpe,
	Max Drawdown, Volatility). Used by backtester and regime performance analysis.

Key Functions:
	- calculate_cagr(equity_curve) → float
	- calculate_sharpe_ratio(portfolio_returns) → float
	- calculate_max_drawdown(equity_curve) → float
	- calculate_volatility(portfolio_returns) → float
	- compute_backtest_metrics(equity_curve, portfolio_returns) → dict

Constants:
	- TRADING_DAYS_PER_YEAR = 252 (Indian equity market)
	- RISK_FREE_RATE = 0.06 (6% for India)

Assumptions:
	- All inputs are daily time series
	- Equity curves are normalized (start at 1.0)
	- Daily returns are log-space or simple returns (< 1% daily, approx equal)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.06


def calculate_cagr(equity_curve: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
	"""Annualized growth rate from normalized equity curve."""
	if equity_curve.empty:
		return 0.0

	start_value = float(equity_curve.iloc[0])
	end_value = float(equity_curve.iloc[-1])
	if start_value <= 0.0 or end_value <= 0.0:
		return 0.0

	periods = max(len(equity_curve) - 1, 1)
	years = periods / trading_days
	if years <= 0.0:
		return 0.0

	return (end_value / start_value) ** (1.0 / years) - 1.0


def calculate_sharpe_ratio(
	portfolio_returns: pd.Series,
	risk_free_rate: float = RISK_FREE_RATE,
	trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
	"""Annualized Sharpe ratio for daily returns."""
	if portfolio_returns.empty:
		return 0.0

	excess_returns = portfolio_returns - (risk_free_rate / trading_days)
	standard_deviation = float(portfolio_returns.std(ddof=0))
	if standard_deviation < 1e-8:
		return 0.0

	return float(excess_returns.mean() / standard_deviation * np.sqrt(trading_days))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
	"""Worst peak-to-trough decline from an equity curve."""
	if equity_curve.empty:
		return 0.0

	running_peak = equity_curve.cummax()
	drawdowns = equity_curve / running_peak - 1.0
	return float(drawdowns.min())


def calculate_volatility(
	portfolio_returns: pd.Series,
	trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
	"""Annualized volatility from daily strategy returns."""
	if portfolio_returns.empty:
		return 0.0

	return float(portfolio_returns.std(ddof=0) * np.sqrt(trading_days))


def calculate_sortino_ratio(
	portfolio_returns: pd.Series,
	risk_free_rate: float = RISK_FREE_RATE,
	trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
	"""Annualized Sortino ratio — penalizes only downside volatility."""
	if portfolio_returns.empty:
		return 0.0

	mar = risk_free_rate / trading_days
	excess_returns = portfolio_returns - mar
	
	# Downside deviation: sqrt(mean(min(0, returns - MAR)^2))
	downside_diff = excess_returns.copy()
	downside_diff[downside_diff > 0] = 0
	downside_deviation = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(trading_days)
	
	if downside_deviation < 1e-8:
		return 0.0

	return float(excess_returns.mean() * trading_days / downside_deviation)


def calculate_calmar_ratio(
	equity_curve: pd.Series,
	trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
	"""Calmar ratio = CAGR / |Max Drawdown|."""
	cagr = calculate_cagr(equity_curve, trading_days=trading_days)
	max_dd = abs(calculate_max_drawdown(equity_curve))
	if max_dd < 1e-8:
		return 0.0
	return cagr / max_dd


def compute_backtest_metrics(
	equity_curve: pd.Series,
	portfolio_returns: pd.Series,
	risk_free_rate: float = RISK_FREE_RATE,
	trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
	"""Bundle core backtest metrics in one place."""
	return {
		"CAGR": calculate_cagr(equity_curve, trading_days=trading_days),
		"Sharpe": calculate_sharpe_ratio(
			portfolio_returns,
			risk_free_rate=risk_free_rate,
			trading_days=trading_days,
		),
		"Sortino": calculate_sortino_ratio(
			portfolio_returns,
			risk_free_rate=risk_free_rate,
			trading_days=trading_days,
		),
		"MaxDrawdown": calculate_max_drawdown(equity_curve),
		"Calmar": calculate_calmar_ratio(equity_curve, trading_days=trading_days),
		"Volatility": calculate_volatility(portfolio_returns, trading_days=trading_days),
	}