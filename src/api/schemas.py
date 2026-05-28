"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    ticker: str = Field(default="^NSEI", description="yfinance ticker symbol")
    start_date: str = Field(default="2015-01-01", description="Start date YYYY-MM-DD")
    end_date: str = Field(default="2024-12-31", description="End date YYYY-MM-DD")
    strategy: str = Field(
        default="all",
        description="Strategy name or 'all'. Options: Buy & Hold, MA Crossover, RSI, Momentum, Bollinger Bands, Dual Momentum",
    )
    commission_pct: float = Field(default=0.0, description="Transaction commission percentage (0.01 = 1%)")
    slippage_pct: float = Field(default=0.0, description="Execution slippage percentage (0.01 = 1%)")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(default="^NSEI", description="yfinance ticker symbol")
    start_date: str = Field(default="2015-01-01", description="Start date YYYY-MM-DD")
    end_date: str = Field(default="2024-12-31", description="End date YYYY-MM-DD")
    strategy: str = Field(
        default="all",
        description="Strategy to focus on, or 'all' for full comparison",
    )
    initial_investment: float = Field(default=100000, description="Initial investment in ₹")
    commission_pct: float = Field(default=0.0, description="Transaction commission percentage")
    slippage_pct: float = Field(default=0.0, description="Execution slippage percentage")


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    CAGR: float
    Sharpe: float
    Sortino: float = 0.0
    MaxDrawdown: float
    Calmar: float = 0.0
    Volatility: float


class StrategyResult(BaseModel):
    strategy: str
    metrics: MetricsResponse
    equity_curve_start: float
    equity_curve_end: float
    n_days: int


class BacktestResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_trading_days: int
    results: list[StrategyResult]


class RegimeResponse(BaseModel):
    current_regime: str
    regime_distribution: dict[str, int]
    total_days: int


class StrategyProbability(BaseModel):
    strategy: str
    probability: float


class RiskForecast(BaseModel):
    worst_case_10: float
    median_50: float
    best_case_90: float


class MarketOutlook(BaseModel):
    outlook: str
    probability: float
    confidence: str
    expected_volatility: str
    horizon: str
    disclaimer: str



class RecommendResponse(BaseModel):
    current_regime: str
    recommended_strategy: str
    recommendation_source: str
    recommended_exposure: str
    probabilities: list[StrategyProbability]
    risk_forecast: RiskForecast | None = None


class AnalyzeResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_trading_days: int
    initial_investment: float
    current_regime: str
    recommended_strategy: str
    recommendation_source: str
    recommendation_reason: str
    recommended_exposure: str
    probabilities: dict[str, float]
    overall_metrics: dict[str, MetricsResponse]
    equity_curves: dict[str, list[dict]]
    ohlc_data: list[dict]  # OHLC for candlestick chart
    regime_heatmap: list[dict]
    regime_timeline: list[dict]
    risk_forecast: dict | None = None
    market_outlook: MarketOutlook | None = None
