"""Global pytest configuration and shared fixtures for the Indian Market Portfolio Intelligence platform.

Provides:
- Deterministic synthetic OHLCV data generators (Normal, Bull, Bear, Sideways, High Volatility, Single Row, Empty).
- Isolated SQLite in-memory database fixtures for SQLAlchemy persistence testing without requiring live PostgreSQL.
- Mock LLM analyst providers and deterministic markdown financial commentary fixtures.
- FastAPI TestClient fixtures for REST API endpoints.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Synthetic Market Data Generator
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    n_bars: int = 500,
    start_date: str = "2020-01-01",
    trend: str = "normal",
    base_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV time-series data with technical features.
    
    Args:
        n_bars: Number of trading bars to generate.
        start_date: Initial date in YYYY-MM-DD.
        trend: Market scenario ('normal', 'bull', 'bear', 'sideways', 'extreme_volatility', 'single', 'empty').
        base_price: Initial close price.
        seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame with DatetimeIndex and columns:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'momentum', 'drawdown']
    """
    if trend == "empty" or n_bars <= 0:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "returns", "volatility", "momentum", "drawdown"],
            index=pd.DatetimeIndex([], name="Date"),
            dtype=float,
        )

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_bars)

    if trend == "single":
        df = pd.DataFrame(
            {
                "Open": [base_price],
                "High": [base_price * 1.01],
                "Low": [base_price * 0.99],
                "Close": [base_price],
                "Volume": [2_500_000],
                "returns": [0.0],
                "volatility": [0.0],
                "momentum": [0.0],
                "drawdown": [0.0],
            },
            index=dates[:1],
        )
        return df

    # Configure drift and volatility parameters based on market regime scenario
    if trend == "bull":
        drift = 0.0012  # Strong positive drift
        vol = 0.010    # Low to moderate volatility
        jumps = rng.choice([0.0, 0.015], size=n_bars, p=[0.95, 0.05])
    elif trend == "bear":
        drift = -0.0015  # Negative drift
        vol = 0.022     # High volatility
        jumps = rng.choice([0.0, -0.025], size=n_bars, p=[0.93, 0.07])
    elif trend == "sideways":
        drift = 0.0000  # Zero drift
        vol = 0.009     # Moderate volatility, mean reverting
        jumps = np.zeros(n_bars)
    elif trend == "extreme_volatility":
        drift = -0.0005
        vol = 0.035     # Extreme volatility
        # Introduce fat-tailed flash shocks
        jumps = rng.choice([0.0, -0.08, 0.06, -0.12], size=n_bars, p=[0.90, 0.04, 0.04, 0.02])
    else:  # normal
        drift = 0.0004
        vol = 0.014
        jumps = np.zeros(n_bars)

    # Generate log returns
    raw_returns = rng.normal(drift, vol, n_bars) + jumps
    if trend == "sideways":
        # Add strong mean-reversion component
        log_price = np.zeros(n_bars)
        log_price[0] = np.log(base_price)
        theta = 0.08  # Mean reversion speed
        for t in range(1, n_bars):
            log_price[t] = log_price[t - 1] + theta * (np.log(base_price) - log_price[t - 1]) + raw_returns[t]
        close = np.exp(log_price)
    else:
        close = base_price * np.cumprod(1.0 + raw_returns)

    # Intraday ranges
    high_noise = np.abs(rng.normal(0, vol * 0.7, n_bars))
    low_noise = np.abs(rng.normal(0, vol * 0.7, n_bars))
    open_noise = rng.uniform(-vol * 0.4, vol * 0.4, n_bars)

    open_price = close * (1.0 + open_noise)
    high_price = np.maximum(open_price, close) * (1.0 + high_noise)
    low_price = np.minimum(open_price, close) * (1.0 - low_noise)
    volume = rng.integers(1_000_000, 15_000_000, size=n_bars)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    # Calculate engineered features
    df["returns"] = df["Close"].pct_change().fillna(0.0)
    df["volatility"] = df["returns"].rolling(20, min_periods=1).std().fillna(0.0)
    df["momentum"] = df["Close"].pct_change(60).fillna(0.0)
    
    # Drawdown calculation
    cummax = df["Close"].cummax()
    df["drawdown"] = (df["Close"] - cummax) / cummax

    return df


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Standard 500-bar synthetic NIFTY 50 OHLCV dataset with feature columns."""
    return generate_synthetic_ohlcv(n_bars=500, trend="normal", seed=42)


@pytest.fixture
def bull_market_df() -> pd.DataFrame:
    """500-bar upward trending Bull regime dataset."""
    return generate_synthetic_ohlcv(n_bars=500, trend="bull", seed=101)


@pytest.fixture
def bear_market_df() -> pd.DataFrame:
    """500-bar downward trending Bear regime dataset with high drawdown."""
    return generate_synthetic_ohlcv(n_bars=500, trend="bear", seed=202)


@pytest.fixture
def sideways_market_df() -> pd.DataFrame:
    """500-bar oscillating Sideways regime dataset."""
    return generate_synthetic_ohlcv(n_bars=500, trend="sideways", seed=303)


@pytest.fixture
def extreme_volatility_df() -> pd.DataFrame:
    """500-bar market dataset featuring extreme shocks and flash crashes."""
    return generate_synthetic_ohlcv(n_bars=500, trend="extreme_volatility", seed=404)


@pytest.fixture
def single_point_df() -> pd.DataFrame:
    """Single-row DataFrame for boundary testing."""
    return generate_synthetic_ohlcv(n_bars=1, trend="single", seed=505)


@pytest.fixture
def empty_market_df() -> pd.DataFrame:
    """0-row empty DataFrame with correct column schema and DatetimeIndex."""
    return generate_synthetic_ohlcv(n_bars=0, trend="empty")


@pytest.fixture
def multi_asset_market_data() -> dict[str, pd.DataFrame]:
    """Dictionary mapping Indian equity/sector tickers to synthetic market datasets."""
    sectors = {
        "^NSEI": ("normal", 42),
        "^NSEBANK": ("bull", 101),
        "^CNXIT": ("bear", 202),
        "^CNXAUTO": ("sideways", 303),
        "^CNXFMCG": ("normal", 404),
        "^CNXENERGY": ("extreme_volatility", 505),
    }
    return {
        ticker: generate_synthetic_ohlcv(n_bars=400, trend=trend, seed=seed)
        for ticker, (trend, seed) in sectors.items()
    }


# ---------------------------------------------------------------------------
# Database Isolation Fixtures (SQLAlchemy 2.0 SQLite In-Memory)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def test_db_engine():
    """Create an isolated in-memory SQLite engine for testing."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        yield engine
        engine.dispose()
    except ImportError:
        # Fallback if sqlalchemy is not yet available in testing context
        yield None


@pytest.fixture(scope="function")
def isolated_db_session(test_db_engine) -> Generator[Any, None, None]:
    """Provide a clean, isolated SQLAlchemy Session with all tables created."""
    if test_db_engine is None:
        yield None
        return

    try:
        from sqlalchemy.orm import Session, sessionmaker

        # Try to import Base from src.db.models if present, else create ad-hoc schema
        try:
            from db.models import Base
            Base.metadata.create_all(bind=test_db_engine)
        except (ImportError, AttributeError):
            from sqlalchemy.orm import declarative_base
            Base = declarative_base()
            Base.metadata.create_all(bind=test_db_engine)

        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.rollback()
            session.close()
            try:
                Base.metadata.drop_all(bind=test_db_engine)
            except Exception:
                pass
    except ImportError:
        yield None


# ---------------------------------------------------------------------------
# Mock LLM Analyst Fixtures
# ---------------------------------------------------------------------------

class DeterministicMockLLMProvider:
    """Mock LLM Provider for deterministic financial commentary generation."""

    def __init__(self, provider_name: str = "MockLLM"):
        self.provider_name = provider_name
        self.call_history: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        metrics: dict[str, Any] | None = None,
        regime_data: dict[str, Any] | None = None,
        risk_data: dict[str, Any] | None = None,
    ) -> str:
        self.call_history.append(
            {
                "prompt": prompt,
                "metrics": metrics,
                "regime_data": regime_data,
                "risk_data": risk_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        regime = (regime_data or {}).get("current_regime", "Bull")
        strategy = (metrics or {}).get("strategy", "Buy & Hold")
        cagr = (metrics or {}).get("CAGR", 0.15)
        sharpe = (metrics or {}).get("Sharpe", 1.25)
        max_dd = (metrics or {}).get("MaxDrawdown", -0.12)

        return f"""# Market Analysis & Strategy Commentary

## Executive Summary
The Indian Equity Market (NIFTY 50) is currently operating under a **{regime}** market regime with established transition dynamics. The quantitative recommendation model selects **{strategy}** as the primary risk-adjusted strategy.

## Regime Context & Transition Dynamics
- **Identified Regime**: {regime}
- **Volatility Character**: Low-to-moderate historical realized volatility.
- **Stationary Persistence**: The current regime displays strong state persistence (>80% transition probability).

## Strategy Recommendation & Rationale
- **Selected Strategy**: {strategy}
- **Expected Metrics**: CAGR of {cagr:.2%}, Sharpe Ratio of {sharpe:.2f}, and Max Drawdown of {max_dd:.2%}.
- **Capital Allocation**: 100% allocation with disciplined stop-loss risk management.

## Risk & Downside Analysis
Monte Carlo simulation suggests downside is contained within historical drawdowns under normal distribution assumptions.
"""


@pytest.fixture
def mock_llm_provider() -> DeterministicMockLLMProvider:
    """Provide a deterministic Mock LLM analyst provider instance."""
    return DeterministicMockLLMProvider()


@pytest.fixture
def mock_analyst_report() -> dict[str, Any]:
    """Sample structured AI Market Analyst report response."""
    return {
        "report_markdown": """# Market Analysis & Strategy Commentary

## Executive Summary
The Indian Equity Market is exhibiting favorable momentum in a Bull regime.

## Regime Context
- **Current Regime**: Bull
- **State Transition Probability**: 0.88

## Strategy Recommendation
- **Recommended Strategy**: Momentum Strategy
- **Sharpe Ratio**: 1.65
- **Max Drawdown**: -9.5%

## Downside Risk Forecast
90% VaR indicates limited tail-risk under the current macroeconomic regime.
""",
        "provider_used": "MockLLM",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "gemini-2.5-flash-mock",
    }


# ---------------------------------------------------------------------------
# API TestClient Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """FastAPI TestClient fixture configured for REST API endpoints."""
    from fastapi.testclient import TestClient
    try:
        from api.main import app
        return TestClient(app)
    except Exception:
        # Fallback if imported from main
        try:
            from main import app
            return TestClient(app)
        except Exception:
            return None
