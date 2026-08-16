"""SQLAlchemy 2.0 Declarative ORM models for backtests, regimes, and benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 ORM models."""
    pass


class BacktestLog(Base):
    """Execution record of single and comparative backtest runs."""
    __tablename__ = "backtest_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    ticker: Mapped[str] = mapped_column(String(32), default="^NSEI", nullable=False, index=True)
    start_date: Mapped[str] = mapped_column(String(20), nullable=False)
    end_date: Mapped[str] = mapped_column(String(20), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Financial parameters (with server defaults for schema tolerance)
    initial_investment: Mapped[Optional[float]] = mapped_column(
        Float, default=100000.0, server_default="100000.0", nullable=True
    )
    commission_pct: Mapped[Optional[float]] = mapped_column(
        Float, default=0.0, server_default="0.0", nullable=True
    )
    slippage_pct: Mapped[Optional[float]] = mapped_column(
        Float, default=0.0, server_default="0.0", nullable=True
    )

    # Core performance metrics
    cagr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calmar: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Recommendation and regime context
    recommended_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    recommendation_source: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    current_regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Structured JSON payloads
    metrics_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    equity_curve_summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_backtest_logs_ticker_created", "ticker", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize model instance to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "strategy": self.strategy,
            "initial_investment": self.initial_investment,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "recommended_strategy": self.recommended_strategy,
            "recommendation_source": self.recommendation_source,
            "current_regime": self.current_regime,
            "metrics_json": self.metrics_json,
            "equity_curve_summary": self.equity_curve_summary,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        return f"<BacktestLog id={self.id} ticker='{self.ticker}' strategy='{self.strategy}' sharpe={self.sharpe}>"


class RegimeSnapshot(Base):
    """Historical snapshot of HMM regime detector outputs."""
    __tablename__ = "regime_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    ticker: Mapped[str] = mapped_column(String(32), default="^NSEI", nullable=False, index=True)
    as_of_date: Mapped[str] = mapped_column(String(20), nullable=False)
    current_regime: Mapped[str] = mapped_column(String(32), nullable=False)
    regime_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    total_trading_days: Mapped[int] = mapped_column(Integer, nullable=False)

    # HMM specifics
    transition_matrix: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    recent_segments: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    stationary_distribution: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    state_posteriors: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_regime_history_ticker_as_of", "ticker", "as_of_date"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize model instance to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "current_regime": self.current_regime,
            "regime_distribution": self.regime_distribution,
            "total_trading_days": self.total_trading_days,
            "transition_matrix": self.transition_matrix,
            "recent_segments": self.recent_segments,
            "stationary_distribution": self.stationary_distribution,
            "state_posteriors": self.state_posteriors,
        }

    def __repr__(self) -> str:
        return f"<RegimeSnapshot id={self.id} ticker='{self.ticker}' regime='{self.current_regime}' as_of='{self.as_of_date}'>"


class ModelBenchmarkRun(Base):
    """Quantitative benchmark run comparing strategy classification models."""
    __tablename__ = "model_benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    ticker: Mapped[str] = mapped_column(String(32), default="^NSEI", nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Time splits
    train_window_start: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    train_window_end: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    test_window_start: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    test_window_end: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Evaluation metrics
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overfitting_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Latency & training benchmarks
    training_time_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inference_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Architecture and diagnostics
    architecture_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_model_benchmarks_ticker_model", "ticker", "model_name", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize model instance to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ticker": self.ticker,
            "model_name": self.model_name,
            "train_window_start": self.train_window_start,
            "train_window_end": self.train_window_end,
            "test_window_start": self.test_window_start,
            "test_window_end": self.test_window_end,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "train_accuracy": self.train_accuracy,
            "overfitting_gap": self.overfitting_gap,
            "training_time_sec": self.training_time_sec,
            "inference_latency_ms": self.inference_latency_ms,
            "architecture_summary": self.architecture_summary,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"<ModelBenchmarkRun id={self.id} model='{self.model_name}' f1={self.f1_score:.4f} acc={self.accuracy:.4f}>"


# ---------------------------------------------------------------------------
# Compatibility schemas for cross-test suite interoperability
# ---------------------------------------------------------------------------

class RegimeSnapshotLegacy(Base):
    """Compatibility model for regime_snapshots table."""
    __tablename__ = "regime_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(50), nullable=False)
    as_of_date: Mapped[str] = mapped_column(String(20), nullable=False)
    current_regime: Mapped[str] = mapped_column(String(50), nullable=False)
    regime_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    transition_matrix: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    total_trading_days: Mapped[int] = mapped_column(Integer, nullable=False)


class ModelBenchmarkRunLegacy(Base):
    """Compatibility model for model_benchmark_runs table."""
    __tablename__ = "model_benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overfitting_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
