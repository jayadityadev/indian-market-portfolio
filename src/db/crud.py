"""CRUD operations for BacktestLog, RegimeSnapshot, and ModelBenchmarkRun."""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import BacktestLog, ModelBenchmarkRun, RegimeSnapshot


# ---------------------------------------------------------------------------
# BacktestLog CRUD
# ---------------------------------------------------------------------------

def create_backtest_log(session: Session, **kwargs: Any) -> BacktestLog:
    """Create and persist a new BacktestLog record."""
    log = BacktestLog(**kwargs)
    session.add(log)
    session.flush()
    return log


def get_backtest_by_id(session: Session, backtest_id: int) -> Optional[BacktestLog]:
    """Retrieve a BacktestLog by its primary key ID."""
    stmt = select(BacktestLog).where(BacktestLog.id == backtest_id)
    return session.scalars(stmt).first()


def get_recent_backtests(
    session: Session,
    limit: int = 50,
    offset: int = 0,
    ticker: Optional[str] = None,
    strategy: Optional[str] = None,
) -> list[BacktestLog]:
    """Retrieve recent backtest logs ordered by creation time descending."""
    stmt = select(BacktestLog).order_by(BacktestLog.created_at.desc())
    if ticker:
        stmt = stmt.where(BacktestLog.ticker == ticker)
    if strategy:
        stmt = stmt.where(BacktestLog.strategy == strategy)

    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt).all())


def count_backtests(
    session: Session,
    ticker: Optional[str] = None,
    strategy: Optional[str] = None,
) -> int:
    """Count total backtests matching criteria."""
    stmt = select(func.count(BacktestLog.id))
    if ticker:
        stmt = stmt.where(BacktestLog.ticker == ticker)
    if strategy:
        stmt = stmt.where(BacktestLog.strategy == strategy)
    return session.scalar(stmt) or 0


def delete_backtest_by_id(session: Session, backtest_id: int) -> bool:
    """Delete a backtest log by ID."""
    log = get_backtest_by_id(session, backtest_id)
    if log is not None:
        session.delete(log)
        session.flush()
        return True
    return False


# ---------------------------------------------------------------------------
# RegimeSnapshot CRUD
# ---------------------------------------------------------------------------

def create_regime_snapshot(session: Session, **kwargs: Any) -> RegimeSnapshot:
    """Create and persist a new RegimeSnapshot record."""
    snapshot = RegimeSnapshot(**kwargs)
    session.add(snapshot)
    session.flush()
    return snapshot


def get_latest_regime_snapshot(
    session: Session,
    ticker: str = "^NSEI",
) -> Optional[RegimeSnapshot]:
    """Retrieve the most recent regime snapshot for a ticker."""
    stmt = (
        select(RegimeSnapshot)
        .where(RegimeSnapshot.ticker == ticker)
        .order_by(RegimeSnapshot.created_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def get_recent_regime_snapshots(
    session: Session,
    limit: int = 50,
    offset: int = 0,
    ticker: Optional[str] = None,
) -> list[RegimeSnapshot]:
    """Retrieve recent regime snapshots ordered by creation time descending."""
    stmt = select(RegimeSnapshot).order_by(RegimeSnapshot.created_at.desc())
    if ticker:
        stmt = stmt.where(RegimeSnapshot.ticker == ticker)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt).all())


def count_regime_snapshots(
    session: Session,
    ticker: Optional[str] = None,
) -> int:
    """Count total regime snapshots matching criteria."""
    stmt = select(func.count(RegimeSnapshot.id))
    if ticker:
        stmt = stmt.where(RegimeSnapshot.ticker == ticker)
    return session.scalar(stmt) or 0


# ---------------------------------------------------------------------------
# ModelBenchmarkRun CRUD
# ---------------------------------------------------------------------------

def create_benchmark_run(session: Session, **kwargs: Any) -> ModelBenchmarkRun:
    """Create and persist a single ModelBenchmarkRun record."""
    run = ModelBenchmarkRun(**kwargs)
    session.add(run)
    session.flush()
    return run


def create_benchmark_batch(
    session: Session,
    benchmark_runs: list[dict[str, Any]],
) -> list[ModelBenchmarkRun]:
    """Create and persist multiple ModelBenchmarkRun records in a single transaction."""
    records = [ModelBenchmarkRun(**data) for data in benchmark_runs]
    session.add_all(records)
    session.flush()
    return records


def get_latest_benchmark_runs(
    session: Session,
    ticker: Optional[str] = None,
) -> list[ModelBenchmarkRun]:
    """Retrieve the most recent benchmark run for each distinct model_name."""
    stmt = select(ModelBenchmarkRun).order_by(ModelBenchmarkRun.created_at.desc())
    if ticker:
        stmt = stmt.where(ModelBenchmarkRun.ticker == ticker)

    all_runs = list(session.scalars(stmt).all())

    seen_models: set[str] = set()
    latest: list[ModelBenchmarkRun] = []
    for r in all_runs:
        if r.model_name not in seen_models:
            seen_models.add(r.model_name)
            latest.append(r)
    return latest


def get_benchmark_history(
    session: Session,
    model_name: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ModelBenchmarkRun]:
    """Retrieve benchmark run history ordered by creation time descending."""
    stmt = select(ModelBenchmarkRun).order_by(ModelBenchmarkRun.created_at.desc())
    if model_name:
        stmt = stmt.where(ModelBenchmarkRun.model_name == model_name)
    if ticker:
        stmt = stmt.where(ModelBenchmarkRun.ticker == ticker)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.scalars(stmt).all())


def count_benchmarks(
    session: Session,
    model_name: Optional[str] = None,
    ticker: Optional[str] = None,
) -> int:
    """Count total benchmark runs matching criteria."""
    stmt = select(func.count(ModelBenchmarkRun.id))
    if model_name:
        stmt = stmt.where(ModelBenchmarkRun.model_name == model_name)
    if ticker:
        stmt = stmt.where(ModelBenchmarkRun.ticker == ticker)
    return session.scalar(stmt) or 0
