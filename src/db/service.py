"""Background execution recording services and database initialization."""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import Engine

from .connection import ensure_sqlite_directory_exists, get_engine
from .crud import (
    create_backtest_log,
    create_benchmark_batch,
    create_regime_snapshot,
)
from .models import Base
from .session import get_db_session

logger = logging.getLogger(__name__)


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all missing database tables defined in ORM Base metadata."""
    target_engine = engine or get_engine()
    url_str = str(target_engine.url)

    if url_str.startswith("sqlite"):
        ensure_sqlite_directory_exists(url_str)

    logger.info("Auto-initializing database schemas via Base.metadata.create_all")
    Base.metadata.create_all(bind=target_engine)
    logger.info("Database schemas initialized successfully.")


def _to_dict(obj: Any) -> dict[str, Any]:
    """Safely convert Pydantic model or dict to python dict."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    return {}


def save_analysis_record(
    payload: Any,
    result: Any,
    notes: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> Optional[dict[str, int]]:
    """Background task: Record full analysis run (BacktestLog + RegimeSnapshot)."""
    try:
        req = _to_dict(payload)
        res = _to_dict(result)

        ticker = req.get("ticker", res.get("ticker", "^NSEI"))
        start_date = str(req.get("start_date", res.get("start_date", "")))
        end_date = str(req.get("end_date", res.get("end_date", "")))
        strategy = req.get("strategy", "all")
        initial_inv = float(req.get("initial_investment", res.get("initial_investment", 100000.0)))
        comm = float(req.get("commission_pct", 0.0))
        slip = float(req.get("slippage_pct", 0.0))

        current_regime = res.get("current_regime")
        recommended_strategy = res.get("recommended_strategy")
        rec_source = res.get("recommendation_source")

        overall_metrics = res.get("overall_metrics", {})

        # Determine primary metrics for the log
        cagr = sharpe = sortino = max_dd = calmar = vol = None
        if recommended_strategy and isinstance(overall_metrics, dict) and recommended_strategy in overall_metrics:
            m = overall_metrics[recommended_strategy]
            if isinstance(m, dict):
                cagr = m.get("CAGR")
                sharpe = m.get("Sharpe")
                sortino = m.get("Sortino")
                max_dd = m.get("MaxDrawdown")
                calmar = m.get("Calmar")
                vol = m.get("Volatility")
        elif isinstance(overall_metrics, dict) and overall_metrics:
            first_key = next(iter(overall_metrics))
            m = overall_metrics[first_key]
            if isinstance(m, dict):
                cagr = m.get("CAGR")
                sharpe = m.get("Sharpe")
                sortino = m.get("Sortino")
                max_dd = m.get("MaxDrawdown")
                calmar = m.get("Calmar")
                vol = m.get("Volatility")

        equity_curves = res.get("equity_curves", {})
        eq_summary: dict[str, Any] = {}
        if isinstance(equity_curves, dict):
            for s_name, points in equity_curves.items():
                if isinstance(points, (list, tuple)) and len(points) > 0:
                    eq_summary[s_name] = {
                        "start": points[0],
                        "end": points[-1],
                        "total_points": len(points),
                    }

        regime_timeline = res.get("regime_timeline", [])
        if not isinstance(regime_timeline, list):
            regime_timeline = []

        # Calculate regime distribution counts from timeline
        regime_distribution: dict[str, int] = {}
        for seg in regime_timeline:
            if isinstance(seg, dict):
                r = seg.get("regime", "Unknown")
                days = int(seg.get("days", 0))
                regime_distribution[r] = regime_distribution.get(r, 0) + days

        if not regime_distribution and current_regime:
            regime_distribution = {current_regime: int(res.get("n_trading_days", 0))}

        with get_db_session(engine) as session:
            # 1. Backtest Log
            bt_log = create_backtest_log(
                session=session,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_investment=initial_inv,
                commission_pct=comm,
                slippage_pct=slip,
                cagr=float(cagr) if cagr is not None else None,
                sharpe=float(sharpe) if sharpe is not None else None,
                sortino=float(sortino) if sortino is not None else None,
                max_drawdown=float(max_dd) if max_dd is not None else None,
                calmar=float(calmar) if calmar is not None else None,
                volatility=float(vol) if vol is not None else None,
                recommended_strategy=recommended_strategy,
                recommendation_source=rec_source,
                current_regime=current_regime,
                metrics_json=overall_metrics if isinstance(overall_metrics, dict) else None,
                equity_curve_summary=eq_summary if eq_summary else None,
                notes=notes,
            )

            # 2. Regime Snapshot
            regime_snap = create_regime_snapshot(
                session=session,
                ticker=ticker,
                as_of_date=end_date or "N/A",
                current_regime=current_regime or "Unknown",
                regime_distribution=regime_distribution,
                total_trading_days=int(res.get("n_trading_days", 0)),
                recent_segments=regime_timeline if regime_timeline else None,
            )

            logger.info(
                "Persisted analysis record: BacktestLog id=%d, RegimeSnapshot id=%d",
                bt_log.id,
                regime_snap.id,
            )
            return {"backtest_log_id": bt_log.id, "regime_snapshot_id": regime_snap.id}

    except Exception as exc:
        logger.exception("Failed to persist analysis record in background task: %s", exc)
        return None


def save_backtest_record(
    payload: Any,
    result: Any,
    engine: Optional[Engine] = None,
) -> list[int]:
    """Background task: Record standalone backtest endpoint execution."""
    try:
        req = _to_dict(payload)
        res = _to_dict(result)

        ticker = req.get("ticker", res.get("ticker", "^NSEI"))
        start_date = str(req.get("start_date", res.get("start_date", "")))
        end_date = str(req.get("end_date", res.get("end_date", "")))
        initial_inv = float(req.get("initial_investment", 100000.0))
        comm = float(req.get("commission_pct", 0.0))
        slip = float(req.get("slippage_pct", 0.0))

        strategy_results = res.get("results", [])
        if not isinstance(strategy_results, list):
            strategy_results = []

        saved_ids: list[int] = []

        with get_db_session(engine) as session:
            for item in strategy_results:
                if not isinstance(item, dict):
                    continue
                strat_name = item.get("strategy", "Unknown")
                m = item.get("metrics", {})
                if not isinstance(m, dict):
                    m = {}

                eq_summary = {
                    "start_val": item.get("equity_curve_start"),
                    "end_val": item.get("equity_curve_end"),
                    "n_days": item.get("n_days"),
                }

                bt = create_backtest_log(
                    session=session,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strat_name,
                    initial_investment=initial_inv,
                    commission_pct=comm,
                    slippage_pct=slip,
                    cagr=float(m.get("CAGR")) if m.get("CAGR") is not None else None,
                    sharpe=float(m.get("Sharpe")) if m.get("Sharpe") is not None else None,
                    sortino=float(m.get("Sortino")) if m.get("Sortino") is not None else None,
                    max_drawdown=float(m.get("MaxDrawdown")) if m.get("MaxDrawdown") is not None else None,
                    calmar=float(m.get("Calmar")) if m.get("Calmar") is not None else None,
                    volatility=float(m.get("Volatility")) if m.get("Volatility") is not None else None,
                    metrics_json={strat_name: m},
                    equity_curve_summary=eq_summary,
                )
                saved_ids.append(bt.id)

            logger.info("Persisted %d backtest logs for ticker %s", len(saved_ids), ticker)
            return saved_ids

    except Exception as exc:
        logger.exception("Failed to persist backtest record: %s", exc)
        return []


def save_benchmark_record(
    benchmark_data: Any,
    engine: Optional[Engine] = None,
) -> list[int]:
    """Background task: Record Model Benchmark run cards."""
    try:
        data = _to_dict(benchmark_data)
        ticker = data.get("ticker", "^NSEI")
        eval_date = str(data.get("evaluation_date", ""))
        models_list = data.get("models", [])
        if not isinstance(models_list, list):
            models_list = []

        records_to_create: list[dict[str, Any]] = []
        for card in models_list:
            if not isinstance(card, dict):
                continue
            records_to_create.append({
                "ticker": ticker,
                "model_name": card.get("model_name", "Unknown"),
                "test_window_end": eval_date,
                "accuracy": float(card.get("accuracy", 0.0)),
                "precision": float(card["precision"]) if card.get("precision") is not None else None,
                "recall": float(card["recall"]) if card.get("recall") is not None else None,
                "f1_score": float(card.get("f1_score", 0.0)),
                "roc_auc": float(card["roc_auc"]) if card.get("roc_auc") is not None else None,
                "train_accuracy": float(card["train_accuracy"]) if card.get("train_accuracy") is not None else None,
                "overfitting_gap": float(card["overfitting_gap"]) if card.get("overfitting_gap") is not None else None,
                "training_time_sec": float(card["training_time_sec"]) if card.get("training_time_sec") is not None else None,
                "inference_latency_ms": float(card["inference_latency_ms"]) if card.get("inference_latency_ms") is not None else None,
                "architecture_summary": card.get("architecture_summary"),
                "details": card.get("details", {}),
            })

        if not records_to_create:
            return []

        with get_db_session(engine) as session:
            created = create_benchmark_batch(session, records_to_create)
            ids = [r.id for r in created]
            logger.info("Persisted %d benchmark run records for ticker %s", len(ids), ticker)
            return ids

    except Exception as exc:
        logger.exception("Failed to persist benchmark records: %s", exc)
        return []
