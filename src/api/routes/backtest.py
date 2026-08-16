"""Backtest API route — POST /backtest."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

# Ensure src/ importable
# routes/ -> api/ -> src/
SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.schemas import BacktestRequest, BacktestResponse, MetricsResponse, StrategyResult
from backtester import run_backtest
from data_pipeline import load_data, fetch_data, engineer_features
from strategies import (
    buy_and_hold, ma_crossover, rsi_strategy, momentum_strategy,
    bollinger_bands, dual_momentum,
)
from db.service import save_backtest_record

router = APIRouter()
SUPPORTED_TICKER = "^NSEI"

STRATEGY_MAP = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Bollinger Bands": bollinger_bands,
    "Dual Momentum": dual_momentum,
}


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtest for specified ticker, date range, and strategy."""
    if req.ticker != SUPPORTED_TICKER:
        raise HTTPException(
            status_code=400,
            detail=f"Current canonical dataset supports {SUPPORTED_TICKER} only.",
        )
    # Load data
    try:
        data_path = PROJECT_ROOT / "data" / "nifty50.parquet"
        if req.ticker == SUPPORTED_TICKER and data_path.exists():
            df = load_data(data_path)
            df = df.loc[req.start_date:req.end_date]
        else:
            raw = fetch_data(req.ticker, req.start_date, req.end_date)
            df = engineer_features(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load failed: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No data for requested range.")

    # Determine strategies to run
    if req.strategy.lower() == "all":
        strategies = STRATEGY_MAP
    elif req.strategy in STRATEGY_MAP:
        strategies = {req.strategy: STRATEGY_MAP[req.strategy]}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {req.strategy}. Options: {list(STRATEGY_MAP.keys())}",
        )

    # Run backtests
    results = []
    for name, func in strategies.items():
        try:
            signals = func(df)
            bt = run_backtest(
                df["Close"],
                signals,
                commission_pct=req.commission_pct,
                slippage_pct=req.slippage_pct,
            )
            m = bt["metrics"]
            eq = bt["equity_curve"]
            results.append(StrategyResult(
                strategy=name,
                metrics=MetricsResponse(**m),
                equity_curve_start=float(eq.iloc[0]),
                equity_curve_end=float(eq.iloc[-1]),
                n_days=len(eq),
            ))
        except Exception as exc:
            # Skip failed strategies, don't crash entire request
            continue

    response = BacktestResponse(
        ticker=req.ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        n_trading_days=len(df),
        results=results,
    )
    background_tasks.add_task(save_backtest_record, req, response)
    return response
