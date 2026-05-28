"""Full Analysis API route — POST /analyze.

Returns the complete pipeline output:
- Backtest metrics (overall + per-regime breakdown) with Sortino + Calmar
- OHLC price data for candlestick charts
- Regime timeline segments
- ML recommendation with source explanation
- Risk forecast per regime
- Regime-conditional performance heatmap
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

import numpy as np
import pandas as pd

# routes/ -> api/ -> src/
SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.schemas import AnalyzeRequest, AnalyzeResponse, MetricsResponse
from backtester import run_backtest
from data_pipeline import load_data, fetch_data, engineer_features
from regime_detector import get_current_regime, get_regime_performance, fit_regimes
from classifier_inference import get_strategy_probabilities
from risk_forecaster import simulate_drawdowns
from market_outlook import get_market_outlook
from strategies import (
    buy_and_hold, ma_crossover, rsi_strategy, momentum_strategy,
    bollinger_bands, dual_momentum,
)

router = APIRouter()

STRATEGY_MAP = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Bollinger Bands": bollinger_bands,
    "Dual Momentum": dual_momentum,
}


@router.post("/analyze", response_model=AnalyzeResponse)
def run_full_analysis(req: AnalyzeRequest):
    """Run the full pipeline: data → features → regimes → backtest → ML → risk."""

    # 1. Load / fetch data
    try:
        data_path = PROJECT_ROOT / "data" / "nifty50.parquet"
        use_cache = False
        if req.ticker == "^NSEI" and data_path.exists():
            df_cache = load_data(data_path)
            if not df_cache.empty and df_cache.index.max().strftime("%Y-%m-%d") >= req.end_date:
                df = df_cache.loc[req.start_date:req.end_date]
                use_cache = True
        
        if not use_cache:
            raw = fetch_data(req.ticker, req.start_date, req.end_date)
            df = engineer_features(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load failed: {exc}")

    if df.empty or len(df) < 60:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis. Need at least 60 trading days.")

    # 2. Extract OHLC for candlestick chart (full resolution + volume)
    ohlc_data = []
    try:
        ohlc_cols = ["Open", "High", "Low", "Close"]
        if all(c in df.columns for c in ohlc_cols):
            ohlc_df = df[ohlc_cols].dropna()
            has_volume = "Volume" in df.columns
            for d, row in ohlc_df.iterrows():
                pt = {
                    "date": d.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                }
                if has_volume and d in df.index:
                    vol = df.loc[d, "Volume"]
                    if pd.notna(vol):
                        pt["volume"] = int(vol)
                ohlc_data.append(pt)
    except Exception:
        pass

    # 3. Regime detection
    try:
        regime_df = fit_regimes(df)
        current_regime = get_current_regime(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regime detection failed: {exc}")

    # 4. Run strategy backtests (single strategy + Buy & Hold if not "all")
    strategies_to_run = STRATEGY_MAP
    if req.strategy and req.strategy != "all" and req.strategy in STRATEGY_MAP:
        strategies_to_run = {"Buy & Hold": buy_and_hold, req.strategy: STRATEGY_MAP[req.strategy]}

    all_results = {}
    overall_metrics = {}
    equity_curves = {}
    for name, func in strategies_to_run.items():
        try:
            signals = func(df)
            bt = run_backtest(
                df["Close"],
                signals,
                commission_pct=req.commission_pct,
                slippage_pct=req.slippage_pct,
            )
            all_results[name] = bt
            overall_metrics[name] = bt["metrics"]
            eq = bt["equity_curve"]
            eq_scaled = eq * req.initial_investment
            equity_curves[name] = [
                {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 0)}
                for d, v in eq_scaled.items()
            ]
        except Exception:
            continue

    # 5. Regime-conditional performance heatmap (THE X-FACTOR)
    regime_heatmap = []
    perf = pd.DataFrame()
    try:
        perf = get_regime_performance(regime_df, all_results)
        if not perf.empty:
            for _, row in perf.iterrows():
                regime_heatmap.append({
                    "strategy": str(row["strategy"]),
                    "regime": str(row["regime"]),
                    "CAGR": round(float(row.get("CAGR", 0)), 4),
                    "Sharpe": round(float(row.get("Sharpe", 0)), 2),
                    "Sortino": round(float(row.get("Sortino", 0)), 2),
                    "MaxDrawdown": round(float(row.get("MaxDrawdown", 0)), 4),
                    "Calmar": round(float(row.get("Calmar", 0)), 2),
                    "Volatility": round(float(row.get("Volatility", 0)), 4),
                })
    except Exception:
        pass

    # 6. Regime timeline segments
    regime_timeline = []
    try:
        r_col = regime_df["regime"].dropna().astype(str)
        if not r_col.empty:
            segments = []
            prev = r_col.iloc[0]
            start_idx = r_col.index[0]
            for i in range(1, len(r_col)):
                curr = r_col.iloc[i]
                if curr != prev:
                    segments.append({
                        "regime": prev,
                        "start": start_idx.strftime("%Y-%m-%d"),
                        "end": r_col.index[i-1].strftime("%Y-%m-%d"),
                        "days": (r_col.index[i-1] - start_idx).days,
                    })
                    start_idx = r_col.index[i]
                    prev = curr
            segments.append({
                "regime": prev,
                "start": start_idx.strftime("%Y-%m-%d"),
                "end": r_col.index[-1].strftime("%Y-%m-%d"),
                "days": (r_col.index[-1] - start_idx).days,
            })
            regime_timeline = segments
    except Exception:
        pass

    # 7. ML recommendation
    models_dir = PROJECT_ROOT / "models"
    probs = {}
    if len(df) >= 252:
        df_recent = df.iloc[-252:]
        probs = get_strategy_probabilities(df_recent, current_regime, models_dir=models_dir)

    max_prob = max(probs.values()) if probs else 0.0
    rec_source = "ml_classifier"
    if max_prob >= 0.55:
        recommended = max(probs, key=probs.get)
    else:
        rec_source = "historical_sharpe"
        if not perf.empty:
            regime_perf = perf[perf["regime"] == current_regime]
            if not regime_perf.empty:
                best_row = regime_perf.sort_values("Sharpe", ascending=False).iloc[0]
                recommended = str(best_row["strategy"])
            else:
                recommended = "Momentum"
        else:
            recommended = "Momentum"

    # 8. Risk forecast
    risk_data = None
    exposure_limit = "100% (Normal Risk)"
    try:
        if recommended in STRATEGY_MAP and recommended in all_results:
            eq = all_results[recommended]["equity_curve"]
            rets = eq.pct_change().dropna()
            r_col = regime_df["regime"].dropna().astype(str)
            regime_dates = r_col.index[r_col == current_regime]
            valid = rets.index.intersection(regime_dates)
            regime_rets = rets.loc[valid]
            if len(regime_rets) >= 30:
                fc = simulate_drawdowns(regime_rets, n_simulations=1000, horizon=63)
                risk_data = fc
                median_dd = abs(fc["median_50"])
                if median_dd >= 0.20:
                    exposure_limit = "25% (Critical Risk)"
                elif median_dd >= 0.15:
                    exposure_limit = "50% (High Risk)"
                elif median_dd >= 0.10:
                    exposure_limit = "75% (Moderate Risk)"
    except Exception:
        pass

    # 9. Market Outlook
    outlook_data = None
    try:
        # We use the full df and strategy probs for the outlook
        outlook_dict = get_market_outlook(df, current_regime, probs)
        from api.schemas import MarketOutlook
        outlook_data = MarketOutlook(**outlook_dict)
    except Exception:
        pass

    return AnalyzeResponse(
        ticker=req.ticker,
        start_date=req.start_date,
        end_date=req.end_date,
        n_trading_days=len(df),
        initial_investment=req.initial_investment,
        current_regime=current_regime,
        recommended_strategy=recommended,
        recommendation_source=rec_source,
        recommendation_reason=_build_reason(recommended, rec_source, current_regime, probs, perf if not perf.empty else None),
        recommended_exposure=exposure_limit,
        probabilities={k: round(v, 4) for k, v in probs.items()},
        overall_metrics={k: MetricsResponse(**v) for k, v in overall_metrics.items()},
        equity_curves=equity_curves,
        ohlc_data=ohlc_data,
        regime_heatmap=regime_heatmap,
        regime_timeline=regime_timeline,
        risk_forecast=risk_data,
        market_outlook=outlook_data,
    )


def _build_reason(strategy: str, source: str, regime: str, probs: dict, perf_df) -> str:
    """Build human-readable recommendation reason."""
    if source == "ml_classifier":
        prob = probs.get(strategy, 0)
        return f"ML classifier predicts {strategy} has {prob*100:.0f}% probability of outperforming in {regime} regime."
    else:
        if perf_df is not None and not perf_df.empty:
            regime_rows = perf_df[perf_df["regime"] == regime]
            if not regime_rows.empty:
                best = regime_rows.sort_values("Sharpe", ascending=False).iloc[0]
                sharpe = best.get("Sharpe", 0)
                return f"Best historical Sharpe ({sharpe:.2f}) in {regime} regime. ML confidence was too low — using proven historical performance."
        return f"Recommended for {regime} regime based on historical risk-adjusted returns."
