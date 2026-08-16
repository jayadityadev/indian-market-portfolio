"""Recommend API route — GET /recommend."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

# routes/ -> api/ -> src/
SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.schemas import RecommendResponse, StrategyProbability, RiskForecast
from data_pipeline import load_data
from backtester import run_backtest
from regime_detector import get_regime_performance
from classifier_inference import get_strategy_probabilities
from model_registry import recommendation_state
from risk_forecaster import simulate_drawdowns
from strategies import (
    buy_and_hold, ma_crossover, rsi_strategy, momentum_strategy,
    bollinger_bands, dual_momentum,
)

import numpy as np
import pandas as pd

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


@router.get("/recommend", response_model=RecommendResponse)
def get_recommendation(ticker: str = Query(default="^NSEI")):
    """Get strategy recommendation with ML probabilities and risk forecast."""
    if ticker != SUPPORTED_TICKER:
        raise HTTPException(
            status_code=400,
            detail=f"Current canonical dataset supports {SUPPORTED_TICKER} only.",
        )
    base = PROJECT_ROOT
    models_dir = base / "models"
    model_state = recommendation_state(models_dir)

    # Load data
    try:
        df = load_data(base / "data" / "nifty50.parquet")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load failed: {exc}")

    regime_df_path = base / "data" / "nifty50_regimes.parquet"
    if not regime_df_path.exists():
        raise HTTPException(status_code=503, detail="Causal regime artifact is unavailable.")
    regime_df = pd.read_parquet(regime_df_path)
    regime_df.index = pd.to_datetime(regime_df.index)
    latest_regime = regime_df.dropna(subset=["regime"]).iloc[-1].to_dict()
    current_regime = str(latest_regime["regime"])

    # ML probabilities
    probs: dict[str, float] = {}
    if len(df) >= 252:
        df_recent = df.iloc[-252:]
        probs = get_strategy_probabilities(df_recent, latest_regime, models_dir=models_dir)
        if model_state["status"] != "validated_ml":
            probs = {}

    prob_list = [StrategyProbability(strategy=k, probability=v) for k, v in probs.items()]

    # Determine recommendation
    max_prob = max(probs.values()) if probs else 0.0
    source = "ml_classifier"

    if max_prob >= 0.55:
        recommended = max(probs, key=probs.get)
    else:
        # Fallback to historical Sharpe
        source = "historical_sharpe"
        all_results = {}
        if regime_df_path.exists():
            for name, func in STRATEGY_MAP.items():
                if name == "Buy & Hold":
                    continue
                try:
                    signals = func(df)
                    all_results[name] = run_backtest(df["Close"], signals)
                except Exception:
                    continue
            perf = get_regime_performance(regime_df, all_results)
            regime_perf = perf[perf["regime"] == current_regime]
            if not regime_perf.empty:
                best_row = regime_perf.sort_values("Sharpe", ascending=False).iloc[0]
                recommended = str(best_row["strategy"])
            else:
                recommended = "Momentum"  # safe default
        else:
            recommended = "Momentum"

    # Risk forecast and exposure limits
    risk = None
    exposure_limit = "100% (Normal Risk)"
    
    try:
        if recommended in STRATEGY_MAP:
            signals = STRATEGY_MAP[recommended](df)
            bt = run_backtest(df["Close"], signals)
            eq = bt["equity_curve"]
            rets = eq.pct_change().dropna()

            regime_df_path = base / "data" / "nifty50_regimes.parquet"
            if regime_df_path.exists():
                rdf = pd.read_parquet(regime_df_path)
                rdf.index = pd.to_datetime(rdf.index)
                regime_dates = rdf.index[rdf["regime"].astype(str) == current_regime]
                valid = rets.index.intersection(regime_dates)
                regime_rets = rets.loc[valid]
                if len(regime_rets) >= 30:
                    fc = simulate_drawdowns(regime_rets, n_simulations=1000, horizon=63)
                    risk = RiskForecast(**fc)
                    
                    # Risk Management Skill: Set exposure limits based on median forecasted drawdown
                    median_dd = abs(fc["median_50"])
                    if median_dd >= 0.20:
                        exposure_limit = "25% (Critical Risk - Severe Drawdown Expected)"
                    elif median_dd >= 0.15:
                        exposure_limit = "50% (High Risk - Elevated Drawdown Expected)"
                    elif median_dd >= 0.10:
                        exposure_limit = "75% (Moderate Risk)"
                        
    except Exception:
        pass  # Risk forecast is optional

    return RecommendResponse(
        current_regime=current_regime,
        recommended_strategy=recommended,
        recommendation_source=source,
        recommendation_status=model_state["status"],
        recommendation_reason=model_state["reason"],
        recommended_exposure=exposure_limit,
        probabilities=prob_list,
        risk_forecast=risk,
    )
