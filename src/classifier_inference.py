"""Classifier Inference.

Loads trained XGBoost / legacy strategy classifier models and scalers.
Predicts the calibrated suitability probability distribution across all 6 strategies:
- Buy & Hold
- MA Crossover
- RSI
- Momentum
- Bollinger Bands
- Dual Momentum
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.recommender import (
    STRATEGY_NAMES,
    XGBoostStrategyRecommender,
)


_RECOMMENDER_CACHE: XGBoostStrategyRecommender | None = None


def _get_recommender(models_dir: Path | str = "models") -> XGBoostStrategyRecommender | None:
    global _RECOMMENDER_CACHE
    if _RECOMMENDER_CACHE is not None and _RECOMMENDER_CACHE.is_fitted_:
        return _RECOMMENDER_CACHE

    dir_path = Path(models_dir)
    xgb_path = dir_path / "xgboost_recommender.joblib"
    if xgb_path.exists():
        try:
            rec = XGBoostStrategyRecommender.load(xgb_path)
            _RECOMMENDER_CACHE = rec
            return rec
        except Exception:
            pass

    return None


def is_recommendation_model_valid(
    models_dir: Path | str = "models", minimum_macro_f1: float = 0.30
) -> bool:
    """Return whether persisted model passed minimum validation quality gate."""
    recommender = _get_recommender(models_dir)
    if recommender is None or not recommender.is_fitted_:
        return False
    return float(recommender.cv_metrics_.get("macro_f1", 0.0)) >= minimum_macro_f1


def get_strategy_probabilities(
    df_recent: pd.DataFrame,
    current_regime: str | dict[str, Any] | pd.Series | None = None,
    models_dir: Path | str = "models",
) -> dict[str, float]:
    """Calculate calibrated suitability probability distribution across all 6 strategies.

    Args:
        df_recent: DataFrame of the most recent ~252 trading days.
        current_regime: The current regime string (e.g. "Bull") or a series of labels.
        models_dir: Directory containing trained models and scalers.

    Returns:
        Dictionary mapping strategy names to probabilities summing to 1.0.
    """
    if df_recent.empty:
        return {strat: round(1.0 / len(STRATEGY_NAMES), 4) for strat in STRATEGY_NAMES}

    models_dir = Path(models_dir)

    # 1. Try modern XGBoost recommender
    recommender = _get_recommender(models_dir)
    if isinstance(current_regime, dict):
        regime_info = dict(current_regime)
        regime_str = str(regime_info.get("regime", "Sideways"))
    else:
        regime_str = current_regime if isinstance(current_regime, str) else "Sideways"
        regime_info = {
            "regime": regime_str,
            "regime_id": 2 if regime_str == "Bull" else (0 if regime_str == "Bear" else 1),
            "prob_bull": 0.8 if regime_str == "Bull" else (0.1 if regime_str == "Bear" else 0.2),
            "prob_bear": 0.8 if regime_str == "Bear" else (0.1 if regime_str == "Bull" else 0.2),
            "prob_sideways": 0.6 if regime_str == "Sideways" else 0.1,
        }

    if recommender is not None and recommender.is_fitted_:
        try:
            res = recommender.recommend(df_recent, regime_info=regime_info)
            if len(res.probabilities) == len(STRATEGY_NAMES):
                return res.probabilities
        except Exception:
            pass

    # 2. Check for legacy individual .pkl models if XGBoost artifact is not yet present
    raw_probs: dict[str, float] = {}
    strategies = ["MA Crossover", "RSI", "Momentum", "Bollinger Bands", "Dual Momentum"]

    # Feature extraction for fallback
    from classifier_features import extract_features

    regime_labels = None
    if current_regime is not None:
        if isinstance(current_regime, str):
            regime_labels = pd.Series([current_regime] * len(df_recent), index=df_recent.index, name="regime")
        elif isinstance(current_regime, dict):
            regime_labels = pd.Series(
                [str(current_regime.get("regime", "Sideways"))] * len(df_recent),
                index=df_recent.index,
                name="regime",
            )
        else:
            regime_labels = current_regime

    features_dict = extract_features(df_recent, regime_labels) if len(df_recent) >= 20 else {}

    if features_dict:
        feature_cols = ["avg_return", "volatility", "momentum", "max_drawdown", "rsi_at_end", "sma_ratio"]
        if regime_labels is not None:
            feature_cols.extend(["regime_label", "regime_stability"])
        X = pd.DataFrame([features_dict])[feature_cols]

        for strategy in strategies:
            strat_key = strategy.replace(" ", "_")
            scaler_path = models_dir / f"{strat_key}_scaler.pkl"
            model_path = models_dir / f"{strat_key}_classifier.pkl"

            if scaler_path.exists() and model_path.exists():
                try:
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    X_scaled = scaler.transform(X)
                    prob = float(model.predict_proba(X_scaled)[0, 1])
                    raw_probs[strategy] = prob
                except Exception:
                    pass

    # If some legacy models exist, complete remaining strategies with quant defaults
    if raw_probs:
        raw_probs.setdefault("Buy & Hold", 0.5)
        raw_probs.setdefault("Dual Momentum", 0.4)
        raw_probs.setdefault("Bollinger Bands", 0.4)
        raw_probs.setdefault("MA Crossover", 0.4)
        raw_probs.setdefault("Momentum", 0.4)
        raw_probs.setdefault("RSI", 0.4)
        total = sum(raw_probs.values())
        return {k: round(raw_probs[k] / total, 4) for k in STRATEGY_NAMES}

    # 3. Deterministic quantitative fallback based on market features and regime
    close_col = [c for c in df_recent.columns if str(c).lower() == "close"]
    if close_col:
        c_series = df_recent[close_col[0]].astype(float)
        ret = float(c_series.pct_change().mean())
        vol = float(c_series.pct_change().std() * np.sqrt(252))
        mom = float((c_series.iloc[-1] / c_series.iloc[0]) - 1.0) if len(c_series) > 1 else 0.0
    else:
        ret, vol, mom = 0.0005, 0.15, 0.05

    scores = {
        "Buy & Hold": max(0.05, 0.15 + ret * 10 - vol * 2),
        "MA Crossover": max(0.05, 0.15 + mom * 5),
        "RSI": max(0.05, 0.15 + vol * 6 - ret * 4),
        "Momentum": max(0.05, 0.15 + mom * 8 + (0.1 if regime_str == "Bull" else 0.0)),
        "Bollinger Bands": max(0.05, 0.15 + vol * 4 + (0.1 if regime_str == "Sideways" else 0.0)),
        "Dual Momentum": max(0.05, 0.15 + (0.15 if regime_str == "Bear" else 0.0) + vol * 3),
    }
    total = sum(scores.values())
    return {k: round(scores[k] / total, 4) for k in STRATEGY_NAMES}
