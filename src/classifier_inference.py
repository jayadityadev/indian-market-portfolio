"""Classifier Inference.

Loads trained strategy classifier models and scalers.
Predicts the probability of each strategy beating the Buy & Hold benchmark
over the next 63 days given the current market regime and lookback data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from classifier_features import extract_features


def get_strategy_probabilities(
    df_recent: pd.DataFrame, current_regime: str | pd.Series | None, models_dir: Path | str = "models"
) -> dict[str, float]:
    """Calculate probability of each strategy outperforming Buy & Hold.
    
    Args:
        df_recent: DataFrame of the most recent ~252 trading days.
        current_regime: The current regime string (e.g. "Bull") or a series of labels.
        models_dir: Directory containing trained .pkl models and scalers.
        
    Returns:
        Dictionary mapping strategy names to probabilities (0.0 to 1.0).
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return {}
        
    # Standardize current_regime to be a series if it's just a string, 
    # to be compatible with extract_features.
    regime_labels = None
    if current_regime is not None:
        if isinstance(current_regime, str):
            regime_labels = pd.Series(
                [current_regime] * len(df_recent), 
                index=df_recent.index, 
                name="regime"
            )
        else:
            regime_labels = current_regime
            
    # Extract features for the recent window
    features_dict = extract_features(df_recent, regime_labels)
    if not features_dict:
        return {}
        
    # Ensure correct order
    feature_cols = ["avg_return", "volatility", "momentum", "max_drawdown", "rsi_at_end", "sma_ratio"]
    if regime_labels is not None:
        feature_cols.extend(["regime_label", "regime_stability"])
        
    # Build single-row DataFrame
    X = pd.DataFrame([features_dict])[feature_cols]
    
    probabilities = {}
    
    # We evaluate MA Crossover, RSI, Momentum. Buy & Hold gets a nominal value or we can leave it out.
    # The models were trained to predict "Will it beat Buy & Hold?", so Buy & Hold probability is
    # conceptually 0.5 (the baseline).
    strategies = ["MA Crossover", "RSI", "Momentum"]
    
    for strategy in strategies:
        strat_key = strategy.replace(" ", "_")
        scaler_path = models_dir / f"{strat_key}_scaler.pkl"
        model_path = models_dir / f"{strat_key}_classifier.pkl"
        
        if scaler_path.exists() and model_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                
            X_scaled_array = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
            
            # predict_proba returns [prob_class_0, prob_class_1]
            prob = float(model.predict_proba(X_scaled)[0, 1])
            probabilities[strategy] = prob
            
    return probabilities
