"""Classifier Feature Engineering.

Implements Phase 2 of the Strategy Suitability Classifier pipeline.
Extracts a fixed-size feature vector from the 252-day lookback window.
CRITICAL: All features are computed strictly from the lookback window.
Zero lookahead bias is allowed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown over the period."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def extract_features(lookback_data: pd.DataFrame, regime_labels: pd.Series | None = None) -> dict[str, float]:
    """Extract features from exactly one 252-day lookback window.
    
    Args:
        lookback_data: DataFrame of length `lookback` ending at the decision point.
        regime_labels: Optional Series of regime labels aligning with lookback_data.
    """
    if lookback_data.empty:
        return {}
        
    prices = lookback_data["Close"]
    daily_returns = prices.pct_change().dropna()
    
    if len(prices) < 200: # Need 200 days for SMA ratio
        return {}
        
    # 1. average return
    avg_return = float(daily_returns.mean())
    
    # 2. volatility
    volatility = float(daily_returns.std() * np.sqrt(252)) # Annualized
    
    # 3. momentum (252-day return)
    momentum = float(prices.iloc[-1] / prices.iloc[0] - 1)
    
    # 4. max drawdown
    max_dd = compute_max_drawdown(prices)
    
    # 5. RSI at end of window
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    period = 14
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi_at_end = float(rsi_series.iloc[-1])
    
    # 6. SMA Ratio (50 / 200) at end of window
    sma_50 = prices.rolling(window=50, min_periods=50).mean().iloc[-1]
    sma_200 = prices.rolling(window=200, min_periods=200).mean().iloc[-1]
    sma_ratio = float(sma_50 / sma_200)
    
    features = {
        "avg_return": avg_return,
        "volatility": volatility,
        "momentum": momentum,
        "max_drawdown": max_dd,
        "rsi_at_end": rsi_at_end,
        "sma_ratio": sma_ratio,
    }
    
    # Handle regime features if labels are provided
    if regime_labels is not None:
        # Align regimes to the lookback window
        window_regimes = regime_labels.loc[regime_labels.index.isin(lookback_data.index)]
        if not window_regimes.empty:
            # Most recent regime label
            current_regime = window_regimes.iloc[-1]
            
            # Encode regime (Bull=2, Sideways=1, Bear=0)
            regime_map = {"Bull": 2, "Sideways": 1, "Bear": 0}
            # Handle if regime is directly the string or inside a dict/df
            regime_str = str(current_regime["regime"]) if isinstance(current_regime, pd.Series) else str(current_regime)
            encoded_regime = regime_map.get(regime_str, 1) # Default sideways if unknown
            
            # Count regime switches
            # Get the series of regime strings
            if isinstance(window_regimes, pd.DataFrame):
                regime_series = window_regimes["regime"]
            else:
                regime_series = window_regimes
                
            switches = int((regime_series != regime_series.shift()).sum() - 1) # -1 because first row is a 'switch' from NaN
            switches = max(0, switches)
            
            features["regime_label"] = encoded_regime
            features["regime_stability"] = switches
            
    return features


def build_feature_matrix(df: pd.DataFrame, window_end_dates: list[pd.Timestamp], regime_labels: pd.DataFrame | None = None, lookback: int = 252) -> pd.DataFrame:
    """Build feature matrix for all windows.
    
    Args:
        df: The full price dataframe.
        window_end_dates: List of dates representing the "present day" for each sample.
        regime_labels: Optional dataframe of regime labels.
        lookback: Number of days to look back from window_end_date.
    """
    feature_rows = []
    
    for end_date in window_end_dates:
        # Locate the index of the end date
        try:
            end_idx = df.index.get_loc(end_date)
            # Ensure we have enough lookback
            if end_idx < lookback - 1:
                print(f"Skipping {end_date} - not enough lookback history.")
                continue
                
            # The lookback window goes up to and INCLUDES the end_date
            # By slicing df.iloc[end_idx - lookback + 1 : end_idx + 1], we get exactly `lookback` rows
            lookback_data = df.iloc[end_idx - lookback + 1 : end_idx + 1]
            
            # Verify the last date in lookback_data is the end_date
            assert lookback_data.index[-1] == end_date, "Lookback data alignment error!"
            
            features = extract_features(lookback_data, regime_labels)
            features["window_end_date"] = end_date
            feature_rows.append(features)
            
        except KeyError:
            print(f"Warning: {end_date} not found in price data index.")
            continue
            
    return pd.DataFrame(feature_rows)
