"""Risk Forecaster — Probabilistic drawdown bands.

Implements Module B of the Post-MVP roadmap.
Uses historical simulation via bootstrapping to estimate the expected
maximum drawdown of a strategy over a future horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Peak-to-trough decline as a fraction."""
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    return float(np.min(drawdowns))


def simulate_drawdowns(
    returns: pd.Series, 
    n_simulations: int = 1000, 
    horizon: int = 63,
    random_seed: int = 42
) -> dict[str, float]:
    """Bootstrap historical returns to estimate future max drawdown.
    
    Args:
        returns: Historical daily returns of the strategy.
        n_simulations: Number of paths to simulate.
        horizon: Number of days to simulate forward.
        random_seed: Seed for reproducibility.
        
    Returns:
        Dictionary with 10th (best case), 50th (median), and 90th (worst case) 
        percentiles of max drawdowns.
    """
    if len(returns) == 0:
        return {"worst_case_10": 0.0, "median_50": 0.0, "best_case_90": 0.0}
        
    rng = np.random.default_rng(seed=random_seed)
    
    # Extract values as numpy array for faster sampling
    returns_arr = returns.values
    
    max_drawdowns = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        # Sample with replacement
        sampled_returns = rng.choice(returns_arr, size=horizon, replace=True)
        
        # Build equity curve (base 1.0)
        equity_curve = np.cumprod(1.0 + sampled_returns)
        
        # Calculate and store max drawdown
        max_drawdowns[i] = max_drawdown(equity_curve)
        
    # Calculate percentiles. 
    # Note: Drawdowns are negative numbers (e.g. -0.15 for 15% drop).
    # 90th percentile of NEGATIVE numbers gives the WORST case (closest to -1).
    # Wait, np.percentile(-0.15, -0.05). The 10th percentile is the smaller number (-0.15).
    # Let's be clear: 
    #   p10 = 10th percentile (e.g. -0.15, deep drawdown, "worst case")
    #   p50 = median
    #   p90 = 90th percentile (e.g. -0.02, shallow drawdown, "best case")
    
    p10 = float(np.percentile(max_drawdowns, 10))
    p50 = float(np.percentile(max_drawdowns, 50))
    p90 = float(np.percentile(max_drawdowns, 90))
    
    return {
        "worst_case_10": p10,
        "median_50": p50,
        "best_case_90": p90
    }
