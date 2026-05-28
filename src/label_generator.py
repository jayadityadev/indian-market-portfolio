"""Label Generator — Creates training data using rolling windows.

This script implements Phase 1 of the Strategy Suitability Classifier pipeline.
It extracts multiple training samples from the single historical time series by
sliding a window across the data. For each window, it computes the forward
performance (Sharpe ratio) of each strategy and assigns a binary label:
1 if the strategy outperforms Buy & Hold, 0 otherwise.

Methodology:
- Lookback window: 252 days (1 trading year) for feature extraction context
- Forward horizon: 63 days (~1 quarter) for evaluating strategy performance
- Step size: 21 days (~1 month) to advance the window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies import (
    buy_and_hold, ma_crossover, momentum_strategy, rsi_strategy,
    bollinger_bands, dual_momentum,
)
from backtester import run_backtest

STRATEGY_MAP = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Bollinger Bands": bollinger_bands,
    "Dual Momentum": dual_momentum,
}


def generate_labels(
    df: pd.DataFrame,
    lookback: int = 252,
    forward: int = 63,
    step: int = 21,
) -> pd.DataFrame:
    """Generate labeled dataset via rolling windows."""
    print(f"Generating labels with lookback={lookback}, forward={forward}, step={step}...")
    
    # Pre-compute signals globally to avoid lookback truncation on 63-day windows
    global_signals = {}
    for name, func in STRATEGY_MAP.items():
        global_signals[name] = func(df)
        
    labeled_rows = []
    total_days = len(df)
    
    # We iterate such that we have both a full lookback window and a full forward window
    for start_idx in range(0, total_days - lookback - forward, step):
        # We define window_end as the current "present day" for this sample
        window_end = start_idx + lookback
        
        # Forward data is what happens *after* the "present day"
        forward_data = df.iloc[window_end : window_end + forward]
        forward_close = forward_data["Close"]
        
        # Run backtests over the forward window to get future performance
        results = {}
        for name in STRATEGY_MAP.keys():
            strat_signals = global_signals[name].iloc[window_end : window_end + forward]
            results[name] = run_backtest(forward_close, strat_signals)
            
        # Extract Sharpe ratios
        bh_sharpe = results["Buy & Hold"]["metrics"]["Sharpe"]
        
        for strategy_name, result in results.items():
            if strategy_name == "Buy & Hold":
                continue
                
            strat_sharpe = result["metrics"]["Sharpe"]
            
            # Label = 1 if strategy beats Buy & Hold risk-adjusted
            label = 1 if strat_sharpe > bh_sharpe else 0
            
            labeled_rows.append({
                "window_end_date": df.index[window_end],  # The "present day" for this sample
                "strategy": strategy_name,
                "strategy_sharpe": strat_sharpe,
                "bh_sharpe": bh_sharpe,
                "label": label
            })

    labels_df = pd.DataFrame(labeled_rows)
    return labels_df


def validate_and_save(labels_df: pd.DataFrame, output_path: Path) -> None:
    """Print distribution metrics and save to parquet."""
    if labels_df.empty:
        print("Error: Generated labels dataframe is empty.")
        return
        
    print("\n--- Label Generation Summary ---")
    num_windows = labels_df["window_end_date"].nunique()
    print(f"Total time windows generated: {num_windows}")
    
    print("\nLabel Distribution by Strategy (1 = Beat Buy & Hold):")
    distributions = []
    
    for strategy in labels_df["strategy"].unique():
        strat_df = labels_df[labels_df["strategy"] == strategy]
        positives = strat_df["label"].sum()
        total = len(strat_df)
        pct = (positives / total) * 100
        distributions.append({"Strategy": strategy, "Positives": positives, "Total": total, "Win Rate (%)": pct})
        print(f"  {strategy:15s}: {pct:.1f}% positives ({positives}/{total})")
        
        if pct < 15 or pct > 85:
            print(f"    ⚠️ Warning: {strategy} has a highly skewed distribution.")
            
    # Save the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(output_path)
    print(f"\nSaved labeled data to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nifty50.parquet", help="Path to input price data")
    parser.add_argument("--output", default="data/labeled_data.parquet", help="Path to save labeled data")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_path = Path(args.output)
    
    if not data_path.exists():
        print(f"Error: Could not find data file {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)
    
    # We ensure we have enough data
    if len(df) < 252 + 63:
        print(f"Error: Not enough data. Need at least 315 days, got {len(df)}")
        return
        
    labels_df = generate_labels(df)
    validate_and_save(labels_df, output_path)


if __name__ == "__main__":
    main()
