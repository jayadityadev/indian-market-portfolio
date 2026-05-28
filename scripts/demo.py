"""Headless demo — generates a markdown report with strategy metrics, regime analysis, and recommendation.

Run: uv run python demo.py
Output: docs/demo_report.md
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from backtester import run_backtest
from data_pipeline import load_data
from regime_detector import get_current_regime, get_regime_performance
from classifier_inference import get_strategy_probabilities
from risk_forecaster import simulate_drawdowns
from strategies import (
    buy_and_hold, ma_crossover, rsi_strategy, momentum_strategy,
    bollinger_bands, dual_momentum,
)

STRATEGY_MAP = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Bollinger Bands": bollinger_bands,
    "Dual Momentum": dual_momentum,
}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "nifty50.parquet"
    regime_path = project_root / "data" / "nifty50_regimes.parquet"
    models_dir = project_root / "models"
    output_path = project_root / "docs" / "demo_report.md"

    print("📊 Loading data...")
    df = load_data(data_path)
    print(f"   {len(df)} trading days | {df.index[0].date()} to {df.index[-1].date()}")

    # Run all strategies
    print("🎯 Running 6 strategies...")
    results = {}
    for name, func in STRATEGY_MAP.items():
        signals = func(df)
        results[name] = run_backtest(df["Close"], signals)
        m = results[name]["metrics"]
        print(f"   {name:20s} | CAGR: {m['CAGR']:7.1%} | Sharpe: {m['Sharpe']:6.2f}")

    # Regime detection
    print("📈 Detecting regimes...")
    regime_df = pd.read_parquet(regime_path)
    regime_df.index = pd.to_datetime(regime_df.index)
    current_regime = get_current_regime(df)
    print(f"   Current regime: {current_regime}")

    # Regime distribution
    regime_counts = regime_df["regime"].dropna().astype(str).value_counts()

    # Regime performance
    regime_perf = get_regime_performance(regime_df, results)

    # Classifier
    print("🤖 Running classifier inference...")
    probs = get_strategy_probabilities(df.iloc[-252:], current_regime, models_dir=models_dir)
    max_prob = max(probs.values()) if probs else 0.0
    if max_prob >= 0.55:
        recommended = max(probs, key=probs.get)
        source = "ML Classifier"
    else:
        # Fallback to historical Sharpe
        rp = regime_perf[regime_perf["regime"] == current_regime]
        rp = rp[rp["strategy"] != "Buy & Hold"]
        if not rp.empty:
            recommended = str(rp.sort_values("Sharpe", ascending=False).iloc[0]["strategy"])
        else:
            recommended = "Momentum"
        source = "Historical Sharpe (ML confidence < 55%)"
    print(f"   Recommended: {recommended} ({source})")

    # Risk forecast
    print("🛡️  Computing risk forecast...")
    strat_eq = results[recommended]["equity_curve"]
    strat_rets = strat_eq.pct_change().dropna()
    rdates = regime_df.index[regime_df["regime"].astype(str) == current_regime]
    valid = strat_rets.index.intersection(rdates)
    regime_rets = strat_rets.loc[valid]
    forecast = simulate_drawdowns(regime_rets, n_simulations=1000, horizon=63)

    # Build report
    print("📝 Generating report...")
    lines = [
        f"# Demo Report — Indian Market Portfolio Intelligence",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Data:** NIFTY 50 ({df.index[0].date()} to {df.index[-1].date()}) — {len(df)} trading days",
        f"",
        f"---",
        f"",
        f"## Strategy Performance",
        f"",
        f"| Strategy | CAGR | Sharpe | Max Drawdown | Volatility |",
        f"|----------|------|--------|--------------|------------|",
    ]

    for name, res in results.items():
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['CAGR']:.1%} | {m['Sharpe']:.2f} | {m['MaxDrawdown']:.1%} | {m['Volatility']:.1%} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Regime Analysis",
        f"",
        f"### Regime Distribution",
        f"",
        f"| Regime | Days | % of Period |",
        f"|--------|------|-------------|",
    ])

    total_days = int(regime_counts.sum())
    for regime_name in ["Bull", "Sideways", "Bear"]:
        count = int(regime_counts.get(regime_name, 0))
        pct = count / total_days * 100 if total_days > 0 else 0
        lines.append(f"| {regime_name} | {count} | {pct:.1f}% |")

    lines.extend([
        f"",
        f"### Current Regime: **{current_regime}**",
        f"",
        f"### Per-Regime Sharpe Ratios",
        f"",
    ])

    # Pivot table
    if not regime_perf.empty:
        pivot = regime_perf.pivot_table(
            index="strategy", columns="regime", values="Sharpe", aggfunc="first"
        )
        lines.append(f"| Strategy | " + " | ".join(pivot.columns) + " |")
        lines.append(f"|----------|" + "|".join(["--------"] * len(pivot.columns)) + "|")
        for strat_name, row in pivot.iterrows():
            vals = " | ".join(f"{v:.2f}" if pd.notna(v) else "N/A" for v in row)
            lines.append(f"| {strat_name} | {vals} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Strategy Recommendation",
        f"",
        f"- **Recommended Strategy:** {recommended}",
        f"- **Selection Method:** {source}",
        f"",
        f"### ML Classifier Probabilities (beating Buy & Hold in next ~63 days)",
        f"",
        f"| Strategy | Probability |",
        f"|----------|-------------|",
    ])

    for strat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {strat} | {prob:.1%} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Risk Forecast ({recommended} in {current_regime} regime, next 63 days)",
        f"",
        f"| Scenario | Max Drawdown |",
        f"|----------|-------------|",
        f"| Best Case (90th pctl) | {forecast['best_case_90']:.1%} |",
        f"| Median (50th pctl) | {forecast['median_50']:.1%} |",
        f"| Worst Case (10th pctl) | {forecast['worst_case_10']:.1%} |",
        f"",
        f"*Based on 1,000 bootstrap simulations of historical returns within the {current_regime} regime.*",
        f"",
        f"---",
        f"",
        f"## Disclaimer",
        f"",
        f"This report is for educational and research purposes only. Past performance does not guarantee future results. "
        f"Always consult a qualified financial advisor before making investment decisions.",
    ])

    report = "\n".join(lines) + "\n"
    output_path.write_text(report)
    print(f"\n✅ Report saved to {output_path}")
    print(f"   Recommendation: {recommended} ({source})")
    print(f"   Risk (median drawdown): {forecast['median_50']:.1%}")


if __name__ == "__main__":
    main()
