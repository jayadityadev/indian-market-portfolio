# Demo Report — Indian Market Portfolio Intelligence

**Generated:** 2026-05-09 18:50
**Data:** NIFTY 50 (2015-04-01 to 2024-12-30) — 2398 trading days

---

## Strategy Performance

| Strategy | CAGR | Sharpe | Max Drawdown | Volatility |
|----------|------|--------|--------------|------------|
| Buy & Hold | 11.2% | 0.36 | -38.4% | 16.6% |
| MA Crossover | 4.8% | -0.05 | -41.9% | 12.4% |
| RSI | 3.5% | -0.16 | -32.7% | 11.7% |
| Momentum | 7.2% | 0.14 | -23.7% | 12.0% |
| Bollinger Bands | 5.5% | 0.02 | -39.9% | 13.6% |
| Dual Momentum | 3.4% | -0.19 | -35.9% | 10.8% |

---

## Regime Analysis

### Regime Distribution

| Regime | Days | % of Period |
|--------|------|-------------|
| Bull | 1705 | 62.4% |
| Sideways | 976 | 35.7% |
| Bear | 51 | 1.9% |

### Current Regime: **Sideways**

### Per-Regime Sharpe Ratios

| Strategy | Bear | Bull | Sideways |
|----------|--------|--------|--------|
| Bollinger Bands | 0.18 | 0.26 | 0.54 |
| Buy & Hold | 0.18 | 0.89 | 0.98 |
| Dual Momentum | 0.00 | 0.05 | 0.27 |
| MA Crossover | -3.26 | 0.21 | 0.41 |
| Momentum | 0.00 | 0.48 | 0.66 |
| RSI | 0.18 | 0.02 | 0.34 |

---

## Strategy Recommendation

- **Recommended Strategy:** Momentum
- **Selection Method:** Historical Sharpe (ML confidence < 55%)

### ML Classifier Probabilities (beating Buy & Hold in next ~63 days)

| Strategy | Probability |
|----------|-------------|
| RSI | 44.8% |
| MA Crossover | 33.3% |
| Momentum | 33.3% |

---

## Risk Forecast (Momentum in Sideways regime, next 63 days)

| Scenario | Max Drawdown |
|----------|-------------|
| Best Case (90th pctl) | -5.4% |
| Median (50th pctl) | -10.4% |
| Worst Case (10th pctl) | -17.6% |

*Based on 1,000 bootstrap simulations of historical returns within the Sideways regime.*

---

## Disclaimer

This report is for educational and research purposes only. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.
