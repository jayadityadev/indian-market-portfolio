# 2-Minute Demo Script

## Setup (before presenting)
- App running: `uv run streamlit run src/app.py`
- Browser open at localhost:8501
- Settings pre-filled: NIFTY 50, 2015–2024, Auto mode

---

## Script

**[0:00–0:20] — Problem**
"Most retail investors copy strategies without validating them.
This platform answers: which strategy actually works — and under what market conditions?"

**[0:20–0:40] — Run the system**
Click "Run Analysis". While loading:
"The system fetches 10 years of NIFTY 50 data, runs 4 strategies through our backtesting engine,
and detects market regimes using KMeans clustering."

**[0:40–1:10] — Show equity curve + metrics**
"Here's 10 years of performance. Buy & Hold is always the benchmark.
Notice that while Buy & Hold has the highest CAGR, the Momentum strategy achieves better risk-adjusted returns during non-trending periods. A higher return with higher drawdown isn't always better."

**[1:10–1:35] — Show regime timeline + heatmap**
"This is the differentiator. The timeline shows WHEN market was Bull, Bear, or Sideways.
The heatmap shows each strategy's Sharpe ratio per regime.
Look at MA Crossover: it performs okay in Bull regimes but significantly underperforms in Sideways. Same strategy, very different behavior across regimes."

**[1:35–2:00] — Recommendation + close**
"Current regime is Sideways. System recommends Momentum based on historical regime performance.
This is not a price prediction — it's decision support grounded in evidence."

---

## Anticipated Questions

**Q: Why not predict future prices?**
A: Prices are noisy. Strategy robustness across regimes is more useful and more honest.

**Q: How are regimes detected?**
A: KMeans clustering on 4 features — returns, volatility, momentum, drawdown. Unsupervised — no labels needed.

**Q: What's next beyond MVP?**
A: Strategy suitability classifier (supervised ML) and probabilistic risk forecasting module.

**Q: Why NIFTY 50 only?**
A: MVP scope. Architecture supports any yfinance ticker — just expand the selector.
