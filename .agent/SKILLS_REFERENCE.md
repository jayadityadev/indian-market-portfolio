# Agent Skills Reference
## Indian Market Portfolio Intelligence Platform

> **Agent instruction:** At start of each session, read this file. Load listed skills for current day by reading their SKILL.md. Use skill workflows when building — don't reinvent what skills already define.

Skills location: `.agent/skills/<skill-name>/SKILL.md`

---

## Day-by-Day Skill Map

### Day 1 — Data Pipeline ✓ DONE
**Skills:** `feature-engineering`
- Use for: returns, rolling volatility, momentum, drawdown computation
- Key ref: `.agent/skills/feature-engineering/references/pitfalls.md` — avoid lookahead in features

---

### Day 2 — Backtesting Engine
**Primary:** `backtest-expert`
**Supporting:** `portfolio-analytics`

- `backtest-expert` → signal contract design, simulator logic, metric formulas (CAGR, Sharpe, MaxDD, Volatility)
- `portfolio-analytics` → cross-check metric implementations against `references/metrics_guide.md`
- Key ref: `.agent/skills/backtest-expert/references/methodology.md`
- Key script: `.agent/skills/portfolio-analytics/scripts/analyze_portfolio.py`

**Critical note from skill audit:** `volatility-modeling` uses `sqrt(365)` (crypto). This project uses `sqrt(252)` (Indian equity). Override wherever scripts assume 365 trading days.

---

### Day 3 — Strategy Library
**Primary:** `backtest-expert`
**Supporting:** `feature-engineering`

- `backtest-expert` → run each strategy through backtester, validate outputs
- `feature-engineering` → confirm signal inputs (RSI, SMA, momentum) computed without lookahead
- Key ref: `.agent/skills/backtest-expert/references/failed_tests.md` — common strategy bugs
- Key script: `.agent/skills/backtest-expert/scripts/evaluate_backtest.py`

---

### Day 4 — Regime Detection
**Primary:** `regime-detection`
**Supporting:** `volatility-modeling`, `feature-engineering`

- `regime-detection` → KMeans clustering setup, centroid interpretation, regime labeling
- `volatility-modeling` → volatility estimators as input features to clustering (use Parkinson or Garman-Klass for efficiency)
- `feature-engineering` → normalize features before clustering
- Key refs:
  - `.agent/skills/regime-detection/references/methodology.md`
  - `.agent/skills/regime-detection/references/strategy_adaptation.md`
- Key scripts:
  - `.agent/skills/regime-detection/scripts/detect_regime.py`
  - `.agent/skills/regime-detection/scripts/regime_backtest.py`

**Critical note:** `volatility-modeling` regime thresholds are crypto-calibrated (50–150% annualized). NIFTY 50 normal vol = 15–25%. Ignore crypto vol tables; use percentile-based classification instead.

---

### Day 5 — Streamlit Dashboard
**Primary:** `trading-visualization`
**Supporting:** `portfolio-analytics`

- `trading-visualization` → equity curve, drawdown chart, regime timeline, trade markers
- `portfolio-analytics` → strategy comparison table, per-regime performance breakdown
- Key refs:
  - `.agent/skills/trading-visualization/references/chart_recipes.md`
  - `.agent/skills/trading-visualization/references/styling_guide.md`
- Key scripts:
  - `.agent/skills/trading-visualization/scripts/chart_generator.py`
  - `.agent/skills/portfolio-analytics/scripts/compare_strategies.py`

---

### Day 6 — Integration + Polish
**Primary:** `walk-forward-validation`, `risk-management`
**Supporting:** `portfolio-analytics`

- `walk-forward-validation` → verify no overfitting in strategy params, check out-of-sample performance
- `risk-management` → drawdown controls, exposure limits for recommendation logic
- `portfolio-analytics` → final end-to-end metrics audit
- Key refs:
  - `.agent/skills/walk-forward-validation/references/overfit_detection.md`
  - `.agent/skills/walk-forward-validation/references/practical_guide.md`
  - `.agent/skills/risk-management/references/drawdown_management.md`
- Key scripts:
  - `.agent/skills/walk-forward-validation/scripts/overfit_detector.py`
  - `.agent/skills/walk-forward-validation/scripts/walk_forward.py`
  - `.agent/skills/risk-management/scripts/drawdown_analyzer.py`

---

### Day 7 — Buffer / Presentation Prep
**Primary:** `trading-visualization`, `portfolio-analytics`

- `trading-visualization` → final chart polish, screenshot-ready outputs
- `portfolio-analytics` → summary metrics for presentation slide
- Key script: `.agent/skills/trading-visualization/scripts/performance_report.py`

---

## Full Skills Inventory

| Skill | Days Active | Purpose |
|---|---|---|
| `backtest-expert` | 2, 3 | Core backtesting logic + validation |
| `feature-engineering` | 1✓, 3, 4 | Feature computation, lookahead prevention |
| `portfolio-analytics` | 2, 5, 6, 7 | Metrics, strategy comparison |
| `regime-detection` | 4 | KMeans clustering, regime labeling |
| `trading-visualization` | 5, 7 | All charts and dashboards |
| `volatility-modeling` | 4 | Vol estimators as clustering features |
| `walk-forward-validation` | 6 | Overfitting detection, robustness check |
| `risk-management` | 6 | Drawdown controls, exposure limits |
| `quant-analyst` | any | General quant reasoning, strategy design questions |

---

## Global Overrides (apply every day)

1. **Trading days = 252** not 365. Any skill using 365 → override to 252.
2. **Risk-free rate = 6%** (India), i.e. `rf = 0.06 / 252` daily.
3. **Data source = yfinance** (`^NSEI` for NIFTY 50). No live feeds.
4. **No short selling.** Signals are 0 or 1 only.
5. **No transaction costs** for MVP. Note this as limitation.
6. **Crypto-specific content in skills** (DeFi, DEX, wallets) → ignore entirely.