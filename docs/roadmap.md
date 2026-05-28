# Indian Market Portfolio Intelligence — Full Project Roadmap

> Strategy evaluation system for Indian equities. NOT price predictor.
> ML-augmented. Regime-aware. Decision-support.

---

## Part 1 — What System Does (4 Core Questions)

1. How did strategy perform historically?
2. Under what market conditions does it work best?
3. Will it work in current conditions?
4. What is downside risk if used now?

---

## Part 2 — System Architecture (Full)

```
[User Input: Streamlit UI]
         ↓
[Data Module]
  ├── yfinance → OHLC (^NSEI / any NSE ticker)
  ├── Feature Engineering:
  │     returns, rolling_vol (20d), momentum (RoC), drawdown_from_peak
  └── Output: clean parquet/CSV

         ↓
[Backtesting Engine]
  ├── Strategy Library (fixed, no user-defined):
  │     Buy & Hold / MA Crossover / RSI / Momentum / Mean Reversion
  ├── Portfolio Simulator:
  │     signals → portfolio value over time (long-only, no leverage)
  │     signal lag: enter T+1 (no lookahead)
  └── Output Metrics:
        CAGR, Sharpe (rf=6%), Max Drawdown, Annualized Volatility

         ↓
[Regime Detection — Unsupervised ML]
  ├── KMeans (k=3): Bull / Bear / Sideways
  ├── Features: returns + volatility + momentum + drawdown (normalized)
  ├── Post-hoc centroid interpretation → label assignment
  ├── Map regime labels → historical dates
  └── Detect current regime from recent window

         ↓
[Strategy Suitability — Supervised ML]  ← Post-MVP
  ├── Classification: strategy beats benchmark? (1/0)
  ├── Features: regime label + market features
  ├── Models: Random Forest, Gradient Boosting, Logistic Regression
  └── Output: probability score + recommended strategy

         ↓
[Risk Forecasting]  ← Post-MVP
  ├── NOT return prediction
  ├── Expected drawdown range
  ├── Volatility bands
  └── Probabilistic risk estimates

         ↓
[Results Aggregator]
  ├── Per-strategy full metrics
  ├── Per-regime strategy performance breakdown
  └── Recommendation: best Sharpe in current regime

         ↓
[Dashboard: Streamlit + Plotly]
  ├── Equity curve (strategy vs benchmark)
  ├── Metrics table
  ├── Regime timeline (colored bar)
  ├── Regime-wise performance heatmap / grouped bar
  └── Recommendation box
```

---

## Part 3 — Phased Roadmap

> **Current status:** Phases 0–4 complete. Classifier (Module A) also built ahead of schedule. Dashboard (`src/app.py`) + integration remaining.

### Phase 0 — Foundation ✅ DONE

**Goal:** Clean data flowing in. No computation yet.

Steps:
1. Setup project structure + virtualenv
2. Fetch NIFTY 50 OHLC via `yfinance` (`^NSEI`)
3. Feature engineering:
   - `daily_returns = close.pct_change()`
   - `rolling_vol = returns.rolling(20).std() * sqrt(252)`
   - `momentum = close / close.shift(252) - 1` (12-month RoC)
   - `drawdown = close / close.cummax() - 1`
4. Save → `data/nifty50.parquet`
5. Validate: no NaNs, correct date index, sane ranges

**Pitfall:** Rolling features create NaN at start. Drop cleanly. Do NOT forward-fill price-derived features — that's lookahead.

**Output:** `data_pipeline.py` → `load_data(ticker, start, end) → DataFrame`

---

### Phase 1 — Backtesting Engine ✅ DONE

**Goal:** Core simulator. Takes price + signals → metrics + equity curve.

**Signal Contract:**
```python
# Every strategy returns this:
signals: pd.Series  # index=date, values={0, 1}
# 1 = long, 0 = flat
# Enter at T+1 open (or T+1 close approx) — NOT same candle
```

**Portfolio Simulator Logic:**
```
position[t] = signal[t-1]   # lag by 1 day
daily_pnl[t] = position[t] * return[t]
portfolio_value[t] = portfolio_value[t-1] * (1 + daily_pnl[t])
```

**Metrics:**
| Metric | Formula |
|--------|---------|
| CAGR | `(end/start)^(252/n_days) - 1` |
| Sharpe | `(mean(daily_ret) - rf/252) / std(daily_ret) * sqrt(252)` |
| Max Drawdown | `min(equity/equity.cummax() - 1)` |
| Volatility | `std(daily_ret) * sqrt(252)` |

`rf = 0.06` (India). `trading_days = 252`.

**Sanity check:** Buy & Hold on NIFTY 2015–2024 → CAGR ~12–14%. Sharpe ~0.5–0.8. If wildly off, simulator bug.

**Output:** `backtester.py` → `run_backtest(prices, signals) → {metrics_dict, equity_curve}`

---

### Phase 2 — Strategy Library ✅ DONE

**Goal:** 4 (→5) strategies, each as single function, plugged into backtester.

**Strategy 1: Buy & Hold**
```python
signals = pd.Series(1, index=df.index)
```
Baseline. Always compare everything to this.

**Strategy 2: MA Crossover**
```python
sma50 = close.rolling(50).mean()
sma200 = close.rolling(200).mean()
signals = (sma50 > sma200).astype(int)
```
Works in trending markets. Fails sideways (whipsaw).

**Strategy 3: RSI**
```python
rsi = compute_rsi(close, 14)
signals = pd.Series(0, index=df.index)
position = 0
for i, r in enumerate(rsi):
    if r < 30: position = 1   # oversold → buy
    if r > 70: position = 0   # overbought → sell
    signals.iloc[i] = position
```
Mean-reversion style. Good sideways. Bad in strong trends.

**Strategy 4: Momentum**
```python
mom_12m = close / close.shift(252) - 1
signals = (mom_12m > 0).astype(int)
```
Trend-following. Long when last 12m positive. Else flat.

**Strategy 5: Mean Reversion (optional)**
```python
zscore = (close - close.rolling(20).mean()) / close.rolling(20).std()
signals = (zscore < -1.5).astype(int)
```
Short-term dips. Higher churn.

**Pitfall:** RSI and Mean Reversion are stateful — loop, don't vectorize naively. Vectorization creates lookahead.

**Output:** `strategies.py` → 5 functions, all tested through backtester.

---

### Phase 3 — Regime Detection ✅ DONE

**Goal:** Label every historical date with market regime. Identify current regime.

**Why regimes?** Markets behave differently. MA Crossover crushes in bull markets. RSI better in sideways. Need to know WHICH regime to recommend WHICH strategy.

**Features for Clustering:**
```python
features = df[['daily_returns', 'rolling_vol', 'momentum', 'drawdown']]
# Normalize — critical. KMeans is distance-based. Unscaled features bias clusters.
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(features.dropna())
```

**Model:**
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)
```

**Label Interpretation (post-hoc):**
Examine cluster centroids:
- High returns + low vol + positive momentum → **Bull**
- Negative returns + high vol + negative momentum → **Bear**
- Near-zero returns + medium vol → **Sideways**

Don't hardcode label → cluster mapping. Centroid values determine it each fit.

**Per-Regime Performance:**
```python
for regime in [0, 1, 2]:
    mask = (df['regime'] == regime)
    regime_returns = strategy_returns[mask]
    # compute Sharpe/CAGR/MaxDD for that slice
```

**Current Regime Detection:**
```python
recent = features.tail(60)  # last 60 trading days
current_features = StandardScaler().transform(recent.mean().values.reshape(1,-1))
current_regime = km.predict(current_features)[0]
```

**Output:** `regime_detector.py` → `fit_regimes(df)`, `get_current_regime()`, per-regime breakdown table

---

### Phase 4 — Dashboard ⏳ IN PROGRESS

**Goal:** Visual frontend, everything wired.

**Layout:**
```
Sidebar:
  - Stock selector (NIFTY 50 hardcoded for MVP, can expand)
  - Date range picker
  - Strategy selector (or "Auto" mode)

Main:
  [Tab 1: Performance]
    - Equity curve (all strategies + benchmark)
    - Metrics table

  [Tab 2: Regime Analysis]
    - Regime timeline (colored bar by date)
    - Per-regime strategy heatmap (rows=strategy, cols=regime, values=Sharpe)

  [Tab 3: Recommendation]
    - Current regime label
    - Best strategy in current regime (by historical Sharpe)
    - "Warning: past performance ≠ future"
```

**Auto Mode Logic:**
```python
best_strategy = regime_performance_table
    .loc[regime_performance_table['regime'] == current_regime]
    .sort_values('sharpe', ascending=False)
    .iloc[0]['strategy']
```

**Output:** `app.py` → `streamlit run app.py`

---

### Phase 5 — Integration + Polish (Day 6)

**Goal:** End-to-end pipeline without breaks. Edge case handling.

Edge cases:
- Date range < 200 days → MA Crossover signals all NaN → show warning, skip strategy
- Single-regime historical period → regime comparison meaningless → note it
- yfinance timeout → retry + cache parquet

**Walk-forward sanity check (not full WFV for MVP):**
Split data 70/30. Run regime detection on train. Apply labels to test. Check if regime-strategy mapping holds. If completely opposite → note as limitation.

**Deliverable:** Full working prototype. `requirements.txt`. `README.md`.

---

### Phase 6 — Buffer + Presentation (Day 7)

Demo flow:
1. NIFTY 50, 2015–2024
2. All strategies
3. Walk through tabs
4. Show regime breakdown
5. Show recommendation

Known limitations to document:
- No transaction costs
- No slippage model
- Survivorship bias (using current NIFTY 50 composition)
- KMeans regime labels not stable across different date ranges
- Strategy suitability classifier not built (post-MVP)
- Risk forecasting not built (post-MVP)

---

## Part 4 — Post-MVP Roadmap

### Module A: Strategy Suitability Classifier ✅ BUILT (ahead of schedule)

**Files:** `src/classifier_features.py`, `src/classifier_inference.py`, `src/label_generator.py`, `src/train_classifier.py`

**Trained models in `models/`:** MA_Crossover, Momentum, RSI (each has `_classifier.pkl` + `_scaler.pkl`)

**Next:** Wire classifier inference into dashboard Tab 3 (Recommendation).


**Problem:** Given current regime + market features, which strategy will outperform?

**Dataset construction:**
```
For each (strategy, time_window):
  - Features: regime label, vol, momentum, drawdown, market trend
  - Target: 1 if strategy_sharpe > benchmark_sharpe else 0
```

**Models (in order of try):**
1. Logistic Regression (baseline, interpretable)
2. Random Forest (captures nonlinear regime interactions)
3. Gradient Boosting / XGBoost (best expected performance)

**Validation:** Time-series cross-validation (no shuffle split — data is temporal).

**Output:** Probability score per strategy. Top strategy recommendation with confidence.

---

### Module B: Risk Forecasting

**Problem:** What is the downside risk of using strategy X in current conditions?

**NOT:** "price will go to X"
**YES:** "expected max drawdown in next 60 days = 8–15% with 80% confidence"

**Approach:**
1. Historical simulation: for each regime, collect drawdown distributions
2. Bootstrap: sample from historical regime-drawdown distribution
3. Output: percentile bands (10th, 50th, 90th) for expected drawdown

**Alternative:** GARCH model for volatility forecasting → convert to drawdown estimate.

---

### Module C: Multi-Asset / Portfolio

Post-MVP:
- Expand from NIFTY 50 index to individual stocks
- Portfolio of stocks with regime-aware allocation
- Correlation-aware diversification

---

### Module D: FastAPI Backend

For production version:
- Replace Streamlit pipeline with FastAPI endpoints
- `/backtest` → run backtest, return metrics JSON
- `/regime` → get current regime
- `/recommend` → get strategy recommendation
- PostgreSQL for storing backtest results, user sessions
- Streamlit OR React frontend consuming API

---

## Part 5 — Tech Stack (Full)

| Layer | MVP | Post-MVP |
|-------|-----|---------|
| Data | yfinance + parquet | NSEpy + PostgreSQL |
| Backend | Direct Python modules | FastAPI |
| Database | SQLite / parquet | PostgreSQL |
| ML | scikit-learn | scikit-learn + optuna |
| Frontend | Streamlit | Streamlit or React |
| Viz | Plotly | Plotly |
| Deployment | Local | Docker + cloud |

---

## Part 6 — Critical Constraints (Never Violate)

1. **No lookahead.** Signal at T uses only data up to T. Enter at T+1.
2. **trading_days = 252.** Not 365 (that's crypto).
3. **rf = 6% annualized** = `0.06/252` daily.
4. **No short selling.** Signals ∈ {0, 1} only.
5. **No transaction costs** for MVP (document as limitation).
6. **Fixed strategy params** for MVP. No optimization = no overfitting.
7. **Always show Buy & Hold as benchmark.**
8. **Regime detection = differentiator.** Don't cut it.

---

## Part 7 — What "Done" Looks Like

MVP done when:
- User selects NIFTY 50, picks date range, picks strategy
- System shows equity curve, 4 metrics, regime timeline, per-regime breakdown, recommendation
- All 4+ strategies run without error
- Regime detection labels make intuitive sense (2020 COVID = Bear, 2021 rally = Bull)
- Auto mode recommends reasonable strategy

Project done (full) when:
- Strategy suitability classifier trained + validated
- Risk forecasting module outputs drawdown bands
- FastAPI backend serving results
- Clean UI with full documentation

---

## Part 8 — File Structure (Actual)

```
indian-market-intelligence/
├── main.py                             # Entry point (run app)
├── pyproject.toml                      # Project config + deps
├── requirements.txt
├── uv.lock
├── README.md
├── TODO.md                             # Task tracking
│
├── src/                                # All source modules
│   ├── app.py                          # Streamlit dashboard
│   ├── data_pipeline.py                # yfinance fetch + feature engineering
│   ├── backtester.py                   # Core backtest engine
│   ├── strategies.py                   # Strategy library (BH, MA, RSI, Momentum)
│   ├── regime_detector.py              # KMeans regime detection
│   ├── classifier_features.py          # Feature engineering for suitability classifier
│   ├── classifier_inference.py         # Load models + run inference
│   ├── label_generator.py              # Generate training labels (strategy beats BH?)
│   ├── train_classifier.py             # Train per-strategy classifiers
│   └── utils.py                        # Shared helpers (metrics, plotting)
│
├── data/
│   ├── nifty50.parquet                 # Raw OHLC + features
│   ├── nifty50_regimes.parquet         # Data with regime labels
│   ├── labeled_data.parquet            # Classifier training data
│   └── regime_model.pkl                # (duplicate — canonical in models/)
│
├── models/
│   ├── MA_Crossover_classifier.pkl     # Trained RF for MA Crossover
│   ├── MA_Crossover_scaler.pkl
│   ├── Momentum_classifier.pkl         # Trained RF for Momentum
│   ├── Momentum_scaler.pkl
│   ├── RSI_classifier.pkl              # Trained RF for RSI
│   └── RSI_scaler.pkl
│
├── regime_model.pkl                    # ← move to models/ (cleanup needed)
│
└── docs/
    ├── SKILLS_REFERENCE.md
    ├── mvp_roadmap.md
    ├── classifier_agent_instructions.md
    ├── demo_script.md
    ├── limitations.md
    ├── summary_report.md
    └── screenshots/
        ├── 01_equity_curve.png
        ├── 02_metrics_table.png
        ├── 03_regime_timeline.png
        ├── 04_regime_heatmap.png
        └── 05_recommendation.png
```

### Cleanup Note
`regime_model.pkl` exists at root AND `data/`. Canonical location → `models/`. Both should be removed; `models/` version kept.

### Post-MVP additions (to add under `src/`)
```
src/
├── risk_forecaster.py          # Bootstrap drawdown bands
├── api/                        # FastAPI backend (post-MVP)
│   ├── main.py
│   ├── routes/
│   │   ├── backtest.py
│   │   ├── regime.py
│   │   └── recommend.py
│   └── schemas.py
└── db/                         # PostgreSQL models (post-MVP)
    ├── models.py
    └── session.py
```