# Agent Instructions — Strategy Suitability Classifier
## Indian Market Portfolio Intelligence Platform

---

## Context Paste (read every session)

> We're building an Indian Market Portfolio Intelligence & Backtesting Platform — ML-augmented strategy evaluation system for Indian equities (NOT a price predictor). Stack: Streamlit frontend, yfinance data, pandas/numpy, scikit-learn, Plotly, SQLite/parquet storage. MVP is DONE: backtesting engine, 4 strategies (Buy & Hold, MA Crossover, RSI, Momentum), KMeans regime detection, Streamlit dashboard.
>
> **Current module: Strategy Suitability Classifier** — ML model that predicts probability of a strategy beating the benchmark under current market conditions. This replaces the rule-based "best historical Sharpe" recommendation with a calibrated probability score.

---

## TODO Tracking

TODO.md exists in project root. At start of every response:
1. Read TODO.md
2. Print current state (all tasks, checked/unchecked)
3. Check off any tasks completed in the previous step
4. Update the file before proceeding

Format:
```
## TODO State
- [x] Task done
- [ ] Task pending
```

---

## Skills Reference

Read `docs/SKILLS_REFERENCE.md` at session start to understand which agent skills apply to which tasks.

**Skills relevant to this module:**

| Skill | Use for |
|---|---|
| `backtest-expert` | Rolling window backtesting logic, metric computation per window |
| `feature-engineering` | Feature extraction per window, lookahead prevention |
| `walk-forward-validation` | Time-series split methodology, overfitting detection |
| `portfolio-analytics` | Per-window Sharpe computation, benchmark comparison |
| `quant-analyst` | Methodology questions — labeling logic, horizon selection |

**Before writing any code for a task**, read the relevant skill's SKILL.md:
```
.agent/skills/<skill-name>/SKILL.md
```
Use the skill's methodology, scripts, and references. Do not reinvent what skills already define.

---

## Global Overrides (always apply)

1. Trading days = **252** not 365
2. Risk-free rate = **6% annualized** → `rf_daily = 0.06 / 252`
3. Data source = **yfinance** (`^NSEI` for NIFTY 50)
4. No short selling — signals are 0 or 1 only
5. No transaction costs for MVP (note as limitation)
6. Ignore any crypto-specific content in skills

---

## What This Module Does (Understand Before Building)

### The Problem with Rule-Based Recommendation
Current system says: *"In Bull regime → Momentum had best Sharpe historically"*
This is a lookup table. It doesn't generalize. It doesn't quantify confidence.

### What the Classifier Does
Trained ML model says: *"Given current market features → Momentum has 73% probability of beating Buy & Hold in next 63 days"*

This is better because:
- Quantifies uncertainty (probability, not just yes/no)
- Uses richer features beyond just regime label
- Generalizes to unseen market conditions
- Upgrades the recommendation from heuristic to data-driven

### The Core ML Framing
- **Task type:** Binary classification (per strategy)
- **One model per strategy** — cleaner, easier to debug, interpretable
- **Input X:** Market features at time T (regime, volatility, momentum, RSI level, etc.)
- **Output y:** Did strategy beat Buy & Hold in the next 63 days? (1 = yes, 0 = no)
- **Why 63 days?** ~1 quarter. Long enough to be meaningful, short enough for enough training samples from 10yr history.

---

## Phase 1 — Label Generation

### Your Goal
Generate a labeled dataset where each row = one time window, each column = features + label (did strategy beat benchmark?). This is the training data for the classifier.

### Why Rolling Windows?
Financial data is a time series — there's only one history. Can't collect more data. Rolling windows = synthetic "many observations" from single time series. Each window is a market scenario the model learns from.

### Why lookback=252 (1 year)?
One year of daily data captures: one full cycle of seasonal patterns, enough vol/momentum signal, regime context. Shorter = noisy. Longer = fewer windows.

### Why forward=63 (~1 quarter)?
Label horizon = how far ahead you're predicting. Too short (5–10 days) = noise dominates. Too long (252 days) = too few windows, market changes too much. 63 days = practical investment horizon with signal.

### Why step=21 (~1 month)?
Step between windows. Controls overlap. step=1 = massive overlap, near-duplicate rows, pseudo-replication. step=21 = low overlap, more independent samples.

### Tasks

**Before writing code:**
1. Read `.agent/skills/backtest-expert/SKILL.md`
2. Read `.agent/skills/feature-engineering/SKILL.md`
3. Read `.agent/skills/quant-analyst/SKILL.md` — consult for rolling window labeling methodology
4. Explain to the user what methodology you're using and why before implementing

**Implementation:**
```python
# Pseudocode — implement properly using skill methodology
for each window_start in range(0, len(df) - lookback - forward, step):
    lookback_data = df[window_start : window_start + lookback]
    forward_data  = df[window_start + lookback : window_start + lookback + forward]
    
    features = extract_features(lookback_data)  # Phase 2
    
    for each strategy in [MA, RSI, Momentum, BH]:
        strategy_sharpe = run_backtest(forward_data, strategy).sharpe
        bh_sharpe       = run_backtest(forward_data, buy_and_hold).sharpe
        label = 1 if strategy_sharpe > bh_sharpe else 0
        
        rows.append({**features, 'strategy': strategy, 'label': label})
```

**Validate before proceeding:**
- Print label distribution per strategy — if any strategy has >80% same label, investigate
- A healthy dataset has 40–65% positives. Highly skewed = poor signal or bug in logic.
- Print number of windows generated — expect ~100–130 for 10yr data with step=21

**Explain to user:**
- How many windows were generated and why
- Label distribution per strategy
- Any strategy with surprising distribution

**Deliverable:** `label_generator.py`

---

## Phase 2 — Feature Engineering for Classifier

### Your Goal
For each window, extract a fixed-size feature vector from the lookback period. These features describe "what the market looked like" and become X in the ML model.

### Why These Features?

| Feature | Why |
|---|---|
| `avg_return` | Overall trend direction in lookback |
| `volatility` | High vol → different strategy works than low vol |
| `momentum` | Rate of change — trending vs mean-reverting conditions |
| `max_drawdown` | Severity of recent drawdown — risk environment |
| `regime_label` | Encoded cluster (Bull/Bear/Sideways) — direct regime signal |
| `regime_stability` | How many regime switches in lookback — stable vs chaotic |
| `rsi_at_end` | Market overbought/oversold at decision point |
| `sma_ratio` | 50SMA / 200SMA — trend structure (>1 = uptrend) |

### Tasks

**Before writing code:**
1. Re-read `.agent/skills/feature-engineering/SKILL.md` — specifically lookahead prevention
2. Consult `.agent/skills/feature-engineering/references/pitfalls.md`
3. Tell user which features you're computing and why each is informative

**Critical rule:** ALL features computed from lookback window only. Zero information from forward window. Any lookahead = data leakage = model useless in production.

**Normalization:** StandardScaler on numeric features before training. Save scaler per strategy (`models/<strategy>_scaler.pkl`) — must use same scaler at inference time.

**Deliverable:** `classifier_features.py` with function:
```python
def build_feature_matrix(df, windows, regime_labels) -> pd.DataFrame:
    # Returns DataFrame: one row per window, columns = features
```

---

## Phase 3 — Train & Evaluate Classifier

### Your Goal
Train one classifier per strategy. Evaluate properly. Save calibrated models.

### Why One Model Per Strategy?
Each strategy has different "when it works" logic. MA Crossover works in trending markets. RSI works in mean-reverting markets. Separate models = each model learns its own strategy's conditions. Shared model = diluted signal.

### Why TimeSeriesSplit, NOT Random Split?
**This is the most critical methodological point.**

Random split: test set contains windows from 2016 in train, 2020 in test — model "sees the future" during training.
TimeSeriesSplit: train on 2015–2020, test on 2021–2024. No information leakage across time.

**Never use `train_test_split` with shuffle=True on time series data. Ever.**

### Why Random Forest?
- Handles non-linear feature interactions (regime × volatility interactions)
- Robust to outliers
- Provides feature importance — tells you which features matter
- Less prone to overfit than deep trees when `max_depth` constrained

### Why Logistic Regression as Baseline?
- Simple, interpretable
- If RF doesn't beat LR significantly → features are linearly separable → no need for complexity
- Good sanity check

### Why Probability Calibration?
Raw RF probabilities are not well-calibrated (tend to cluster near 0 and 1). `CalibratedClassifierCV` adjusts output so "60% probability" actually means the event happens 60% of the time. Critical for trustworthy recommendations.

### Tasks

**Before writing code:**
1. Read `.agent/skills/walk-forward-validation/SKILL.md`
2. Read `.agent/skills/walk-forward-validation/references/overfit_detection.md`
3. Tell user the cross-validation methodology you're using and why

**Implementation notes:**
- `TimeSeriesSplit(n_splits=5)` — 5 folds
- RF params: `max_depth=4, n_estimators=100, min_samples_leaf=5` (prevent overfit on small dataset)
- Calibrate: `CalibratedClassifierCV(rf, method='isotonic', cv=3)`
- Evaluate per fold: accuracy, F1, AUC-ROC
- Print feature importances — explain to user which features drove predictions

**Report to user after training:**
```
Strategy: MA_Crossover
  Logistic Regression — AUC: 0.58, F1: 0.54
  Random Forest       — AUC: 0.63, F1: 0.61
  Selected: Random Forest
  Top features: regime_label (0.31), volatility (0.22), sma_ratio (0.18)
```

**What if AUC < 0.55 for all strategies?**
Tell the user — this means the model is barely better than random. Likely causes: too few samples (~100 windows), noisy labels, or strategy performance truly is unpredictable. This is honest and expected. The system still works — just show lower confidence scores and fallback to historical Sharpe.

**Deliverable:** `train_classifier.py` + `models/` directory with `.pkl` files

---

## Phase 4 — Dashboard Integration

### Your Goal
Load trained models at app startup. When user queries current conditions, extract features from most recent 252-day window, run predict_proba, display ranked probabilities.

### Why Fallback Logic?
Model trained on ~100 windows. Confidence matters. If highest probability across all strategies is < 0.55, model is essentially saying "I don't know." Show fallback: "Low confidence — defaulting to historical Sharpe ranking." Honest system > overconfident system.

### Tasks

**Before writing code:**
1. Review existing `app.py` recommendation section
2. Plan where classifier output fits in dashboard flow
3. Tell user what changes you're making to app.py and why

**`classifier_inference.py` must:**
```python
def get_strategy_probabilities(df_recent, regime_label) -> dict:
    # df_recent = last 252 days of data
    # Returns: {'MA_Crossover': 0.71, 'RSI': 0.58, 'Momentum': 0.63, 'Buy_Hold': 0.49}
```

**Dashboard changes:**
- Recommendation box: show ranked strategies with probability bars (Plotly horizontal bar)
- Add tooltip/caption: "Probability = model's confidence strategy beats Buy & Hold in next ~63 days"
- If confidence low: show warning banner, show historical Sharpe ranking instead
- Show which regime the model detected (existing feature, just surface it clearly)

**Explain to user:**
- What "73% probability" means in plain language
- Why the model might show low confidence
- What fallback means

---


## Communication Style with User

User is learning. For every significant decision:

1. **State what you're doing:** "I'm now generating rolling window labels."
2. **Explain why:** "We use rolling windows because financial data is a time series — we can't collect more data, so we extract multiple training samples by sliding a window across history."
3. **Show intermediate results:** Print window counts, label distributions, AUC scores.
4. **Flag surprises:** If label distribution is skewed, say so and explain what it means.
5. **Validate before moving on:** After each phase, show a summary of what was built and confirm it looks correct before proceeding.

Do not silently skip steps. Do not assume user understands quant methodology. Explain the intuition behind every design choice.

---

## Pitfalls — Watch For These

| Pitfall | What it looks like | Fix |
|---|---|---|
| Lookahead in features | Feature uses data from forward window | Strict: features from `df[:window_start+lookback]` only |
| Random split on time series | `train_test_split(shuffle=True)` | Use `TimeSeriesSplit` only |
| Too few windows | <80 rows in labeled_data | Check date range — need 10yr minimum |
| Extreme label skew | One strategy always 1 or always 0 | Check backtest logic — likely a bug |
| Uncalibrated probabilities | All probs near 0.9 or 0.1 | Apply `CalibratedClassifierCV` |
| Scaler mismatch at inference | Train scaler on train set, apply different at inference | Save + load same scaler object |
| Overconfident recommendation | Show 95% probability with 100 training samples | Apply fallback threshold at 0.55 |