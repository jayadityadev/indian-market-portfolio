# P2 Retraining Report

Date: 2026-08-16
Ticker: `^NSEI`
Dataset: `2007-12-12` through `2026-08-14`

## Validation Fix

XGBoost walk-forward folds now fit `StandardScaler` only on each fold's training partition. Final production fitting still fits its scaler on the complete training dataset after validation.

## Staged Results

No staged artifact replaced production model files.

| Model | Metric | Result | Gate |
|---|---:|---:|---:|
| XGBoost corrected CV | Macro F1 | 0.180 | 0.300 |
| Best XGBoost tested variant | Macro F1 | 0.190 | 0.300 |
| LSTM corrected test split | Macro F1 | 0.233 | benchmark only |
| Cost-aware shallow XGBoost | Macro F1 | 0.250 | 0.300 |
| Best dense-window candidate | Macro F1 | 0.257 | 0.300 |

XGBoost variants tested: depth 2/75 estimators, depth 3/150, depth 4/150, and depth 3/250. None cleared promotion gate.

Best cost-aware candidate used `0.1%` commission, `0.1%` slippage, Sharpe utility, depth 2, and 100 estimators. Its CV accuracy was `0.344` versus a majority baseline of `0.372`; it does not qualify for promotion.

Locked holdout evaluation also showed the candidate below majority baseline, so the raw Macro F1 gate is not sufficient by itself.

The best dense-window result used 204 decision windows at 21-day spacing, 0.1% commission, 0.1% slippage, Sharpe utility, depth 2, and 100 estimators. Accuracy was `0.348` versus majority baseline `0.372`; it remains rejected.

## Decision

Keep production recommendation state as `historical_fallback`. Frozen production registry remains valid. LSTM remains academic benchmark only.

## Verification

- `536` tests passed.
- Frozen registry verification passed.
- Staged artifacts stored outside repository under `/tmp/opencode/p2-staging/`.
