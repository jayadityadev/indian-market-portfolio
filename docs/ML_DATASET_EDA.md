# ML Dataset EDA

Date: 2026-08-16

## Data Integrity

- `4,579` NIFTY 50 rows from `2007-12-12` through `2026-08-14`.
- No duplicate dates, missing values, or OHLC consistency violations.
- `1,273` rows have zero volume; volume is not used as a trusted predictive feature.
- Daily return annualized volatility: `20.4%`.
- Return behavior varies materially by year and regime; regime conditioning remains necessary.

## Target Findings

The recommender target selects strategy with maximum forward 63-day Sharpe over 305 windows.

Corrected global-signal target counts:

| Strategy | Winners |
|---|---:|
| Buy & Hold | 107 |
| RSI | 83 |
| Bollinger Bands | 90 |
| MA Crossover | 14 |
| Dual Momentum | 7 |
| Momentum | 4 |

`101/305` windows have winner margin below `0.05` Sharpe. Hard winner labels are therefore noisy, especially for low-frequency winners.

Earlier recommender target generation recomputed strategies on the 63-day forward slice. That erased 126/252-day lookbacks and made Momentum/Dual Momentum impossible winners. It now computes signals on the full historical frame, then evaluates only the future slice.

## Feature Findings

No feature missingness found. Highest univariate mutual information against winner target:

- `max_drawdown_252d`
- `sma_ratio_50_200`
- `regime_persistence`
- `high_low_ratio_20d`
- `regime_switch_count_252d`

Feature scores are exploratory only; they are not promotion evidence.

## Validation Findings

Purging and embargo were previously configured in trading-day units while the splitter operates on training-row units. With 14-day label spacing, `63` rows over-purged folds. Corrected default configuration is five purge samples and one embargo sample, matching approximately 63 trading days and 14 trading days.

Corrected candidate XGBoost CV Macro F1: `0.161`. Promotion gate remains `0.300`.

Suitability-ensemble candidate CV Macro F1: `0.185`, mean ROC AUC: `0.542`. It also remains below promotion quality.

## Decision

Suitability formulation has been implemented as a candidate mode, but it also fails validation. ML is frozen behind historical fallback while the platform is completed.
