# Domain Model

## Product Boundary

Indian Market Portfolio Intelligence is beginner-first decision support for choosing among predefined strategies using Indian market risk context. Current scope is NIFTY 50 daily data and a 63-trading-day recommendation horizon. Portfolio optimization and order execution remain future scope.

## Actors

- Beginner investor: consumes plain-language risk and strategy explanations.
- Researcher: inspects historical results, assumptions, and validation evidence.
- Data provider: supplies market prices and news articles.
- Model provider: supplies optional language-model summaries only.

## Core Entities

| Entity | Meaning | Required identity |
|---|---|---|
| `MarketDataSnapshot` | Canonical OHLCV observations used by every analysis | ticker, start/end dates, source, fetched-at |
| `FeatureSnapshot` | Features derived only from data available at each observation | ticker, observation date, feature version |
| `RegimeAssessment` | Bull, Bear, or Sideways state with probabilities | ticker, as-of date, model version |
| `StrategyDefinition` | Immutable rule for one predefined strategy | strategy name, rule version |
| `BacktestRun` | Historical replay of strategy signals and execution assumptions | run id, data snapshot, strategy, cost model |
| `RiskForecast` | Forward drawdown distribution estimate | run id, horizon, method, seed/version |
| `StrategyRecommendation` | Ranked strategy suitability result | as-of date, horizon, source, confidence |
| `NewsArticle` | Retrieved external article with provenance | URL, publisher, published-at, fetched-at |
| `NewsEvent` | Deduplicated market-relevant group of articles | event id, topic, affected market |
| `NewsSummary` | Cited plain-language summary of one or more events | event id, source URLs, generated-at |

## Domain Events

- `MarketDataFetched`
- `FeaturesBuilt`
- `RegimeAssessmentCreated`
- `BacktestCompleted`
- `RiskForecastCreated`
- `StrategyRecommendationCreated`
- `NewsArticlesFetched`
- `NewsEventGrouped`
- `NewsSummaryCreated`

## Invariants

- Every analysis identifies ticker, date range, data source, and freshness.
- Feature and regime values for date `t` cannot use observations after `t`.
- Recommendations expose whether source is validated ML, historical fallback, or unavailable.
- News claims link to retrieved source URLs and show article age.
- Cached news is labelled stale; unavailable data is never replaced with invented content.
- News provides context and warnings; it does not silently alter quantitative probabilities.
- A recommendation cannot be presented as research-grade unless its validation status passes data and walk-forward contracts.

## User Flow

1. User selects market and analysis period.
2. System loads or fetches canonical market data.
3. System validates data freshness and integrity.
4. System detects regime and evaluates six predefined strategies.
5. System estimates risk over 63 trading days.
6. System ranks strategies and explains recommendation source.
7. System fetches India/global market news, caches it, and displays cited context separately.
