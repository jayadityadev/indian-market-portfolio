# Project: Indian Market Portfolio Intelligence Platform Refactor

## Architecture
A research-oriented, regime-aware quantitative trading and strategy evaluation system for Indian equities.
- **Data Ingestion & Feature Engineering** (`src/data_pipeline.py`, `src/classifier_features.py`): Canonical NIFTY 50 OHLCV pipeline, validation contracts, causal window features, and derived parquet artifacts.
- **Machine Learning & Regimes** (`src/models/`, `src/regime_detector.py`, `src/recommender.py`, `src/models/lstm_benchmark.py`):
  - 3-State Gaussian HMM (Bull, Bear, Sideways) capturing temporal regime transitions, transition matrices, and state probabilities.
  - 6-strategy XGBoost recommendation classifier with validation-gated scores; current production state uses historical fallback.
  - PyTorch LSTM-DNN academic benchmark model (based on Alam et al., IEEE Access 2024) and evaluation comparison pipeline.
- **Database Persistence** (`src/db/`): SQLAlchemy 2.0 with Neon PostgreSQL (`DATABASE_URL`) and SQLite fallback, handling `BacktestLog`, `RegimeSnapshot`, `ModelBenchmarkRun`.
- **AI Market Analyst** (`src/llm/`): Multi-provider LLM client (Gemini `google-genai`, Groq, NVIDIA NIM, OpenRouter, Mock fallback) generating structured Markdown financial commentary.
- **API Layer** (`src/api/`): FastAPI RESTful `/api/v1/*` routes for analysis, regime, recommendations, backtests, benchmark, news, and LLM reports, with database persistence.
- **User Interfaces** (`frontend/`, `src/app.py`):
  - Next.js React 19 dashboard with HMM regime timeline, Model Benchmark comparison view, interactive AI Market Analyst report viewer.
  - Streamlit application (`src/app.py`) synchronized with backend upgrades.
- **Market News** (`src/news.py`, `src/api/routes/news.py`):
  - Free GDELT/RSS retrieval, provenance-preserving cache, and optional cited LLM summary.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dependencies & Config | Upgrade pyproject.toml (`hmmlearn`, `xgboost`, `torch`, `sqlalchemy`, `asyncpg`, `google-genai`, `groq`, etc.) | M1 | Survey |
| 2 | Gaussian HMM Regime Detection | 3-state HMM on NIFTY 50 OHLCV data with transition matrix and posteriors | M1 | R1 |
| 3 | XGBoost Strategy Classifier | 6-strategy classifier with validation-gated outputs and historical fallback | M1 | R1 |
| 4 | LSTM-DNN Academic Benchmark | PyTorch 2-LSTM + 4-Dense benchmark architecture and evaluation comparison pipeline | M1 | R1 |
| 5 | Database Connection & Engine | SQLAlchemy 2.0 with Neon PostgreSQL (`DATABASE_URL`) & SQLite fallback | M2 | R2 |
| 6 | Relational ORM Schemas | Models for backtest execution logs, regime history, model benchmarks | M2 | R2 |
| 7 | Async Execution Recording | Background recording of run metadata and analysis snapshots | M2 | R2 |
| 8 | Multi-Provider LLM Integration | Waterfall client: Gemini -> Groq -> NVIDIA NIM -> OpenRouter -> Mock | M3 | R3 |
| 9 | Structured Financial Commentary | Prompt generator & Markdown analyst report generation | M3 | R3 |
| 10 | RESTful API Routes | FastAPI `/api/v1/*` (`/analyze`, `/regime`, `/recommend`, `/backtest`, `/llm-report`, `/benchmark`) | M4 | R4 |
| 11 | Next.js Frontend Dashboard | Regime timeline, Benchmark comparison view, AI Analyst viewer, React 19 fixes | M4 | R4 |
| 12 | Streamlit App Synchronization | Synchronize `src/app.py` with HMM, XGBoost, Benchmark, and Analyst features | M4 | R4 |
| 13 | Comprehensive E2E Testing Suite | Tiers 1-4 test suite, offline fixtures, test runner publishing `TEST_READY.md` | E2E-Track | Dual Track |
| 14 | Final Verification & Hardening | Runtime smoke checks, persistence ingestion, documentation, and adversarial coverage | M5 | Current |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core ML Stack & Benchmark | Gaussian HMM, XGBoost, LSTM-DNN, benchmark comparison pipeline, pyproject dependencies | None | IMPLEMENTED; ML PROMOTION BLOCKED |
| M2 | Database Persistence Layer | SQLAlchemy engine, schemas, SQLite fallback, async background recording | None | DONE |
| M3 | AI Market Analyst Report | LLM client waterfall (Gemini/Groq/NVIDIA/OpenRouter/Mock), Markdown prompt & generator | None | DONE |
| M4 | REST API & Frontend Integration | FastAPI routes, Next.js dashboard, Streamlit compatibility | M1, M2, M3 | DONE |
| E2E | E2E Testing Track | Unit, integration, adversarial, API, database, and workload suites | None | DONE |
| M5 | Final Verification & Hardening | Runtime demo verification, persistence ingestion, stale-doc cleanup, graph rebuild | M4, E2E | IN PROGRESS |

## Interface Contracts
### ML Engine ↔ REST API / DB
- `GaussianHMMRegimeDetector.fit_predict(df)` -> returns dict with `regimes` (Series), `regime_names` (list: Bull, Bear, Sideways), `transition_matrix` (3x3 array), `stationary_distribution` (array), `state_posteriors` (array).
- `XGBoostStrategyClassifier.predict_proba(features)` -> returns six strategy scores only when artifact validation passes; otherwise recommendation provider returns historical fallback with status/reason.
- `LSTMBenchmarkPipeline.run_comparison(features, targets)` -> returns dict with `xgboost_metrics` (acc, precision, recall, f1, log_loss), `lstm_metrics` (acc, precision, recall, f1, val_loss, overfitting_gap), `epochs_history`.

### Database Layer ↔ API Endpoints
- `get_db_session()` -> SQLAlchemy `Session` / `AsyncSession` context manager.
- `record_backtest_log(session, payload, metrics, run_id)` -> logs backtest execution.
- `record_regime_snapshot(session, symbol, regimes, transition_matrix)` -> logs regime history.
- `record_benchmark_run(session, comparison_data)` -> logs benchmark run.

### LLM Analyst ↔ API Endpoints
- `generate_llm_market_report(metrics: dict, regime_data: dict, risk_data: dict, provider_override: Optional[str] = None)` -> returns `{"report_markdown": str, "provider_used": str, "generated_at": str}`.

### FastAPI REST Endpoints (`/api/v1`)
- `POST /api/v1/analyze`: Returns technical indicators, portfolio metrics, regime detection, strategy recommendation, stored to DB.
- `GET /api/v1/regime`: Returns current and historical 3-state HMM regime timeline and transition probabilities.
- `GET /api/v1/recommend`: Returns validation-gated ML scores or explicit historical fallback.
- `POST /api/v1/backtest`: Executes strategy backtest, returns metrics, logs to DB.
- `POST /api/v1/llm-report`: Generates structured Markdown AI Market Analyst report.
- `GET /api/v1/benchmark`: Returns comparative evaluation between XGBoost and LSTM-DNN models.

## Code Layout
- `src/`
  - `data_pipeline.py`, `data_contracts.py` (market data and artifact validation)
  - `classifier_features.py` (recommendation features)
  - `models/` (`regime_detector.py`, `recommender.py`, `lstm_benchmark.py`)
  - `db/` (`connection.py`, `models.py`, `crud.py`)
  - `llm/` (`analyst.py`, `prompts.py`, `providers.py`)
  - `api/` (`v1/` routers, schemas, dependencies)
  - `main.py` (FastAPI app entrypoint)
  - `app.py` (Streamlit demo dashboard)
- `frontend/`
  - `src/` (Next.js React dashboard: pages, components, lib/api.ts)
- `tests/`
  - Unit, integration, and E2E test suites
