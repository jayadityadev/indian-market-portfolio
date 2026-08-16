# Indian Market Portfolio Intelligence

> Regime-aware quantitative trading intelligence for Indian equity markets — powered by Gaussian HMM, XGBoost, LSTM-DNN, and a multi-provider LLM analyst.

---

## What It Is

A **research-oriented strategy evaluation platform** that combines:

- **Gaussian HMM** regime detection (Bull / Bear / Sideways) replacing the old KMeans approach
- **XGBoost vs LSTM-DNN comparative benchmark**; LSTM remains academic and ML recommendations are validation-gated
- **6 backtested trading strategies** with realistic slippage and commission modeling
- **Multi-provider LLM Market Analyst** (Gemini → Groq → NVIDIA NIM → OpenRouter → Mock waterfall)
- **Next.js 16 + React 19 frontend** with Beginner and Professional dashboard modes
- **FastAPI `/api/v1/` backend** with Neon PostgreSQL persistence (SQLite fallback and automatic table initialization)
- **Automated unit, integration, adversarial, E2E, and runtime smoke tests** covering API and persistence contracts

---

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install Python deps
git clone <repo>
cd indian-market-portfolio
uv sync

# 3. Set up environment variables
cp .env.example .env          # fill in your API keys

# 4. Start the FastAPI backend
uv run uvicorn api.main:app --app-dir src --port 8000 --reload

# 5. Start the Next.js frontend (separate terminal)
cd frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| API health | http://localhost:8000/health |

---

## Environment Variables (`.env`)

```ini
GEMINI_API_KEY=...
GROQ_API_KEY=...
NVIDIA_NIM_API_KEY=...
OPENROUTER_API_KEY=...
DATABASE_URL=postgresql://...@.../neondb?sslmode=require   # Neon Postgres
```

All LLM keys are optional — the analyst waterfall falls back to a deterministic offline mock when no keys are configured.

---

## API Endpoints (`/api/v1/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/regime` | Current HMM regime + historical distribution |
| `GET` | `/api/v1/recommend` | Validation-gated ML scores or explicit historical fallback |
| `POST` | `/api/v1/backtest` | Multi-strategy backtest with slippage/commission |
| `POST` | `/api/v1/analyze` | Full pipeline — regime + backtest + curves + heatmap |
| `GET` | `/api/v1/benchmark` | XGBoost vs LSTM-DNN performance comparison |
| `POST` | `/api/v1/llm-report` | AI market analyst report (multi-provider waterfall) |
| `GET` | `/api/v1/news` | Live India/global market news with cached fallback and source links |

### `POST /api/v1/analyze` — example request

```json
{
  "ticker": "^NSEI",
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "strategy": "all",
  "initial_investment": 100000
}
```

### `POST /api/v1/llm-report` — pinning a provider

```bash
curl -X POST "http://localhost:8000/api/v1/llm-report?provider=nvidia:meta/llama-3.3-70b-instruct" \
  -H "Content-Type: application/json" \
  -d '{"current_regime": "Bull", "recommended_strategy": "Momentum"}'
```

LLM provider cascade: **Gemini → Groq → NVIDIA NIM → OpenRouter → Mock**

---

## Architecture

```
yfinance OHLCV
       │
       ▼
Data Pipeline ─────────────────────────────────────────────────────────┐
  └─ feature engineering (RSI, MACD, ADX, volatility, rolling returns)  │
       │                                                                 │
       ▼                                                                 │
Regime Engine (Gaussian HMM, 3 states)                                  │
  └─ Bull / Bear / Sideways labels                                      │
       │                                                                 │
       ▼                                                                 │
Strategy Engine (6 strategies)    ML Recommender (XGBoost + fallback)    │
  └─ signals → BacktestEngine      └─ regime-conditional probabilities  │
       │                                      │                         │
       ▼                                      │                         │
ML Benchmark Pipeline                         │                         │
  ├─ XGBoost classifier                       │                         │
  └─ LSTM-DNN (PyTorch)                       │                         │
       │                                      │                         │
       ▼                                      ▼                         │
Risk Forecaster (bootstrap MC)    LLM Analyst (provider waterfall)      │
  └─ drawdown bands                └─ Gemini / Groq / NVIDIA / OR / Mock│
       │                                      │                         │
       └──────────────────────────────────────┘                         │
                          │                                             │
                          ▼                                             │
                FastAPI /api/v1/                                        │
                          │                                             │
                          ▼                                             │
               Neon PostgreSQL (persistence)                            │
               ├─ BacktestHistory                                       │
               ├─ MarketRegimeLog                                       │
               └─ ModelBenchmarkResults                                 │
                          │                                             │
                          ▼                                             │
              Next.js 16 + React 19 Frontend ◄─────────────────────────┘
              ├─ / (Home — Beginner / Pro mode)
              ├─ /regime (HMM timeline view)
              ├─ /benchmark (XGBoost vs LSTM)
              └─ /report (AI Analyst with provider picker)
```

---

## Frontend Pages

| Route | Description |
|---|---|
| `/` | Main dashboard — Beginner and Professional modes with full analysis |
| `/regime` | HMM regime timeline and historical distribution |
| `/benchmark` | Side-by-side XGBoost vs LSTM-DNN metric comparison |
| `/report` | AI Market Analyst report with provider selector |

---

## Strategy Library

| Strategy | Logic | Best Regime |
|----------|-------|-------------|
| Buy & Hold | Always long (benchmark) | Bull |
| MA Crossover | 50-day SMA > 200-day SMA | Trending |
| RSI | Buy oversold (<30), sell overbought (>70) | Sideways |
| Momentum | Long when 12-month return positive | Bull |
| Bollinger Bands | Buy below lower band, sell above upper | Sideways |
| Dual Momentum | Long when both 12m and 6m momentum positive | Strong trends |

---

## Project Structure

```
src/
├── api/
│   ├── main.py              # FastAPI app + CORS + /api/v1 router mounting
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── analyze.py       # POST /api/v1/analyze — full pipeline
│       ├── backtest.py      # POST /api/v1/backtest
│       ├── regime.py        # GET  /api/v1/regime
│       ├── recommend.py     # GET  /api/v1/recommend
│       ├── benchmark.py     # GET  /api/v1/benchmark (XGBoost vs LSTM)
│       └── llm_report.py    # POST /api/v1/llm-report (LLM waterfall)
├── llm/
│   ├── client.py            # LLMClient waterfall orchestrator
│   ├── providers.py         # Gemini / Groq / NVIDIA NIM / OpenRouter / Mock
│   ├── analyst.py           # generate_analyst_report() high-level API
│   └── prompts.py           # 5-section institutional markdown prompt builder
├── db/
│   ├── models.py            # SQLAlchemy 2.0 ORM models
│   └── session.py           # Neon Postgres + SQLite fallback connection
├── models/
│   ├── regime_detector.py   # Gaussian HMM regime detection and causal artifacts
│   ├── recommender.py       # XGBoost strategy classifier and suitability candidate
│   └── lstm_benchmark.py    # LSTM-DNN academic benchmark model (PyTorch)
├── data_pipeline.py         # yfinance fetch + 20+ feature engineering
├── backtester.py            # Execution engine (slippage, commission, lag)
├── strategies.py            # 6 trading strategy implementations
├── regime_detector.py       # Regime detection (wraps HMM)
├── classifier_*.py          # ML training pipeline
└── risk_forecaster.py       # Bootstrap Monte Carlo drawdown simulation

frontend/
├── src/app/
│   ├── page.tsx             # Home — Beginner / Pro mode
│   ├── regime/page.tsx      # HMM regime timeline
│   ├── benchmark/page.tsx   # XGBoost vs LSTM comparison
│   ├── report/page.tsx      # AI Analyst report + provider picker
│   └── components/
│       ├── NavBar.tsx
│       ├── BeginnerView.tsx
│       ├── ProView.tsx
│       ├── CandlestickChart.tsx
│       ├── MagicBento.tsx
│       └── StrategyLibrary.tsx
└── src/lib/
    ├── api.ts               # fetchAnalysis() → POST /api/v1/analyze
    ├── benchmark.ts         # fetchBenchmark() → GET /api/v1/benchmark
    ├── regime.ts            # fetchRegime() → GET /api/v1/regime
    └── report.ts            # fetchReport() → POST /api/v1/llm-report

tests/
├── conftest.py              # Shared fixtures (synthetic OHLCV, mock LLM, DB)
├── test_e2e_v1_routes.py    # 50 E2E tests for all /api/v1/* endpoints (NEW)
├── test_e2e_api_llm.py      # LLM waterfall + API contract tests
├── test_e2e_ml_db.py        # ML pipeline + persistence integration tests
├── test_db.py               # Database model + query tests
├── test_llm.py              # LLM provider unit tests
└── ...                      # Adversarial challenger suites

data/
├── nifty50.parquet          # OHLCV + engineered features
├── nifty50_regimes.parquet  # HMM regime labels
├── labeled_data.parquet     # Classifier training data
└── benchmark_results.json   # XGBoost vs LSTM benchmark output
```

---

## Tests

```bash
# Run full suite
uv run pytest tests/ -v

# Run only new E2E route tests
uv run pytest tests/test_e2e_v1_routes.py -v
```

Test count changes as contracts evolve; run suite locally for current status.

---

## Limitations

- Historical daily data only (no live market feed)
- `/benchmark` serves canonical-dataset academic benchmark output; it does not approve ML recommendations
- LLM analyst requires at least one API key (or runs in offline mock mode)
- Neon Postgres is preferred for persistence; SQLite fallback is intended for local development
- Strategy parameters are fixed; walk-forward ML validation exists, but no production ML candidate currently clears promotion gates
- News depends on live GDELT/RSS availability; cached articles are labelled stale
- Current scope is NIFTY 50 daily decision support; portfolio optimization, multi-asset support, and order execution are deferred
