# Original User Request

## Initial Request — 2026-08-16T04:59:17Z

Refactor the Indian Market Portfolio Intelligence platform into a production-ready, regime-aware quantitative trading and strategy evaluation system for Indian equities, featuring Hidden Markov Model (HMM) regime detection, an XGBoost strategy classifier, an LSTM-DNN academic benchmark, Neon PostgreSQL persistence, and an LLM-powered AI Market Analyst.

Working directory: /home/jayaditya/Projects/MajorProject/indian-market-portfolio
Integrity mode: development

## Requirements

### R1. Core ML Stack Upgrade & Academic Benchmark
Upgrade the quantitative machine learning pipeline:
- Replace K-Means with Gaussian Hidden Markov Models (HMM) to capture temporal regime transitions (Bull, Bear, Sideways).
- Upgrade the strategy recommendation classifier from Random Forest to XGBoost with probability outputs across the 6 strategies.
- Implement an LSTM-DNN model (as specified in the academic base paper) and an evaluation pipeline comparing its performance and overfitting profile against the XGBoost classifier.

### R2. Database Persistence Layer (Neon PostgreSQL)
Introduce a relational database layer using SQLAlchemy:
- Connect to Neon PostgreSQL using `DATABASE_URL` from the `.env` file (with SQLite fallback if unset for testing).
- Maintain schemas/models for backtest execution logs, detected market regime history, and model benchmark comparisons.
- Ensure analysis and backtest endpoints record execution records asynchronously.

### R3. AI Market Analyst Report (LLM Integration)
Implement an automated financial commentary layer:
- Build a RESTful endpoint `POST /api/v1/llm-report` that accepts backtest metrics, market regime breakdown, and risk estimates.
- Support free API providers (Gemini via `google-genai`, Groq, NVIDIA NIM, OpenRouter) configured via `.env`.
- Output structured Markdown market analysis explaining the regime context and justifying strategy selection.

### R4. REST API & Next.js Frontend Integration
Unify and expose the platform:
- Structure all FastAPI endpoints under standard RESTful `/api/v1` routes (`/analyze`, `/regime`, `/recommend`, `/backtest`, `/llm-report`, `/benchmark`).
- Update the primary Next.js React dashboard (`frontend/`) to incorporate the new HMM regime timeline, Model Benchmark comparison view (LSTM vs XGBoost), and the interactive AI Market Analyst report viewer (with `src/app.py` Streamlit kept synchronized as a Python demo).

## Acceptance Criteria

### Machine Learning & Regimes
- [ ] Gaussian HMM model fits on historical NIFTY 50 OHLCV data and assigns 3 distinct market regimes.
- [ ] XGBoost classifier outputs calibrated suitability probabilities for the 6 strategies.
- [ ] LSTM-DNN model trains on sequences and outputs evaluation metrics alongside XGBoost in `GET /api/v1/benchmark`.

### Database & Persistence
- [ ] SQLAlchemy database engine connects to Neon PostgreSQL without connection leaks.
- [ ] Executing `POST /api/v1/analyze` stores run metadata and regime snapshots in PostgreSQL.

### LLM Analyst & REST API
- [ ] `POST /api/v1/llm-report` returns a structured Markdown explanation when supplied with analysis outputs.
- [ ] All FastAPI endpoints respond with valid JSON schemas under `/api/v1/*`.
- [ ] Automated tests in `pytest tests/` pass cleanly.
- [ ] Next.js frontend (`frontend/`) builds cleanly (`npm run build`) and renders the updated Beginner/Pro views, Model Benchmark, and AI Analyst Report.
