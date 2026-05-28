# Architectural Handoff: Indian Market Intelligence

This document provides a comprehensive overview of the **Indian Market Intelligence** platform to facilitate project continuation in a new environment.

## 1. Project Purpose
An ML-augmented strategy evaluation platform for Indian equities. It combines traditional quantitative backtesting with unsupervised market regime detection (KMeans) and Random Forest-based strategy recommendations.

## 2. Current Project Structure

### Root Directory
- `src/`: Backend source code (Python).
- `frontend/`: Next.js application (TypeScript/Tailwind/CSS).
- `data/`: Persistence layer (Parquet files).
- `models/`: Serialized ML models and scalers.
- `tests/`: Pytest suite for core logic.
- `.agents/`: Agent workflows and specialized skill instructions.

### Backend (`src/`)
- `api/`: FastAPI implementation (`main.py`, `schemas.py`, and domain routes).
- `backtester.py`: Core logic for portfolio simulation and performance metrics.
- `data_pipeline.py`: Data ingestion via `yfinance` and feature engineering.
- `regime_detector.py`: KMeans clustering for Market Regime classification.
- `strategies.py`: Implementation of 6 core trading strategies (RSI, MA, etc.).
- `risk_forecaster.py`: Bootstrapped drawdown forecasting.
- `classifier_*.py`: ML models for strategy suitability scoring.

### Frontend (`frontend/src/app/`)
- `page.tsx`: Main dashboard entry point and state management.
- `components/`: Modular UI elements (`MagicBento`, `BeginnerView`, `ProView`, `ThemeToggle`).
- `globals.css`: Core design system and "Bento Box" styling.

---

## 3. Implemented Systems

### Backtesting Engine
- Supports multiple technical strategies.
- Calculates CAGR, Sharpe Ratio, Max Drawdown, and Volatility.
- Generates precise equity curves and signal data.

### Market Regime Detection
- Uses KMeans ($k=3$) to cluster market states: **Bull**, **Bear**, and **Sideways**.
- Features are derived from price momentum and volatility.

### Strategy Recommendation (ML)
- A Random Forest classifier evaluates which strategy is most "suitable" for the detected market regime based on historical performance probabilities.

---

## 4. Architectural Decisions

1. **Decoupled Architecture**: 
   - Backend is a stateless FastAPI server.
   - Frontend is a Next.js (App Router) client communicating via REST.
2. **Persistence**:
   - Uses **Parquet** files for high-speed local data storage instead of a traditional SQL database.
3. **Design System**:
   - Transitioning to a **Bento Box** aesthetic (high border-radius, soft shadows, glassmorphism).
   - "Vanilla" CSS in `globals.css` preferred for complex animations (GSAP).
4. **Dependency Management**:
   - **Backend**: `uv` (Astral) for lightning-fast Python package management.
   - **Frontend**: `npm` for React/Next.js ecosystem.

---

## 5. API Contracts

| Method | Endpoint | Payload | Returns |
|--------|----------|---------|---------|
| `POST` | `/backtest` | `{ticker, start, end, strategy}` | Equity curve + performance metrics |
| `GET` | `/regime` | N/A | Current market regime label |
| `GET` | `/recommend` | N/A | ML confidence scores per strategy |
| `GET` | `/health` | N/A | API status |

---

## 6. Important Dependencies

### Backend (Python)
- `fastapi` / `uvicorn`: API Layer.
- `yfinance`: Market data source.
- `pandas` / `numpy`: Data manipulation.
- `scikit-learn`: KMeans and Random Forest models.
- `pydantic`: Schema validation.

### Frontend (React/Next.js)
- `lucide-react`: Icons.
- `lightweight-charts`: High-performance TradingView-like visualizations.
- `gsap`: Advanced UI animations.
- `recharts`: Simple statistical charts.

---

## 7. Pending Tasks & Roadmap

### High Priority (Current Focus)
- [ ] **Next.js UI Overhaul**: Complete frontend redesign in `frontend/src/app/page.tsx` and `frontend/src/app/components/` while keeping existing backend contract unchanged.
- [ ] **Frontend Integration Hardening**: Ensure `frontend/src/lib/api.ts` is aligned with FastAPI response models and has robust loading/error states.
- [ ] **Advanced Tab**: Implement Strategy Sandbox UX and backend parameter plumbing.

### Long Term
- [ ] **Multi-Asset Support**: Expand from NIFTY 50 index to individual stocks and multi-asset portfolios.
- [ ] **Real-time data integration** (live feed).
- [ ] **Transaction cost and slippage modeling** in `src/backtester.py`.
- [ ] **Advanced regime detection** using HMM (Hidden Markov Models).

---

## 8. Known Issues & Risks

- **Risky Files**: `src/backtester.py` contains complex vectorised math; modify with caution.
- **Data Quality**: `yfinance` can be throttled or return NaNs; retry logic is implemented but not foolproof.
- **ML Limitations**: The classifier is trained on a relatively small set of historical regimes; confidence scores should be treated as directional, not absolute.
- **Regime Labeling**: Unsupervised KMeans labels (0, 1, 2) are mapped manually to Bull/Bear/Sideways; this mapping may shift if the dataset changes significantly.

## 9. Current Status Snapshot (2026-05-10)

- **Backend Core**: Operational. Data pipeline, strategy engine, regime detector, classifier inference, and risk forecaster are implemented.
- **API Layer**: Operational. FastAPI app and primary routes are in place (`/health`, `/backtest`, `/regime`, `/recommend`).
- **Testing**: Core pytest suite exists and currently reports green in project TODO tracking.
- **Frontend State**: Functional but visual/UX redesign still in progress (see `implementation-plan.md`).
- **Primary Delivery Gap**: Product polish and advanced UX flows, not core quant engine correctness.

---

## 10. Detailed Execution Flow

1. **User Request**: User selects Ticker, Strategy, and Date Range in the Next.js Sidebar and clicks "Run Analysis".
2. **Frontend State**: `page.tsx` triggers `runAnalysis()`, which calls the FastAPI `/backtest` endpoint.
3. **Data Fetching**: `src/data_pipeline.py` uses `yfinance` to download historical OHLC data, which is cached in `data/nifty50.parquet`.
4. **Logic Execution**:
   - `backtester.py` simulates trades based on the selected logic in `strategies.py`.
   - `regime_detector.py` uses pre-trained KMeans to label each day's market state.
   - `risk_forecaster.py` runs 1000+ bootstrap simulations to estimate downside risk.
5. **ML Ranking**: `classifier_inference.py` predicts the "Ideal Strategy" for the current environment.
6. **Visualization**: The resulting JSON is parsed by the frontend and rendered using `Lightweight Charts` (Equity Curve) and `MagicBento` (Metrics).

## 11. Database & Persistence
While there is no SQL server, the system treats the `data/` folder as a structured store:
- **`nifty50.parquet`**: Master OHLC + Technical Indicator features.
- **`labeled_data.parquet`**: Strategy performance metrics grouped by regime windows (used for training).
- **`models/*.pkl`**: Serialized state of the analytical models.

## 12. Assumptions & Constraints
- **Stationarity**: Risk forecasting assumes that future volatility patterns will resemble historical patterns within the same regime.
- **Liquidity**: Assumes all trades can be executed at the close price with zero slippage (no transaction costs modeled yet).
- **Index Only**: Current models are optimized for NIFTY 50 and may not generalize well to small-cap stocks without retraining.
- **Offline Models**: ML models do not "learn" from user interactions; they require manual retraining scripts (`train_classifier.py`) to update.

## 13. Execution Backlog (Next 2 Sprints)

### Sprint A — Frontend Completion
1. Finalize dashboard layout and card hierarchy in `frontend/src/app/page.tsx`.
2. Consolidate reusable widgets under `frontend/src/app/components/` (metrics, charts, recommendation cards).
3. Add explicit loading/error/empty states in `frontend/src/lib/api.ts` consumers.
4. Validate responsive behavior (desktop + mobile) and preserve existing API contract.

### Sprint B — Quant Realism + Expansion
1. Add optional transaction-cost/slippage parameters to `src/backtester.py` with default off-switch for backward compatibility.
2. Extend ticker universe handling in `src/data_pipeline.py` and API schemas for multi-asset input.
3. Add benchmark tests for regime-label stability in `src/regime_detector.py`.
4. Expand tests in `tests/` for new backtest realism parameters and API schema compatibility.
