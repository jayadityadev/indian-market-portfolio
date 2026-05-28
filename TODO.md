# Project TODO — derived from docs/roadmap.md

Summary: Phases 0–6 complete. All modules built. FastAPI backend live.

## Core Phases

- [x] Phase 0 — Foundation (data pipeline)
	- Files: `src/data_pipeline.py`
	- Deliverables: clean OHLC fetch, feature engineering, parquet output, retry logic

- [x] Phase 1 — Backtesting Engine
	- Files: `src/backtester.py`
	- Deliverables: portfolio simulator, metrics (CAGR, Sharpe, MaxDD, Vol)

- [x] Phase 2 — Strategy Library (6 strategies)
	- Files: `src/strategies.py`
	- Deliverables: Buy & Hold, MA Crossover, RSI, Momentum, Bollinger Bands, Dual Momentum

- [x] Phase 3 — Regime Detection
	- Files: `src/regime_detector.py`
	- Deliverables: KMeans regimes, per-date labels, current regime detector

- [x] Phase 4 — Dashboard (Streamlit)
	- Files: `src/app.py`
	- Deliverables: Full UI with equity curves, metrics, regime timeline, heatmap, ML recommendation, risk forecast

- [x] Phase 5 — Integration + Polish
	- Deliverables: Edge-case handling, yfinance retries, short-range warnings, unit tests (20 passing)

- [x] Phase 6 — Demo + Presentation
	- Deliverables: `demo.py` generates `docs/demo_report.md`, README updated

## Modules

- [x] Module A — Strategy Suitability Classifier
	- Files: `src/classifier_features.py`, `src/classifier_inference.py`, `src/label_generator.py`, `src/train_classifier.py`, `models/`

- [x] Module B — Risk Forecasting
	- Files: `src/risk_forecaster.py`
	- Deliverables: bootstrap drawdown bands, percentile-based risk estimates

- [ ] Module C — Multi-Asset / Portfolio (deferred)
	- Deliverables: expand from index → individual stocks, portfolio allocation

- [x] Module D — FastAPI Backend
	- Files: `src/api/main.py`, `src/api/schemas.py`, `src/api/routes/`
	- Deliverables: `/backtest`, `/regime`, `/recommend` endpoints

## Next Sprints (from architecture_handoff.md § 13)

### Sprint A — Frontend Completion
- [ ] A1: Finalize dashboard layout and card hierarchy in `frontend/src/app/page.tsx`
- [ ] A2: Consolidate reusable widgets in `frontend/src/app/components/` (metrics cards, charts, recommendation)
- [ ] A3: Add explicit loading/error/empty states in `frontend/src/lib/api.ts` consumers
- [ ] A4: Validate responsive behavior (desktop + mobile) and preserve API contract

### Sprint B — Quant Realism + Expansion
- [ ] B1: Add optional transaction-cost/slippage parameters to `src/backtester.py` (default off-switch)
- [ ] B2: Extend ticker universe handling in `src/data_pipeline.py` and API schemas for multi-asset
- [ ] B3: Add regime-label stability benchmarks in `src/regime_detector.py` tests
- [ ] B4: Expand `tests/` coverage for backtest realism params and API schema compatibility

## Notes
- Source: `docs/roadmap.md` (current status + phased plan).
- Tests: `uv run pytest tests/ -v` (20 tests passing)
- API: `uv run uvicorn api.main:app --app-dir src`
- Demo: `uv run python demo.py`
