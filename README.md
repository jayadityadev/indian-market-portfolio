# Indian Market Portfolio Intelligence

ML-augmented strategy evaluation platform for Indian equities.
Backtests 6 strategies (Buy & Hold, MA Crossover, RSI, Momentum, Bollinger Bands, Dual Momentum) with KMeans regime detection and ML-powered recommendations.

## Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo>
cd indian-market-intelligence
uv sync
```

## Run

### Streamlit Dashboard
```bash
uv run streamlit run src/app.py
```
Open http://localhost:8501

### FastAPI Backend
```bash
uv run uvicorn api.main:app --app-dir src --port 8000
```
API docs at http://localhost:8000/docs

### Headless Demo (generates report)
```bash
uv run python demo.py
```
Output: `docs/demo_report.md`

## Usage

1. Select NIFTY 50 and date range in sidebar
2. Choose strategy or "Auto" mode
3. Click "Run Analysis"
4. Review equity curve, metrics, regime analysis, recommendation, risk forecast

## Architecture

```
User Input (Streamlit / FastAPI)
         ↓
Data Pipeline (yfinance → OHLC → feature engineering)
         ↓
Backtesting Engine (signals → portfolio metrics + equity curve)
         ↓
Regime Detection (KMeans k=3 → Bull / Bear / Sideways)
         ↓
Strategy Classifier (Random Forest → probability scores)
         ↓
Risk Forecaster (bootstrap → drawdown bands)
         ↓
Dashboard / API Response
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/backtest` | Run backtest for ticker/dates/strategy |
| GET | `/regime` | Current regime + distribution |
| GET | `/recommend` | Strategy recommendation with ML probabilities |

## Strategy Library

| Strategy | Logic | Best Regime |
|----------|-------|-------------|
| Buy & Hold | Always long (benchmark) | Bull |
| MA Crossover | 50-day SMA > 200-day SMA | Trending |
| RSI | Buy oversold (<30), sell overbought (>70) | Sideways |
| Momentum | Long when 12-month return positive | Bull |
| Bollinger Bands | Buy below lower band, sell above upper | Sideways |
| Dual Momentum | Long when both 12m and 6m momentum positive | Strong trends |

## Project Structure

```
src/
├── app.py              # Streamlit dashboard
├── data_pipeline.py    # yfinance fetch + feature engineering
├── backtester.py       # Backtesting engine
├── strategies.py       # 6 trading strategies
├── regime_detector.py  # KMeans regime detection
├── classifier_*.py     # ML strategy classifier
├── risk_forecaster.py  # Bootstrap drawdown estimation
└── api/                # FastAPI backend
    ├── main.py         # App + CORS + routes
    ├── schemas.py      # Pydantic models
    └── routes/         # /backtest, /regime, /recommend

tests/
└── test_core.py        # 20 unit tests

data/
├── nifty50.parquet           # OHLC + features
├── nifty50_regimes.parquet   # Regime labels
└── labeled_data.parquet      # Classifier training data

models/
├── *_classifier.pkl    # Trained Random Forest models
├── *_scaler.pkl        # Feature scalers
└── regime_model.pkl    # KMeans regime model
```

## Tests

```bash
uv run pytest tests/ -v
```

## Known Limitations

- Historical data only (no live feed)
- NIFTY 50 index only (no individual stocks in MVP)
- No transaction costs modeled
- Strategy parameters fixed (no optimization)
- Regime detection is unsupervised — Bear cluster may be small
- ML classifier trained on ~100 windows — low confidence expected
- Risk forecasting assumes stationarity within regimes
