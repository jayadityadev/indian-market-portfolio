# Indian Market Portfolio Intelligence Frontend

Next.js 16 / React 19 dashboard for NIFTY 50 regime-aware strategy research.

## Run

Start backend from project root:

```bash
uv run uvicorn api.main:app --app-dir src --port 8000
```

Start frontend from `frontend/`:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Pages

- `/`: Beginner and Professional analysis dashboard.
- `/regime`: Causal HMM regime timeline.
- `/benchmark`: Canonical-dataset academic XGBoost/LSTM comparison.
- `/report`: Provider-aware AI analyst report.

Frontend consumes FastAPI `/api/v1/*` routes. ML recommendations display explicit historical fallback while model validation remains below promotion gate.

Production check: `npm run build`.
