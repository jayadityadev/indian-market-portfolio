"""FastAPI backend for Indian Market Portfolio Intelligence.

Run: uv run uvicorn api.main:app --reload --app-dir src
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ importable for all modules
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.routes.backtest import router as backtest_router
from api.routes.regime import router as regime_router
from api.routes.recommend import router as recommend_router
from api.routes.analyze import router as analyze_router

app = FastAPI(
    title="Indian Market Portfolio Intelligence API",
    description=(
        "ML-augmented strategy evaluation for Indian equities. "
        "Provides backtesting, regime detection, and strategy recommendations."
    ),
    version="0.1.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(backtest_router, tags=["Backtest"])
app.include_router(regime_router, tags=["Regime"])
app.include_router(recommend_router, tags=["Recommendation"])
app.include_router(analyze_router, tags=["Full Analysis"])


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
