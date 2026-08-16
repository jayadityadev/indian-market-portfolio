"""FastAPI backend for Indian Market Portfolio Intelligence.

Run: uv run uvicorn api.main:app --reload --app-dir src
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ importable for all modules
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from api.routes.backtest import router as backtest_router
from api.routes.regime import router as regime_router
from api.routes.recommend import router as recommend_router
from api.routes.analyze import router as analyze_router
from api.routes.benchmark import router as benchmark_router
from api.routes.llm_report import router as llm_report_router
from api.routes.news import router as news_router
from db.connection import check_connection_health
from db.service import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize persistence before serving requests; keep API usable if DB degrades."""
    try:
        init_db()
    except Exception:
        logger.exception("Database initialization failed; API will run in degraded mode.")
    yield

app = FastAPI(
    title="Indian Market Portfolio Intelligence API",
    description=(
        "ML-augmented strategy evaluation for Indian equities. "
        "Provides backtesting, regime detection, and strategy recommendations."
    ),
    version="0.1.0",
    lifespan=lifespan,
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
app.include_router(backtest_router, prefix="/api/v1", tags=["Backtest"])
app.include_router(regime_router, prefix="/api/v1", tags=["Regime"])
app.include_router(recommend_router, prefix="/api/v1", tags=["Recommendation"])
app.include_router(analyze_router, prefix="/api/v1", tags=["Full Analysis"])
app.include_router(benchmark_router, prefix="/api/v1", tags=["Benchmark"])
app.include_router(llm_report_router, prefix="/api/v1", tags=["LLM Report"])
app.include_router(news_router, prefix="/api/v1", tags=["News"])


@app.get("/health")
def health_check():
    """Report API and database health without exposing credentials."""
    database = check_connection_health()
    return {
        "status": "ok" if database["status"] == "healthy" else "degraded",
        "version": "0.1.0",
        "database": database,
    }
