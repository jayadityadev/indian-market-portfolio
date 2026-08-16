"""Regime API route — GET /regime."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

# routes/ -> api/ -> src/
SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.schemas import RegimeResponse
from data_pipeline import load_data

import pandas as pd

router = APIRouter()
SUPPORTED_TICKER = "^NSEI"


@router.get("/regime", response_model=RegimeResponse)
def get_regime_endpoint(ticker: str = Query(default="^NSEI")):
    """Get current market regime and historical regime distribution."""
    if ticker != SUPPORTED_TICKER:
        raise HTTPException(
            status_code=400,
            detail=f"Current canonical dataset supports {SUPPORTED_TICKER} only.",
        )
    try:
        data_path = PROJECT_ROOT / "data" / "nifty50.parquet"
        df = load_data(data_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load failed: {exc}")

    # Distribution from regime parquet
    regime_path = PROJECT_ROOT / "data" / "nifty50_regimes.parquet"
    distribution: dict[str, int] = {}
    total = 0
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df.index = pd.to_datetime(regime_df.index)
        counts = regime_df["regime"].dropna().astype(str).value_counts()
        distribution = counts.to_dict()
        total = int(counts.sum())
        current = str(regime_df["regime"].dropna().iloc[-1])
        regime_source = "walk_forward_artifact"
    else:
        raise HTTPException(status_code=503, detail="Causal regime artifact is unavailable.")

    return RegimeResponse(
        current_regime=current,
        regime_distribution=distribution,
        total_days=total,
        regime_source=regime_source,
    )
