# src/api/routes/llm_report.py
"""LLM Market Analyst endpoint — POST /api/v1/llm-report.

Delegates to the full multi-provider waterfall LLMClient (llm/client.py) which
cascades: Gemini → Groq → NVIDIA NIM → OpenRouter → Mock (offline fallback).

All API keys are read from .env:
  GEMINI_API_KEY, GROQ_API_KEY, NVIDIA_NIM_API_KEY, OPENROUTER_API_KEY

An optional ``provider`` query param lets the caller pin a specific provider
(e.g. ?provider=nvidia or ?provider=groq:llama-3.3-70b-versatile).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.analyst import generate_analyst_report

router = APIRouter()


@router.post("/llm-report", summary="Generate AI market analyst report", tags=["LLM Report"])
def llm_report_endpoint(
    payload: Dict[str, Any],
    provider: Optional[str] = Query(
        default=None,
        description="Pin a specific provider: gemini | groq | nvidia | openrouter | mock. "
                    "Supports model suffix, e.g. 'nvidia:meta/llama-3.3-70b-instruct'.",
    ),
) -> Dict[str, Any]:
    """Generate an institutional-grade AI Market Analyst report.

    **Payload fields** (passed from ``/api/v1/analyze`` response):
    - ``current_regime`` (required) – Bull / Bear / Sideways
    - ``recommended_strategy`` – e.g. Momentum
    - ``recommendation_source`` – ml_classifier | historical_sharpe
    - ``overall_metrics`` – dict of strategy → MetricsResponse
    - ``risk_forecast`` – Monte Carlo percentiles
    - ``market_outlook`` – narrative string
    - ``probabilities`` – list of {strategy, probability}

    **Query param**:
    - ``provider`` – optional provider override (gemini, groq, nvidia, openrouter, mock)

    **Returns**:
    ```json
    {
      "report": "## Executive Summary...",
      "provider_used": "nvidia",
      "model_used": "meta/llama-3.3-70b-instruct",
      "generated_at": "2026-08-16T05:30:00+00:00",
      "fallback_history": ["gemini: skipped (no key)", "groq: success"]
    }
    ```
    """
    if "current_regime" not in payload:
        raise HTTPException(status_code=400, detail="Invalid payload: 'current_regime' is required.")

    # Map payload fields to generate_analyst_report arguments
    backtest_metrics = payload.get("overall_metrics")
    regime_data = {
        "current_regime": payload.get("current_regime", "Bull"),
        "recommended_strategy": payload.get("recommended_strategy", ""),
        "probabilities": payload.get("probabilities", []),
    }
    risk_metrics = payload.get("risk_forecast")
    if risk_metrics and isinstance(risk_metrics, dict):
        risk_metrics = {
            "worst_case_10": risk_metrics.get("worst_case_10"),
            "median_50": risk_metrics.get("median_50"),
            "best_case_90": risk_metrics.get("best_case_90"),
            "recommended_exposure": payload.get("recommended_exposure", ""),
        }
    ticker = payload.get("ticker", "^NSEI")
    raw_prompt = payload.get("custom_prompt") or payload.get("market_outlook")
    custom_prompt: str | None = None
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        custom_prompt = raw_prompt.strip()
    elif isinstance(raw_prompt, dict):
        outlook_text = raw_prompt.get("outlook") or raw_prompt.get("disclaimer") or ""
        if isinstance(outlook_text, str) and outlook_text.strip():
            custom_prompt = f"Market Outlook: {outlook_text.strip()}"

    try:
        result = generate_analyst_report(
            backtest_metrics=backtest_metrics,
            regime_data=regime_data,
            risk_metrics=risk_metrics,
            ticker=ticker,
            provider_override=provider,
            custom_prompt=custom_prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM report generation failed: {exc}")

    return {
        "report": result["report_markdown"],
        "provider_used": result["provider_used"],
        "model_used": result["model_used"],
        "generated_at": result["generated_at"],
        "fallback_history": result["fallback_history"],
    }
