"""High-level AI Market Analyst service interface for Indian Market Portfolio Intelligence."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from llm.client import LLMClient
from llm.prompts import (
    build_analyst_user_prompt,
    get_analyst_system_prompt,
)

logger = logging.getLogger(__name__)


def generate_analyst_report(
    backtest_metrics: dict[str, Any] | None = None,
    regime_data: dict[str, Any] | None = None,
    risk_metrics: dict[str, Any] | None = None,
    ticker: str = "^NSEI",
    provider_override: str | None = None,
    custom_prompt: str | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Generate an institutional-grade AI Market Analyst report for Indian equities.

    Args:
        backtest_metrics: Dictionary of performance metrics across quantitative strategies.
        regime_data: Dictionary containing HMM regime state, transition matrix, and strategy probabilities.
        risk_metrics: Dictionary containing Monte Carlo forward drawdown percentiles and exposure limits.
        ticker: Ticker symbol (e.g. '^NSEI', 'RELIANCE.NS', 'TCS.NS').
        provider_override: Optional LLM provider override ('gemini', 'groq', 'nvidia', 'openrouter', 'mock').
        custom_prompt: Optional user-defined instructions or custom focus area.
        client: Optional pre-configured LLMClient instance.

    Returns:
        dict matching the LLMReportResponse contract:
            - ticker (str)
            - current_regime (str)
            - recommended_strategy (str)
            - provider_used (str)
            - model_used (str)
            - report_markdown (str)
            - generated_at (str, ISO-8601 UTC timestamp)
            - fallback_history (list[str])
    """
    resolved_ticker = ticker.strip() if ticker and isinstance(ticker, str) and ticker.strip() else "^NSEI"

    # Extract regime and recommended strategy with robust fallbacks
    current_regime = "Bull"
    recommended_strategy = "Momentum"

    if regime_data:
        current_regime = regime_data.get("current_regime") or "Bull"
        rec_strat = regime_data.get("recommended_strategy")
        if rec_strat and isinstance(rec_strat, str) and rec_strat.strip():
            recommended_strategy = rec_strat.strip()
        elif "probabilities" in regime_data and regime_data["probabilities"]:
            probs = regime_data["probabilities"]
            if isinstance(probs, dict) and probs:
                def _safe_prob_val(v: Any) -> float:
                    if v is None:
                        return -1.0
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        try:
                            return float(v.rstrip("%"))
                        except ValueError:
                            return -1.0
                    return -1.0

                valid_probs = {k: _safe_prob_val(v) for k, v in probs.items() if _safe_prob_val(v) >= 0}
                if valid_probs:
                    recommended_strategy = max(valid_probs.items(), key=lambda x: x[1])[0]
            elif isinstance(probs, list) and probs:
                def _get_prob(item: Any) -> float:
                    if isinstance(item, dict):
                        p = item.get("probability")
                        if p is not None and isinstance(p, (int, float)):
                            return float(p)
                        if isinstance(p, str):
                            try:
                                return float(p.rstrip("%"))
                            except ValueError:
                                pass
                    return 0.0

                valid_items = [p for p in probs if isinstance(p, dict) and p.get("strategy")]
                if valid_items:
                    sorted_probs = sorted(valid_items, key=_get_prob, reverse=True)
                    strat = sorted_probs[0].get("strategy")
                    if strat and isinstance(strat, str):
                        recommended_strategy = strat.strip()

    system_prompt = get_analyst_system_prompt()
    user_prompt = build_analyst_user_prompt(
        backtest_metrics=backtest_metrics,
        regime_data=regime_data,
        risk_metrics=risk_metrics,
        ticker=resolved_ticker,
        custom_prompt=custom_prompt,
    )

    llm_client = client or LLMClient()
    gen_result = llm_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider_override=provider_override,
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "ticker": resolved_ticker,
        "current_regime": current_regime,
        "recommended_strategy": recommended_strategy,
        "provider_used": gen_result["provider_used"],
        "model_used": gen_result["model_used"],
        "report_markdown": gen_result["content"],
        "generated_at": timestamp,
        "fallback_history": gen_result["fallback_history"],
    }
