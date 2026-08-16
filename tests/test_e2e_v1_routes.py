"""E2E Integration Tests for /api/v1/* routes.

Covers (happy path + boundary/error):
  GET  /api/v1/health
  GET  /api/v1/regime
  GET  /api/v1/recommend
  POST /api/v1/backtest
  POST /api/v1/analyze
  GET  /api/v1/benchmark      (NEW – XGBoost vs LSTM)
  POST /api/v1/llm-report     (NEW – AI Analyst)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ANALYZE_PAYLOAD: dict[str, Any] = {
    "ticker": "^NSEI",
    "start_date": "2020-01-01",
    "end_date": "2022-12-31",
    "strategy": "all",
    "initial_investment": 100000.0,
}

ANALYZE_TS_KEYS = [
    "ticker", "start_date", "end_date", "n_trading_days", "initial_investment",
    "current_regime", "recommended_strategy", "recommendation_source",
    "recommendation_reason", "recommended_exposure", "probabilities",
    "overall_metrics", "equity_curves", "ohlc_data", "regime_heatmap",
    "regime_timeline", "risk_forecast",
]

VALID_REGIMES = {"Bull", "Bear", "Sideways"}

LLM_REPORT_PAYLOAD: dict[str, Any] = {
    "current_regime": "Bull",
    "recommended_strategy": "Momentum",
    "recommendation_source": "ml_classifier",
    "recommendation_reason": "Highest Sharpe in Bull regime historically.",
    "recommended_exposure": "90% Equity",
    "risk_forecast": {"worst_case_10": -0.10, "median_50": -0.04, "best_case_90": -0.01},
    "market_outlook": "Bullish momentum sustained by FII inflows.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_benchmark_file(tmp_path: Path) -> Path:
    data = {
        "xgboost": {"CAGR": 0.18, "Sharpe": 1.42, "Sortino": 2.05,
                    "MaxDrawdown": -0.12, "Calmar": 1.50, "Volatility": 0.12},
        "lstm":    {"CAGR": 0.14, "Sharpe": 1.15, "Sortino": 1.60,
                    "MaxDrawdown": -0.16, "Calmar": 0.87, "Volatility": 0.14},
    }
    p = tmp_path / "benchmark_results.json"
    p.write_text(json.dumps(data))
    return p


# LLM Report helpers - use ?provider=mock query param to avoid real API calls
MOCK_PROVIDER_PARAM = "?provider=mock"


# ===========================================================================
# 1. Health check
# ===========================================================================

class TestHealthRoute:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_v1_prefix_no_crash(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code in (200, 404)

    def test_health_post_not_allowed(self):
        assert client.post("/health").status_code == 405


# ===========================================================================
# 2. Regime — GET /api/v1/regime
# ===========================================================================

class TestRegimeRoute:
    def test_regime_schema(self):
        resp = client.get("/api/v1/regime")
        assert resp.status_code == 200
        for key in ("current_regime", "regime_distribution", "total_days"):
            assert key in resp.json()

    def test_regime_valid_values(self):
        resp = client.get("/api/v1/regime?ticker=^NSEI")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_regime"] in VALID_REGIMES
        assert isinstance(data["regime_distribution"], dict)
        assert isinstance(data["total_days"], int) and data["total_days"] >= 0

    def test_regime_distribution_sums_to_total(self):
        resp = client.get("/api/v1/regime?ticker=^NSEI")
        data = resp.json()
        if data["regime_distribution"]:
            assert sum(data["regime_distribution"].values()) == data["total_days"]

    def test_regime_default_equals_explicit(self):
        r1 = client.get("/api/v1/regime").json()
        r2 = client.get("/api/v1/regime?ticker=^NSEI").json()
        assert r1["current_regime"] == r2["current_regime"]

    def test_regime_custom_ticker_no_crash(self):
        assert client.get("/api/v1/regime?ticker=RELIANCE.NS").status_code == 400

    def test_regime_post_not_allowed(self):
        assert client.post("/api/v1/regime").status_code == 405


# ===========================================================================
# 3. Recommend — GET /api/v1/recommend
# ===========================================================================

class TestRecommendRoute:
    def test_recommend_schema(self):
        resp = client.get("/api/v1/recommend?ticker=^NSEI")
        assert resp.status_code == 200
        for key in ("current_regime", "recommended_strategy", "recommendation_source", "probabilities"):
            assert key in resp.json()

    def test_recommend_regime_valid(self):
        assert client.get("/api/v1/recommend?ticker=^NSEI").json()["current_regime"] in VALID_REGIMES

    def test_recommend_source_known(self):
        src = client.get("/api/v1/recommend?ticker=^NSEI").json()["recommendation_source"]
        assert src in ("ml_classifier", "historical_sharpe")

    def test_recommend_probabilities_valid(self):
        probs = client.get("/api/v1/recommend?ticker=^NSEI").json()["probabilities"]
        assert isinstance(probs, list)
        for entry in probs:
            assert "strategy" in entry and "probability" in entry
            assert 0.0 <= entry["probability"] <= 1.0

    def test_recommend_post_not_allowed(self):
        assert client.post("/api/v1/recommend").status_code == 405


# ===========================================================================
# 4. Backtest — POST /api/v1/backtest
# ===========================================================================

class TestBacktestRoute:
    BASE = {"ticker": "^NSEI", "start_date": "2020-01-01", "end_date": "2022-12-31", "strategy": "all"}

    def test_backtest_schema(self):
        resp = client.post("/api/v1/backtest", json=self.BASE)
        assert resp.status_code == 200
        data = resp.json()
        for k in ("ticker", "n_trading_days", "results"):
            assert k in data
        assert len(data["results"]) >= 1

    def test_backtest_metrics_complete(self):
        resp = client.post("/api/v1/backtest", json=self.BASE)
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            m = r["metrics"]
            for field in ("CAGR", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "Volatility"):
                assert field in m and isinstance(m[field], (int, float))

    def test_backtest_single_strategy(self):
        payload = {**self.BASE, "strategy": "Momentum"}
        resp = client.post("/api/v1/backtest", json=payload)
        assert resp.status_code == 200
        assert "Momentum" in [r["strategy"] for r in resp.json()["results"]]

    def test_backtest_unknown_strategy_400(self):
        resp = client.post("/api/v1/backtest", json={**self.BASE, "strategy": "INVALID_XYZ"})
        assert resp.status_code == 400

    def test_backtest_with_commission_slippage(self):
        resp = client.post("/api/v1/backtest", json={**self.BASE, "commission_pct": 0.001, "slippage_pct": 0.0005})
        assert resp.status_code == 200

    def test_backtest_get_not_allowed(self):
        assert client.get("/api/v1/backtest").status_code == 405

    def test_backtest_malformed_types_422(self):
        """String where float expected triggers Pydantic 422."""
        resp = client.post("/api/v1/backtest", json={**self.BASE, "commission_pct": "high"})
        assert resp.status_code == 422


# ===========================================================================
# 5. Analyze — POST /api/v1/analyze
# ===========================================================================

class TestAnalyzeRoute:
    def test_analyze_frontend_contract(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        for key in ANALYZE_TS_KEYS:
            assert key in data, f"TS contract broken: '{key}' missing"

    def test_analyze_regime_valid(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        assert resp.json()["current_regime"] in VALID_REGIMES

    def test_analyze_equity_curves(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        assert resp.status_code == 200
        for _, curve in resp.json()["equity_curves"].items():
            if curve:
                pt = curve[0]
                assert "date" in pt and isinstance(pt["date"], str)
                assert "value" in pt and isinstance(pt["value"], (int, float))

    def test_analyze_ohlc_fields(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        ohlc = resp.json()["ohlc_data"]
        if ohlc:
            for field in ("date", "open", "high", "low", "close"):
                assert field in ohlc[0]

    def test_analyze_regime_timeline(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        for seg in resp.json()["regime_timeline"]:
            assert seg["regime"] in VALID_REGIMES
            assert "start" in seg and "end" in seg
            assert isinstance(seg["days"], int)

    def test_analyze_regime_heatmap(self):
        resp = client.post("/api/v1/analyze", json=ANALYZE_PAYLOAD)
        for entry in resp.json()["regime_heatmap"]:
            for f in ("strategy", "regime", "CAGR", "Sharpe"):
                assert f in entry

    def test_analyze_single_strategy(self):
        resp = client.post("/api/v1/analyze", json={**ANALYZE_PAYLOAD, "strategy": "Momentum"})
        assert resp.status_code == 200
        metrics = resp.json()["overall_metrics"]
        assert "Momentum" in metrics and "Buy & Hold" in metrics

    def test_analyze_short_range_400(self):
        resp = client.post("/api/v1/analyze", json={**ANALYZE_PAYLOAD, "start_date": "2022-01-01", "end_date": "2022-01-10"})
        assert resp.status_code == 400
        assert "Insufficient data" in resp.json()["detail"]

    def test_analyze_malformed_422(self):
        assert client.post("/api/v1/analyze", json={"initial_investment": "one_lakh"}).status_code == 422

    def test_analyze_get_not_allowed(self):
        assert client.get("/api/v1/analyze").status_code == 405


# ===========================================================================
# 6. Benchmark — GET /api/v1/benchmark  (NEW)
# ===========================================================================

class TestBenchmarkRoute:
    def test_benchmark_schema(self, tmp_path):
        p = make_benchmark_file(tmp_path)
        with patch("api.routes.benchmark.BENCHMARK_FILE", p):
            resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 200
        assert "xgboost" in resp.json() and "lstm" in resp.json()

    def test_benchmark_metric_types(self, tmp_path):
        p = make_benchmark_file(tmp_path)
        with patch("api.routes.benchmark.BENCHMARK_FILE", p):
            data = client.get("/api/v1/benchmark").json()
        for model in ("xgboost", "lstm"):
            for metric in ("CAGR", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "Volatility"):
                assert isinstance(data[model][metric], (int, float))

    def test_benchmark_missing_file_500(self):
        with patch("api.routes.benchmark.BENCHMARK_FILE", Path("/no/such/file.json")):
            resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 500
        assert "detail" in resp.json()

    def test_benchmark_corrupt_json_500(self, tmp_path):
        p = tmp_path / "benchmark_results.json"
        p.write_text("<<<CORRUPT>>>")
        with patch("api.routes.benchmark.BENCHMARK_FILE", p):
            assert client.get("/api/v1/benchmark").status_code == 500

    def test_benchmark_xgboost_cagr_gt_lstm(self, tmp_path):
        p = make_benchmark_file(tmp_path)
        with patch("api.routes.benchmark.BENCHMARK_FILE", p):
            data = client.get("/api/v1/benchmark").json()
        assert data["xgboost"]["CAGR"] > data["lstm"]["CAGR"]

    def test_benchmark_post_not_allowed(self):
        assert client.post("/api/v1/benchmark").status_code == 405

    def test_benchmark_ts_interface(self, tmp_path):
        """All MetricsResponse fields must be present (frontend/src/lib/benchmark.ts)."""
        p = make_benchmark_file(tmp_path)
        with patch("api.routes.benchmark.BENCHMARK_FILE", p):
            data = client.get("/api/v1/benchmark").json()
        for model in ("xgboost", "lstm"):
            for field in ("CAGR", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "Volatility"):
                assert field in data[model], f"BenchmarkResponse.{model}.{field} missing"


# ===========================================================================
# 7. LLM Report — POST /api/v1/llm-report  (NEW)
# ===========================================================================

class TestLLMReportRoute:
    """Uses ?provider=mock to avoid real API keys in CI."""

    def test_report_ok(self):
        resp = client.post("/api/v1/llm-report" + MOCK_PROVIDER_PARAM, json=LLM_REPORT_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data and isinstance(data["report"], str) and len(data["report"]) > 0

    def test_report_missing_regime_400(self):
        resp = client.post("/api/v1/llm-report" + MOCK_PROVIDER_PARAM,
                           json={"recommended_strategy": "Momentum"})
        assert resp.status_code == 400
        assert "Invalid payload" in resp.json()["detail"]

    def test_report_mock_provider_metadata(self):
        """Endpoint returns provider_used, model_used, generated_at, fallback_history."""
        resp = client.post("/api/v1/llm-report" + MOCK_PROVIDER_PARAM, json=LLM_REPORT_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider_used"] == "mock"
        assert "model_used" in data
        assert "generated_at" in data
        assert isinstance(data["fallback_history"], list)

    def test_report_waterfall_auto_selects_available_provider(self):
        """Without provider param, waterfall picks best available from .env."""
        resp = client.post("/api/v1/llm-report", json=LLM_REPORT_PAYLOAD)
        # Should succeed — at least mock is always available
        assert resp.status_code == 200
        assert "report" in resp.json()
        assert resp.json()["provider_used"] in ("gemini", "groq", "nvidia", "openrouter", "mock")

    def test_report_pinned_provider_mock(self):
        """?provider=mock forces MockProvider."""
        resp = client.post("/api/v1/llm-report?provider=mock", json=LLM_REPORT_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["provider_used"] == "mock"

    def test_report_bear_regime(self):
        bear = {**LLM_REPORT_PAYLOAD, "current_regime": "Bear", "recommended_strategy": "RSI"}
        resp = client.post("/api/v1/llm-report" + MOCK_PROVIDER_PARAM, json=bear)
        assert resp.status_code == 200 and "report" in resp.json()

    def test_report_get_not_allowed(self):
        assert client.get("/api/v1/llm-report").status_code == 405

    def test_report_ts_interface(self):
        """frontend/src/lib/report.ts expects {report: string} — now also has metadata."""
        payload = {"current_regime": "Sideways", "recommended_strategy": "Bollinger Bands",
                   "recommendation_source": "historical_sharpe"}
        resp = client.post("/api/v1/llm-report" + MOCK_PROVIDER_PARAM, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data and isinstance(data["report"], str)


# ===========================================================================
# 8. Cross-cutting error handling
# ===========================================================================

class TestCrossCutting:
    def test_unknown_route_404(self):
        resp = client.get("/api/v1/nonexistent_route_xyz_12345")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_docs_accessible(self):
        assert client.get("/docs").status_code == 200

    def test_openapi_schema_lists_new_routes(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema_str = str(resp.json())
        assert "/api/v1/benchmark" in schema_str
        assert "/api/v1/llm-report" in schema_str

    def test_root_does_not_crash(self):
        assert client.get("/").status_code in (200, 404)
