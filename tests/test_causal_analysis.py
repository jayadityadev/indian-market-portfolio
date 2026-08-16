from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detector import get_causal_regimes_for_analysis


def test_causal_analysis_prefers_complete_walk_forward_artifact(tmp_path):
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"Close": [100.0, 101.0, 100.5, 102.0]}, index=index)
    artifact = prices.copy()
    artifact["regime_id"] = [0, 1, 1, 2]
    artifact["regime"] = ["Bear", "Sideways", "Sideways", "Bull"]
    artifact["prob_bear"] = [1.0, 0.0, 0.0, 0.0]
    artifact["prob_sideways"] = [0.0, 1.0, 1.0, 0.0]
    artifact["prob_bull"] = [0.0, 0.0, 0.0, 1.0]
    artifact["regime_source"] = "walk_forward"
    path = tmp_path / "regimes.parquet"
    artifact.to_parquet(path)

    result, source = get_causal_regimes_for_analysis(prices, path)
    assert source == "walk_forward_artifact"
    assert result["regime"].iloc[-1] == "Bull"


def test_causal_analysis_marks_short_history_retrospective():
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    close = 100.0 * np.exp(np.cumsum(np.full(20, 0.001)))
    prices = pd.DataFrame({"Close": close}, index=index)
    result, source = get_causal_regimes_for_analysis(prices)
    assert source == "retrospective_insufficient_history"
    assert result["regime_source"].iloc[0] == "retrospective_insufficient_history"
