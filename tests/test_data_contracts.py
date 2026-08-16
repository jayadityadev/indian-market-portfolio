from __future__ import annotations

import pandas as pd

from data_contracts import validate_label_frame, validate_price_frame, validate_regime_frame


def price_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=252, freq="D")
    close = pd.Series(100.0, index=index)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1000,
        },
        index=index,
    )


def test_price_contract_accepts_valid_ohlcv():
    result = validate_price_frame(price_frame())
    assert result.valid
    assert result.facts["rows"] == 252


def test_price_contract_rejects_bad_candle_relationship():
    frame = price_frame()
    frame.iloc[0, frame.columns.get_loc("High")] = 99
    result = validate_price_frame(frame)
    assert not result.valid
    assert any("High" in error for error in result.errors)


def test_regime_contract_rejects_incomplete_coverage():
    prices = price_frame()
    regimes = pd.DataFrame(
        {
            "regime_id": 1,
            "regime": "Sideways",
            "prob_bear": 0.2,
            "prob_sideways": 0.6,
            "prob_bull": 0.2,
        },
        index=prices.index[:10],
    )
    result = validate_regime_frame(regimes, prices.index)
    assert not result.valid
    assert any("coverage" in error for error in result.errors)


def test_label_contract_rejects_small_training_set():
    labels = pd.DataFrame(
        {
            "window_end_date": pd.date_range("2020-01-01", periods=10),
            "strategy": "Momentum",
            "label": 1,
        }
    )
    result = validate_label_frame(labels)
    assert not result.valid
    assert any("unique dates" in error for error in result.errors)
