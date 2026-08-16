"""Unit and integration tests for Gaussian HMM Market Regime Detection."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.regime_detector import (
    CANONICAL_REGIME_NAMES,
    GaussianHMMRegimeDetector,
)
from regime_detector import (
    fit_regimes,
    get_current_regime,
    get_regime_performance,
    get_regime_details,
)


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generate 300 days of synthetic OHLCV data with bull, bear, and sideways phases."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    # Bull phase (100 days)
    bull_rets = np.random.normal(loc=0.0015, scale=0.008, size=100)
    # Bear phase (100 days)
    bear_rets = np.random.normal(loc=-0.002, scale=0.022, size=100)
    # Sideways phase (100 days)
    side_rets = np.random.normal(loc=0.0001, scale=0.006, size=100)

    rets = np.concatenate([bull_rets, bear_rets, side_rets])
    price = 100.0 * np.exp(np.cumsum(rets))

    high = price * (1.0 + np.abs(np.random.normal(0, 0.005, size=n_days)))
    low = price * (1.0 - np.abs(np.random.normal(0, 0.005, size=n_days)))
    open_p = price * (1.0 + np.random.normal(0, 0.002, size=n_days))
    volume = np.random.lognormal(mean=14.0, sigma=0.4, size=n_days)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def test_hmm_fit_predict_structure(synthetic_ohlcv_df: pd.DataFrame):
    """Verify GaussianHMMRegimeDetector fit_predict produces expected columns and non-null values."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    labeled_df = detector.fit_predict(synthetic_ohlcv_df)

    expected_cols = ["regime_id", "regime", "prob_bear", "prob_sideways", "prob_bull"]
    for col in expected_cols:
        assert col in labeled_df.columns
        assert labeled_df[col].isna().sum() == 0

    assert set(labeled_df["regime"].unique()).issubset({"Bear", "Sideways", "Bull"})
    assert set(labeled_df["regime_id"].unique()).issubset({0, 1, 2})


def test_hmm_canonical_order(synthetic_ohlcv_df: pd.DataFrame):
    """Validate canonical ordering: Bull (state 2) has higher returns than Bear (state 0)."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    labeled_df = detector.fit_predict(synthetic_ohlcv_df)

    log_returns = np.log(synthetic_ohlcv_df["Close"] / synthetic_ohlcv_df["Close"].shift(1)).fillna(0.0)

    bull_returns = log_returns[labeled_df["regime_id"] == 2]
    bear_returns = log_returns[labeled_df["regime_id"] == 0]

    assert bull_returns.mean() > bear_returns.mean()


def test_transition_matrix_stochastic(synthetic_ohlcv_df: pd.DataFrame):
    """Verify Markov transition matrix is (3, 3), non-negative, and rows sum to 1.0."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)

    trans_mat = detector.get_transition_matrix()
    assert trans_mat.shape == (3, 3)
    assert np.all(trans_mat >= 0.0)
    assert np.all(trans_mat <= 1.0)
    np.testing.assert_allclose(trans_mat.sum(axis=1), np.ones(3), rtol=1e-5)


def test_stationary_distribution_eigen(synthetic_ohlcv_df: pd.DataFrame):
    """Verify stationary distribution satisfies sum(pi) = 1, pi >= 0, and pi * P = pi."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)

    pi = detector.get_stationary_distribution()
    p = detector.get_transition_matrix()

    assert pi.shape == (3,)
    assert np.all(pi >= 0.0)
    assert abs(np.sum(pi) - 1.0) < 1e-4

    # pi * P == pi
    pi_p = np.dot(pi, p)
    np.testing.assert_allclose(pi_p, pi, atol=1e-3)


def test_posterior_probabilities_sum(synthetic_ohlcv_df: pd.DataFrame):
    """Verify that Bayesian state posteriors sum to 1.0 for each observation."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)
    posteriors = detector.predict_proba(synthetic_ohlcv_df)

    assert posteriors.shape == (len(synthetic_ohlcv_df), 3)
    assert np.all(posteriors >= 0.0)
    assert np.all(posteriors <= 1.0)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(len(synthetic_ohlcv_df)), rtol=1e-5)


def test_expected_durations_positive(synthetic_ohlcv_df: pd.DataFrame):
    """Verify that expected regime dwell durations are positive and >= 1.0 trading days."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)

    durations = detector.get_expected_durations()
    assert set(durations.keys()) == {"Bear", "Sideways", "Bull"}
    for name, dur in durations.items():
        assert dur >= 1.0


def test_serialization_roundtrip(synthetic_ohlcv_df: pd.DataFrame):
    """Verify model bundle save and load produces byte/float-identical predictions."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)
    preds_orig = detector.predict(synthetic_ohlcv_df)
    probs_orig = detector.predict_proba(synthetic_ohlcv_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "test_hmm.pkl"
        detector.save(model_file)
        assert model_file.exists()

        loaded = GaussianHMMRegimeDetector.load(model_file)
        preds_loaded = loaded.predict(synthetic_ohlcv_df)
        probs_loaded = loaded.predict_proba(synthetic_ohlcv_df)

        np.testing.assert_array_equal(preds_orig, preds_loaded)
        np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-6)
        np.testing.assert_allclose(
            detector.get_transition_matrix(), loaded.get_transition_matrix(), atol=1e-6
        )


def test_legacy_api_compatibility(synthetic_ohlcv_df: pd.DataFrame):
    """Verify legacy facade functions (fit_regimes, get_current_regime, get_regime_performance)."""
    labeled_df = fit_regimes(synthetic_ohlcv_df)
    assert "regime" in labeled_df.columns
    assert "regime_id" in labeled_df.columns

    current_r = get_current_regime(synthetic_ohlcv_df, window=30)
    assert current_r in CANONICAL_REGIME_NAMES

    # Dummy strategy results with equity curves
    mock_strat_results = {
        "Buy & Hold": {
            "equity_curve": pd.Series(
                np.cumprod(1.0 + np.random.normal(0.0005, 0.01, size=len(synthetic_ohlcv_df))),
                index=synthetic_ohlcv_df.index,
            )
        },
        "Momentum": {
            "equity_curve": pd.Series(
                np.cumprod(1.0 + np.random.normal(0.0008, 0.009, size=len(synthetic_ohlcv_df))),
                index=synthetic_ohlcv_df.index,
            )
        },
    }

    perf = get_regime_performance(labeled_df, mock_strat_results)
    assert isinstance(perf, pd.DataFrame)
    if not perf.empty:
        assert "regime" in perf.columns
        assert "strategy" in perf.columns
        assert "Sharpe" in perf.columns


def test_edge_cases_short_data():
    """Verify ValueError is raised when fitting on fewer than 3 samples."""
    short_df = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2022-01-01", periods=2, freq="D"),
    )
    detector = GaussianHMMRegimeDetector()
    with pytest.raises(ValueError, match="at least 3"):
        detector.fit(short_df)


def test_edge_cases_missing_volume(synthetic_ohlcv_df: pd.DataFrame):
    """Verify feature extractor handles missing Volume column gracefully."""
    df_no_vol = synthetic_ohlcv_df[["Open", "High", "Low", "Close"]].copy()
    detector = GaussianHMMRegimeDetector(random_state=42)
    labeled_df = detector.fit_predict(df_no_vol)
    assert "regime" in labeled_df.columns
    assert len(labeled_df) == len(df_no_vol)


def test_regime_summary_dict(synthetic_ohlcv_df: pd.DataFrame):
    """Verify get_regime_summary returns all expected dictionary keys."""
    detector = GaussianHMMRegimeDetector(random_state=42)
    detector.fit(synthetic_ohlcv_df)
    summary = detector.get_regime_summary()

    assert "regime_names" in summary
    assert "transition_matrix" in summary
    assert "stationary_distribution" in summary
    assert "expected_durations_days" in summary
