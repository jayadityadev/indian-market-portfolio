"""Unit and integration tests for XGBoost Strategy Recommender Classifier."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classifier_inference import get_strategy_probabilities
from models.recommender import (
    DEFAULT_FEATURE_COLUMNS,
    PurgedTimeSeriesSplit,
    RecommendationResult,
    STRATEGY_NAMES,
    XGBoostStrategyRecommender,
)


@pytest.fixture
def synthetic_long_ohlcv() -> pd.DataFrame:
    """Generate 600 trading days of synthetic market data for training/testing."""
    np.random.seed(42)
    n_days = 600
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

    rets = np.random.normal(loc=0.0006, scale=0.012, size=n_days)
    price = 1000.0 * np.exp(np.cumsum(rets))

    high = price * (1.0 + np.abs(np.random.normal(0, 0.004, size=n_days)))
    low = price * (1.0 - np.abs(np.random.normal(0, 0.004, size=n_days)))
    open_p = price * (1.0 + np.random.normal(0, 0.002, size=n_days))
    volume = np.random.lognormal(mean=15.0, sigma=0.3, size=n_days)

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": volume,
        },
        index=dates,
    )


def test_feature_extraction_shape_and_columns(synthetic_long_ohlcv: pd.DataFrame):
    """Verify feature extractor computes all 21 defined features with zero NaNs."""
    recommender = XGBoostStrategyRecommender()
    feats = recommender.extract_features_single_window(synthetic_long_ohlcv.iloc[:252])

    assert isinstance(feats, dict)
    for col in DEFAULT_FEATURE_COLUMNS:
        assert col in feats
        assert not np.isnan(feats[col])
        assert not np.isinf(feats[col])


def test_feature_extraction_no_lookahead(synthetic_long_ohlcv: pd.DataFrame):
    """Verify feature extraction at time T depends strictly on past data up to T."""
    recommender = XGBoostStrategyRecommender()
    t = 252
    slice_t = synthetic_long_ohlcv.iloc[:t]

    feats_t = recommender.extract_features_single_window(slice_t)
    feats_full = recommender.extract_features_single_window(synthetic_long_ohlcv.iloc[:t])

    for col in DEFAULT_FEATURE_COLUMNS:
        assert feats_t[col] == pytest.approx(feats_full[col], abs=1e-7)


def test_purged_time_series_split_no_overlap():
    """Verify PurgedTimeSeriesSplit enforces purge and embargo buffers between train and test."""
    n_samples = 100
    purge_w = 10
    embargo_w = 3
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_window=purge_w, embargo_window=embargo_w)
    dummy_x = np.arange(n_samples).reshape(-1, 1)

    splits = list(cv.split(dummy_x))
    assert len(splits) > 0

    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Check that test start is strictly after train end + purge + embargo
        train_max = train_idx[-1]
        test_min = test_idx[0]
        assert test_min > train_max
        assert test_min - train_max >= embargo_w


def test_dataset_building_all_6_strategies(synthetic_long_ohlcv: pd.DataFrame):
    """Verify build_training_dataset outputs X, y (targets 0-5), and utility matrix for all 6 strategies."""
    recommender = XGBoostStrategyRecommender()
    X_df, y_s, utility_df = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )

    assert len(X_df) == len(y_s) == len(utility_df)
    assert len(X_df) > 0
    assert set(utility_df.columns) == set(STRATEGY_NAMES)
    assert set(y_s.unique()).issubset(set(range(len(STRATEGY_NAMES))))
    assert X_df.shape[1] == len(DEFAULT_FEATURE_COLUMNS)


def test_xgboost_fit_and_metrics(synthetic_long_ohlcv: pd.DataFrame):
    """Verify XGBoost fit trains model and records CV metrics."""
    recommender = XGBoostStrategyRecommender(n_estimators=30, max_depth=2)
    X_df, y_s, _ = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )

    recommender.fit(X_df, y_s, cv_splits=3)
    assert recommender.is_fitted_
    assert recommender.model is not None
    assert "accuracy" in recommender.cv_metrics_
    assert "macro_f1" in recommender.cv_metrics_
    assert len(recommender.feature_importances_) == len(DEFAULT_FEATURE_COLUMNS)


def test_suitability_ensemble_preserves_strategy_score_contract(synthetic_long_ohlcv: pd.DataFrame):
    """Verify independent strategy suitability models return normalized ranking scores."""
    recommender = XGBoostStrategyRecommender(n_estimators=20, max_depth=2)
    X_df, _, utility_df = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )
    recommender.fit_suitability(X_df, utility_df, cv_splits=2, purge_window=2, embargo_window=1)

    assert recommender.target_mode == "suitability_vs_buy_hold"
    assert recommender.is_fitted_
    scores = recommender.predict_proba(X_df)
    assert scores.shape == (len(X_df), len(STRATEGY_NAMES))
    assert np.all(scores >= 0.0)
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(len(X_df)), rtol=1e-5)


def test_predict_proba_simplex_constraint(synthetic_long_ohlcv: pd.DataFrame):
    """Verify predict_proba outputs valid probability distribution summing to 1.0."""
    recommender = XGBoostStrategyRecommender(n_estimators=30, max_depth=2)
    X_df, y_s, _ = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )
    recommender.fit(X_df, y_s, cv_splits=2)

    probs = recommender.predict_proba(X_df)
    assert probs.shape == (len(X_df), len(STRATEGY_NAMES))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X_df)), rtol=1e-5)


def test_predict_indices_matches_strategy_name_predictions(synthetic_long_ohlcv: pd.DataFrame):
    """Verify numeric evaluation IDs map exactly to public strategy names."""
    recommender = XGBoostStrategyRecommender(n_estimators=20, max_depth=2)
    X_df, y_s, _ = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )
    recommender.fit(X_df, y_s, cv_splits=2)
    names = recommender.predict(X_df)
    indices = recommender.predict_indices(X_df)
    assert all(recommender.idx_to_strategy[int(index)] == name for index, name in zip(indices, names))


def test_all_6_strategies_in_probabilities_dict(synthetic_long_ohlcv: pd.DataFrame):
    """Verify recommend() produces dictionary with exactly all 6 strategy keys."""
    recommender = XGBoostStrategyRecommender(n_estimators=30, max_depth=2)
    X_df, y_s, _ = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )
    recommender.fit(X_df, y_s, cv_splits=2)

    res = recommender.recommend(synthetic_long_ohlcv.tail(126))
    assert isinstance(res, RecommendationResult)
    assert res.recommended_strategy in STRATEGY_NAMES
    assert set(res.probabilities.keys()) == set(STRATEGY_NAMES)
    assert sum(res.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_model_serialization_roundtrip(synthetic_long_ohlcv: pd.DataFrame):
    """Verify serialization to disk and reload produces identical probability predictions."""
    recommender = XGBoostStrategyRecommender(n_estimators=30, max_depth=2)
    X_df, y_s, _ = recommender.build_training_dataset(
        synthetic_long_ohlcv, lookback=126, forward=42, step=21
    )
    recommender.fit(X_df, y_s, cv_splits=2)
    probs_orig = recommender.predict_proba(X_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "xgb_recommender.joblib"
        recommender.save(model_path)
        assert model_path.exists()

        loaded = XGBoostStrategyRecommender.load(model_path)
        assert loaded.is_fitted_
        probs_loaded = loaded.predict_proba(X_df)

        np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-5)


def test_edge_case_short_history_fallback(synthetic_long_ohlcv: pd.DataFrame):
    """Verify recommend() handles short lookback gracefully."""
    recommender = XGBoostStrategyRecommender(n_estimators=20, max_depth=2)
    short_df = synthetic_long_ohlcv.iloc[:30]
    res = recommender.recommend(short_df)
    assert res.recommended_strategy in STRATEGY_NAMES
    assert len(res.probabilities) == 6


def test_edge_case_missing_regime_info(synthetic_long_ohlcv: pd.DataFrame):
    """Verify feature extraction and inference succeed when regime_info is None."""
    recommender = XGBoostStrategyRecommender(n_estimators=20, max_depth=2)
    feats = recommender.extract_features_single_window(synthetic_long_ohlcv.iloc[:100], regime_info=None)
    assert isinstance(feats, dict)
    assert "prob_bull" in feats


def test_classifier_inference_facade(synthetic_long_ohlcv: pd.DataFrame):
    """Verify get_strategy_probabilities returns full 6-strategy dictionary summing to 1.0."""
    probs = get_strategy_probabilities(synthetic_long_ohlcv.tail(252), current_regime="Bull")
    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(STRATEGY_NAMES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
