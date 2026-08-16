"""Regime Detection Module — Gaussian HMM & Backward Compatibility Facade.

Purpose:
    Detects market regimes (Bull, Bear, Sideways) using a 3-State Gaussian Hidden Markov Model.
    Provides backward-compatible facades for existing API routes, backtesting engine,
    and Streamlit application.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Support relative and absolute imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.regime_detector import GaussianHMMRegimeDetector, CANONICAL_REGIME_NAMES
from utils import compute_backtest_metrics


REGIME_FEATURE_COLUMNS = ["returns", "volatility", "momentum", "drawdown"]
REGIME_NAMES = ["Bull", "Bear", "Sideways"]
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "regime_model.pkl"
DEFAULT_REGIME_DATA_PATH = Path(__file__).parent.parent / "data" / "nifty50_regimes.parquet"
DEFAULT_WINDOW = 60

_REGIME_DETECTOR_CACHE: GaussianHMMRegimeDetector | None = None


def _validate_feature_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    # Check if either legacy or OHLCV columns exist
    has_legacy = all(col in df.columns for col in REGIME_FEATURE_COLUMNS)
    has_ohlc = "Close" in df.columns or "close" in df.columns
    if not has_legacy and not has_ohlc:
        raise KeyError(
            "df must contain regime features (returns, volatility, momentum, drawdown) or Close price."
        )


def _load_or_create_detector(df: pd.DataFrame | None = None) -> GaussianHMMRegimeDetector:
    global _REGIME_DETECTOR_CACHE
    if _REGIME_DETECTOR_CACHE is not None and _REGIME_DETECTOR_CACHE.is_fitted_:
        return _REGIME_DETECTOR_CACHE

    if DEFAULT_MODEL_PATH.exists():
        try:
            detector = GaussianHMMRegimeDetector.load(DEFAULT_MODEL_PATH)
            _REGIME_DETECTOR_CACHE = detector
            return detector
        except Exception:
            pass

    if df is not None:
        detector = GaussianHMMRegimeDetector()
        detector.fit(df)
        detector.save(DEFAULT_MODEL_PATH)
        _REGIME_DETECTOR_CACHE = detector
        return detector

    raise FileNotFoundError("No fitted regime model is available. Call fit_regimes(df) first.")


def fit_regimes(df: pd.DataFrame, persist: bool = False) -> pd.DataFrame:
    """Fit 3-state Gaussian HMM and attach regime labels + posteriors to DataFrame."""
    _validate_feature_frame(df)
    if len(df) < 3:
        raise ValueError("Need at least 3 non-null rows to fit regime model.")

    detector = GaussianHMMRegimeDetector()
    labeled_frame = detector.fit_predict(df)

    # Persistence is explicit so synthetic tests and ad-hoc analysis cannot
    # overwrite canonical production artifacts.
    if persist:
        detector.save(DEFAULT_MODEL_PATH)
        DEFAULT_REGIME_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            labeled_frame.to_parquet(DEFAULT_REGIME_DATA_PATH)
        except Exception:
            pass

    global _REGIME_DETECTOR_CACHE
    _REGIME_DETECTOR_CACHE = detector

    if persist:
        labeled_frame.attrs["regime_model_path"] = str(DEFAULT_MODEL_PATH)
        labeled_frame.attrs["regime_data_path"] = str(DEFAULT_REGIME_DATA_PATH)
    labeled_frame.attrs["transition_matrix"] = detector.get_transition_matrix()
    labeled_frame.attrs["stationary_distribution"] = detector.get_stationary_distribution()
    labeled_frame.attrs["expected_durations"] = detector.get_expected_durations()
    return labeled_frame


def fit_regimes_walk_forward(
    df: pd.DataFrame,
    min_train: int = 504,
    test_size: int = 21,
) -> pd.DataFrame:
    """Generate regime labels with expanding-window, no-future-information fits.

    Initial training observations are marked ``in_sample`` because they are needed
    to warm up the first model. Every later block is predicted by a model fitted
    only on observations before that block.
    """
    _validate_feature_frame(df)
    if len(df) <= min_train:
        raise ValueError(f"Need more than min_train={min_train} rows, got {len(df)}.")
    if test_size < 1:
        raise ValueError("test_size must be positive")

    output = df.copy()
    output["regime_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    output["regime"] = pd.Series(pd.NA, index=df.index, dtype="string")
    for column in ("prob_bear", "prob_sideways", "prob_bull"):
        output[column] = float("nan")
    output["regime_source"] = pd.Series(pd.NA, index=df.index, dtype="string")

    warmup = GaussianHMMRegimeDetector(n_iter=200)
    warmup.fit(df.iloc[:min_train])
    warmup_ids = warmup.predict(df.iloc[:min_train])
    warmup_probs = warmup.predict_proba(df.iloc[:min_train])
    output.iloc[:min_train, output.columns.get_loc("regime_id")] = warmup_ids
    output.iloc[:min_train, output.columns.get_loc("regime")] = [
        CANONICAL_REGIME_NAMES[index] for index in warmup_ids
    ]
    for position, column in enumerate(("prob_bear", "prob_sideways", "prob_bull")):
        output.iloc[:min_train, output.columns.get_loc(column)] = warmup_probs[:, position]
    output.iloc[:min_train, output.columns.get_loc("regime_source")] = "in_sample"

    for test_start in range(min_train, len(df), test_size):
        test_end = min(test_start + test_size, len(df))
        detector = GaussianHMMRegimeDetector(n_iter=200)
        detector.fit(df.iloc[:test_start])
        test_df = df.iloc[test_start:test_end]
        ids = detector.predict(test_df)
        probabilities = detector.predict_proba(test_df)
        output.iloc[test_start:test_end, output.columns.get_loc("regime_id")] = ids
        output.iloc[test_start:test_end, output.columns.get_loc("regime")] = [
            CANONICAL_REGIME_NAMES[index] for index in ids
        ]
        for position, column in enumerate(("prob_bear", "prob_sideways", "prob_bull")):
            output.iloc[test_start:test_end, output.columns.get_loc(column)] = probabilities[:, position]
        output.iloc[test_start:test_end, output.columns.get_loc("regime_source")] = "walk_forward"

    return output


def get_causal_regimes_for_analysis(
    df: pd.DataFrame,
    artifact_path: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load causal artifact or compute causal labels for an analysis window."""
    _validate_feature_frame(df)
    if artifact_path is not None:
        path = Path(artifact_path)
        if path.exists():
            try:
                artifact = pd.read_parquet(path)
                artifact.index = pd.to_datetime(artifact.index)
                selected = artifact.reindex(df.index)
                if not selected.empty and selected.notna().all(axis=None):
                    return selected, "walk_forward_artifact"
            except Exception:
                pass

    if len(df) > 252:
        min_train = min(504, max(126, len(df) // 2))
        if len(df) > min_train:
            return fit_regimes_walk_forward(df, min_train=min_train), "walk_forward_computed"

    retrospective = fit_regimes(df)
    retrospective["regime_source"] = "retrospective_insufficient_history"
    return retrospective, "retrospective_insufficient_history"


def get_current_regime(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> str:
    """Classify the most recent market regime from recent lookback window."""
    _validate_feature_frame(df)
    detector = _load_or_create_detector(df)
    return detector.get_current_regime(df, window=window)


def _slice_equity_metrics(equity_curve: pd.Series) -> dict[str, float]:
    portfolio_returns = equity_curve.pct_change().fillna(0.0)
    return compute_backtest_metrics(equity_curve, portfolio_returns)


def get_regime_performance(
    df: pd.DataFrame, all_strategy_results: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Compute per-regime strategy metrics from strategy equity curves."""
    _validate_feature_frame(df)
    if "regime" not in df.columns:
        df = fit_regimes(df)

    rows: list[dict[str, Any]] = []
    # Support both canonical order and presence in data
    regime_order = [r for r in ["Bull", "Sideways", "Bear"] if r in set(df["regime"].dropna().astype(str))]
    for regime_name in regime_order:
        regime_dates = df.index[df["regime"].astype(str) == regime_name]
        if len(regime_dates) == 0:
            continue
        for strategy_name, strategy_result in all_strategy_results.items():
            equity_curve = strategy_result.get("equity_curve")
            if equity_curve is None or not isinstance(equity_curve, pd.Series):
                continue
            regime_equity = equity_curve.reindex(regime_dates).dropna()
            if regime_equity.empty:
                continue
            metrics = _slice_equity_metrics(regime_equity)
            rows.append(
                {
                    "regime": regime_name,
                    "strategy": strategy_name,
                    "observations": int(len(regime_equity)),
                    **metrics,
                }
            )

    performance = pd.DataFrame(rows)
    if performance.empty:
        return performance
    return performance.sort_values(["regime", "strategy"]).reset_index(drop=True)


def get_regime_details(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Retrieve full analytics summary of the Gaussian HMM regime detector."""
    detector = _load_or_create_detector(df)
    return detector.get_regime_summary()
