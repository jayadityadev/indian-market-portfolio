"""Regime Detection Module — KMeans clustering on market features.

Purpose:
	Detects market regimes (Bull, Bear, Sideways) using unsupervised learning (KMeans k=3).
	Classifies historical periods and the current market state. Enables regime-aware strategy
	recommendations and per-regime performance analysis.

Inputs:
	- DataFrame with feature columns: returns, volatility, momentum, drawdown
	- DatetimeIndex required

Outputs:
	- Regime labels persisted to nifty50_regimes.parquet
	- Fitted KMeans model + label_map + scaler saved to regime_model.pkl
	- get_current_regime() → str ("Bull", "Bear", "Sideways")
	- get_regime_performance() → per-strategy, per-regime Sharpe ratios (DataFrame)

Key Functions:
	- fit_regimes(df) → DataFrame with regime labels attached
	- get_current_regime(df) → current market regime as string
	- get_regime_performance(df, all_strategy_results) → performance by regime

Implementation Notes:
	- Features normalized with StandardScaler before clustering
	- Bull cluster: max(returns + momentum)
	- Bear cluster: min(returns - volatility)
	- Sideways: remaining cluster
	- Label mapping persisted in pkl to avoid re-derivation on each load
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import compute_backtest_metrics


REGIME_FEATURE_COLUMNS = ["returns", "volatility", "momentum", "drawdown"]
REGIME_NAMES = ["Bull", "Bear", "Sideways"]
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "regime_model.pkl"
DEFAULT_REGIME_DATA_PATH = Path(__file__).parent.parent / "data" / "nifty50_regimes.parquet"
DEFAULT_WINDOW = 60

_REGIME_CACHE: dict[str, Any] | None = None


def _validate_feature_frame(df: pd.DataFrame) -> None:
	missing_columns = [column for column in REGIME_FEATURE_COLUMNS if column not in df.columns]
	if missing_columns:
		raise KeyError(
			"df must contain regime features: " + ", ".join(missing_columns)
		)
	if not isinstance(df.index, pd.DatetimeIndex):
		raise TypeError("df must be indexed by a pandas DatetimeIndex.")


def _build_centroid_frame(scaler: StandardScaler, kmeans: KMeans) -> pd.DataFrame:
	centroids = scaler.inverse_transform(kmeans.cluster_centers_)
	return pd.DataFrame(centroids, columns=REGIME_FEATURE_COLUMNS)


def _assign_cluster_labels(centroid_frame: pd.DataFrame) -> dict[int, str]:
	label_map: dict[int, str] = {}
	remaining_clusters = set(centroid_frame.index)

	bull_cluster = int((centroid_frame["returns"] + centroid_frame["momentum"]).idxmax())
	label_map[bull_cluster] = "Bull"
	remaining_clusters.discard(bull_cluster)

	bear_candidates = centroid_frame.loc[list(remaining_clusters)]
	bear_cluster = int((bear_candidates["returns"] - bear_candidates["volatility"]).idxmin())
	label_map[bear_cluster] = "Bear"
	remaining_clusters.discard(bear_cluster)

	if remaining_clusters:
		label_map[int(next(iter(remaining_clusters)))] = "Sideways"

	return label_map


def _persist_regime_bundle(bundle: dict[str, Any], model_path: Path | str = DEFAULT_MODEL_PATH) -> Path:
	destination = Path(model_path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(bundle, destination)
	return destination


def _load_regime_bundle(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
	source = Path(model_path)
	if not source.exists():
		raise FileNotFoundError(f"Regime model file not found: {source}")
	loaded_bundle = joblib.load(source)
	if not isinstance(loaded_bundle, dict):
		raise TypeError("Regime model file did not contain the expected bundle dictionary.")
	return loaded_bundle


def _get_regime_bundle(df: pd.DataFrame | None = None) -> dict[str, Any]:
	global _REGIME_CACHE
	if _REGIME_CACHE is not None:
		return _REGIME_CACHE

	if DEFAULT_MODEL_PATH.exists():
		_REGIME_CACHE = _load_regime_bundle(DEFAULT_MODEL_PATH)
		return _REGIME_CACHE

	if df is None:
		raise FileNotFoundError("No fitted regime model is available. Call fit_regimes(df) first.")

	_ = fit_regimes(df)
	if _REGIME_CACHE is None:
		raise RuntimeError("Regime model cache was not populated after fitting.")
	return _REGIME_CACHE


def fit_regimes(df: pd.DataFrame) -> pd.DataFrame:
	"""Fit a 3-cluster regime model and attach labels to the historical frame."""
	_validate_feature_frame(df)
	feature_frame = df.loc[:, REGIME_FEATURE_COLUMNS].dropna().copy()
	if len(feature_frame) < len(REGIME_NAMES):
		raise ValueError("Need at least 3 non-null rows to fit regime clusters.")

	scaler = StandardScaler()
	feature_matrix = scaler.fit_transform(feature_frame)
	kmeans = KMeans(n_clusters=len(REGIME_NAMES), random_state=42, n_init=10)
	kmeans.fit(feature_matrix)
	centroid_frame = _build_centroid_frame(scaler, kmeans)
	cluster_to_label = _assign_cluster_labels(centroid_frame)

	labeled_frame = df.copy()
	labeled_frame["regime_id"] = pd.Series(pd.array([pd.NA] * len(labeled_frame), dtype="Int64"), index=labeled_frame.index)
	labeled_frame["regime"] = pd.Series(pd.array([pd.NA] * len(labeled_frame), dtype="string"), index=labeled_frame.index)

	predicted_clusters = pd.Series(kmeans.predict(feature_matrix), index=feature_frame.index, dtype="int64")
	labeled_frame.loc[predicted_clusters.index, "regime_id"] = predicted_clusters.astype("Int64")
	labeled_frame.loc[predicted_clusters.index, "regime"] = predicted_clusters.map(cluster_to_label)

	bundle = {
		"kmeans": kmeans,
		"scaler": scaler,
		"cluster_to_label": cluster_to_label,
		"centroid_frame": centroid_frame,
		"feature_columns": REGIME_FEATURE_COLUMNS,
		"window": DEFAULT_WINDOW,
	}
	_persist_regime_bundle(bundle, DEFAULT_MODEL_PATH)
	DEFAULT_REGIME_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
	labeled_frame.to_parquet(DEFAULT_REGIME_DATA_PATH)
	global _REGIME_CACHE
	_REGIME_CACHE = bundle

	labeled_frame.attrs["regime_model_path"] = str(DEFAULT_MODEL_PATH)
	labeled_frame.attrs["regime_data_path"] = str(DEFAULT_REGIME_DATA_PATH)
	labeled_frame.attrs["regime_centroids"] = centroid_frame
	labeled_frame.attrs["cluster_to_label"] = cluster_to_label
	return labeled_frame


def get_current_regime(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> str:
	"""Classify the most recent market regime from a rolling feature window."""
	_validate_feature_frame(df)
	bundle = _get_regime_bundle(df)
	recent_frame = df.loc[:, REGIME_FEATURE_COLUMNS].dropna().tail(window)
	if recent_frame.empty:
		raise ValueError("Not enough recent data to classify the current regime.")

	feature_matrix = bundle["scaler"].transform(recent_frame)
	predictions = bundle["kmeans"].predict(feature_matrix)
	current_cluster = int(pd.Series(predictions).mode().iloc[0])
	return str(bundle["cluster_to_label"][current_cluster])


def _slice_equity_metrics(equity_curve: pd.Series) -> dict[str, float]:
	portfolio_returns = equity_curve.pct_change().fillna(0.0)
	return compute_backtest_metrics(equity_curve, portfolio_returns)


def get_regime_performance(df: pd.DataFrame, all_strategy_results: dict[str, dict[str, object]]) -> pd.DataFrame:
	"""Compute per-regime strategy metrics from strategy equity curves."""
	_validate_feature_frame(df)
	if "regime" not in df.columns:
		df = fit_regimes(df)

	rows: list[dict[str, Any]] = []
	regime_order = [regime_name for regime_name in REGIME_NAMES if regime_name in set(df["regime"].dropna().astype(str))]
	for regime_name in regime_order:
		regime_dates = df.index[df["regime"].astype(str) == regime_name]
		if len(regime_dates) == 0:
			continue
		for strategy_name, strategy_result in all_strategy_results.items():
			equity_curve = strategy_result["equity_curve"]
			if not isinstance(equity_curve, pd.Series):
				raise TypeError("Each strategy result must include an equity_curve Series.")
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
