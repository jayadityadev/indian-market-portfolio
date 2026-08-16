"""Gaussian Hidden Markov Model Regime Detector.

Implements a 3-state Gaussian HMM (Bear, Sideways, Bull) for market regime detection,
computing state transition probabilities, stationary distributions via left eigenvectors,
Bayesian state posteriors, and expected regime durations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


CANONICAL_REGIME_NAMES = ["Bear", "Sideways", "Bull"]  # 0=Bear, 1=Sideways, 2=Bull
LEGACY_REGIME_NAMES = ["Bull", "Bear", "Sideways"]


class GaussianHMMRegimeDetector:
    """3-State Gaussian Hidden Markov Model for financial regime detection.

    Canonical states:
        0: Bear (Low / Negative returns, High volatility)
        1: Sideways (Moderate returns, Medium / Low volatility)
        2: Bull (High / Positive returns, Moderate / Low volatility)
    """

    FEATURE_NAMES: list[str] = ["log_return", "volatility_20d", "volume_ratio", "momentum_20d"]
    LEGACY_FEATURE_NAMES: list[str] = ["returns", "volatility", "momentum", "drawdown"]

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 42,
        min_covar: float = 1e-4,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.min_covar = min_covar

        self.model: GaussianHMM | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_fitted_: bool = False

        # Canonical ordering mapping: raw_state_idx -> canonical_state_idx (0=Bear, 1=Sideways, 2=Bull)
        self.raw_to_canonical_: np.ndarray | None = None
        self.canonical_to_raw_: np.ndarray | None = None

        # Analytics properties (in canonical order)
        self.transition_matrix_: np.ndarray | None = None
        self.stationary_distribution_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.expected_durations_: dict[str, float] = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract stationary regime features from OHLCV or feature DataFrame.

        Accepts either raw OHLCV data or pre-computed feature DataFrames.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Check if pre-computed canonical features exist
        if all(col in df.columns for col in self.FEATURE_NAMES):
            feat = df[self.FEATURE_NAMES].copy()
            return feat.ffill().bfill().fillna(0.0)

        # Check if pre-computed legacy features exist
        if all(col in df.columns for col in self.LEGACY_FEATURE_NAMES):
            feat = df[self.LEGACY_FEATURE_NAMES].copy()
            return feat.ffill().bfill().fillna(0.0)

        # Compute from OHLCV
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "close" not in cols_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 1:
                feat = df[numeric_cols[:4]].copy()
                return feat.ffill().bfill().fillna(0.0)
            raise KeyError("DataFrame must contain 'Close' or feature columns.")

        close_col = cols_lower["close"]
        close_s = df[close_col].astype(float)

        # 1. Log Return
        log_ret = np.log(close_s / close_s.shift(1)).fillna(0.0)

        # 2. Rolling 20-day annualized realized volatility
        vol_20d = (log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252)).fillna(0.0)

        # 3. Log Volume Ratio
        if "volume" in cols_lower:
            vol_s = df[cols_lower["volume"]].astype(float).fillna(0.0)
            # NIFTY index history contains long periods without meaningful index
            # volume. Do not turn missing market-volume semantics into a signal.
            if float((vol_s == 0).mean()) > 0.20:
                volume_ratio = pd.Series(0.0, index=df.index)
            else:
                vol_sma = vol_s.rolling(window=20, min_periods=5).mean().fillna(1.0)
                volume_ratio = np.log((vol_s + 1.0) / (vol_sma + 1.0))
        else:
            volume_ratio = pd.Series(0.0, index=df.index)

        # 4. 20-day Price Momentum
        mom_20d = ((close_s - close_s.shift(20)) / (close_s.shift(20) + 1e-8)).fillna(0.0)

        features = pd.DataFrame(
            {
                "log_return": log_ret,
                "volatility_20d": vol_20d,
                "volume_ratio": volume_ratio,
                "momentum_20d": mom_20d,
            },
            index=df.index,
        )
        return features.ffill().bfill().fillna(0.0)

    def _compute_canonical_mapping(
        self, X_scaled: np.ndarray, raw_posteriors: np.ndarray, raw_returns: np.ndarray
    ) -> None:
        """Sort raw hidden states into canonical order: 0=Bear, 1=Sideways, 2=Bull."""
        n_components = self.n_states
        # Compute empirical mean return under each raw state
        raw_state_returns = np.zeros(n_components)
        for k in range(n_components):
            weights = raw_posteriors[:, k]
            weight_sum = np.sum(weights)
            if weight_sum > 1e-6:
                raw_state_returns[k] = np.sum(weights * raw_returns) / weight_sum
            else:
                raw_state_returns[k] = float(self.model.means_[k, 0])

        # State with lowest return is Bear (0), highest is Bull (2), middle is Sideways (1)
        sorted_indices = np.argsort(raw_state_returns)  # [lowest, ..., highest]
        self.canonical_to_raw_ = sorted_indices  # canonical_to_raw_[0] = raw index of Bear
        self.raw_to_canonical_ = np.zeros(n_components, dtype=int)
        for can_idx, raw_idx in enumerate(sorted_indices):
            self.raw_to_canonical_[raw_idx] = can_idx

    def _compute_stationary_distribution(self, trans_mat: np.ndarray) -> np.ndarray:
        """Compute the stationary distribution pi satisfying pi * P = pi and sum(pi) = 1."""
        n = trans_mat.shape[0]
        try:
            eigvals, eigvecs = np.linalg.eig(trans_mat.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            stationary = np.real(eigvecs[:, idx])
            if np.all(stationary <= 0):
                stationary = -stationary
            stationary = np.maximum(stationary, 0.0)
            sum_stat = np.sum(stationary)
            if sum_stat > 1e-8:
                stationary = stationary / sum_stat
            else:
                stationary = np.ones(n) / n
        except Exception:
            stationary = np.ones(n) / n
        return stationary

    def _compute_expected_durations(self, trans_mat: np.ndarray) -> dict[str, float]:
        """Compute expected dwell duration in trading days: E[D_i] = 1 / (1 - A_ii)."""
        durations = {}
        for i, name in enumerate(CANONICAL_REGIME_NAMES):
            diag = float(trans_mat[i, i])
            exit_prob = max(1.0 - diag, 1e-4)
            durations[name] = float(round(1.0 / exit_prob, 2))
        return durations

    def fit(self, df: pd.DataFrame) -> GaussianHMMRegimeDetector:
        """Fit Gaussian HMM on market features and compute canonical state parameters."""
        if len(df) < self.n_states:
            raise ValueError(f"Need at least {self.n_states} non-null rows to fit HMM.")

        features_df = self.engineer_features(df)
        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            min_covar=self.min_covar,
        )
        self.model.fit(X_scaled)

        # Raw state posteriors
        raw_posteriors = self.model.predict_proba(X_scaled)

        # Primary return signal for economic state ordering
        if "log_return" in features_df.columns:
            raw_returns = features_df["log_return"].values
        elif "returns" in features_df.columns:
            raw_returns = features_df["returns"].values
        else:
            raw_returns = X[:, 0]

        # Compute canonical mapping
        self._compute_canonical_mapping(X_scaled, raw_posteriors, raw_returns)

        # Permute transition matrix to canonical order (0=Bear, 1=Sideways, 2=Bull)
        P = self.canonical_to_raw_
        raw_trans = self.model.transmat_
        canonical_trans = raw_trans[P, :][:, P]
        # Re-normalize rows to ensure exact stochasticity
        row_sums = canonical_trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix_ = canonical_trans / row_sums

        # Stationary distribution
        self.stationary_distribution_ = self._compute_stationary_distribution(self.transition_matrix_)

        # Canonical means & covariances
        self.means_ = self.model.means_[P]
        if self.covariance_type == "full":
            self.covars_ = self.model.covars_[P]
        elif self.covariance_type == "diag":
            self.covars_ = self.model.covars_[P]

        # Expected durations
        self.expected_durations_ = self._compute_expected_durations(self.transition_matrix_)
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict canonical regime IDs (0=Bear, 1=Sideways, 2=Bull) using Viterbi decoding."""
        if not self.is_fitted_ or self.model is None:
            self.fit(df)
        features_df = self.engineer_features(df)
        X_scaled = self.scaler.transform(features_df.values)
        raw_states = self.model.predict(X_scaled)
        return self.raw_to_canonical_[raw_states]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Compute state posteriors [P(Bear), P(Sideways), P(Bull)] for each bar."""
        if not self.is_fitted_ or self.model is None:
            self.fit(df)
        features_df = self.engineer_features(df)
        X_scaled = self.scaler.transform(features_df.values)
        raw_posteriors = self.model.predict_proba(X_scaled)
        P = self.canonical_to_raw_
        canonical_posteriors = raw_posteriors[:, P]
        row_sums = canonical_posteriors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return canonical_posteriors / row_sums

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit model and return annotated DataFrame with canonical regimes and posterior probabilities."""
        self.fit(df)
        regime_ids = self.predict(df)
        posteriors = self.predict_proba(df)

        labeled_df = df.copy()
        labeled_df["regime_id"] = pd.Series(regime_ids, index=df.index, dtype="int64")
        labeled_df["regime"] = pd.Series(
            [CANONICAL_REGIME_NAMES[i] for i in regime_ids], index=df.index, dtype="string"
        )
        labeled_df["prob_bear"] = pd.Series(posteriors[:, 0], index=df.index, dtype="float64")
        labeled_df["prob_sideways"] = pd.Series(posteriors[:, 1], index=df.index, dtype="float64")
        labeled_df["prob_bull"] = pd.Series(posteriors[:, 2], index=df.index, dtype="float64")
        return labeled_df

    def get_current_regime(self, df: pd.DataFrame, window: int = 60) -> str:
        """Determine current market regime from recent lookback window."""
        if not self.is_fitted_:
            self.fit(df)
        recent_df = df.tail(window)
        if recent_df.empty:
            raise ValueError("Not enough data to classify current regime.")
        regime_ids = self.predict(recent_df)
        mode_val = int(pd.Series(regime_ids).mode().iloc[-1])
        return CANONICAL_REGIME_NAMES[mode_val]

    def get_transition_matrix(self) -> np.ndarray:
        """Return (3, 3) canonical transition matrix."""
        if not self.is_fitted_ or self.transition_matrix_ is None:
            raise RuntimeError("Model not fitted.")
        return self.transition_matrix_.copy()

    def get_stationary_distribution(self) -> np.ndarray:
        """Return (3,) stationary distribution vector [w_bear, w_sideways, w_bull]."""
        if not self.is_fitted_ or self.stationary_distribution_ is None:
            raise RuntimeError("Model not fitted.")
        return self.stationary_distribution_.copy()

    def get_expected_durations(self) -> dict[str, float]:
        """Return dictionary of expected dwell duration per regime in trading days."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")
        return self.expected_durations_.copy()

    def get_regime_summary(self) -> dict[str, Any]:
        """Return comprehensive regime summary analytics dictionary."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted.")
        return {
            "regime_names": CANONICAL_REGIME_NAMES,
            "transition_matrix": self.transition_matrix_.tolist(),
            "stationary_distribution": {
                name: float(self.stationary_distribution_[i])
                for i, name in enumerate(CANONICAL_REGIME_NAMES)
            },
            "expected_durations_days": self.expected_durations_,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
        }

    def save(self, model_path: Path | str) -> Path:
        """Serialize model bundle to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "canonical_to_raw_": self.canonical_to_raw_,
            "raw_to_canonical_": self.raw_to_canonical_,
            "transition_matrix_": self.transition_matrix_,
            "stationary_distribution_": self.stationary_distribution_,
            "means_": self.means_,
            "covars_": self.covars_,
            "expected_durations_": self.expected_durations_,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "is_fitted_": self.is_fitted_,
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, model_path: Path | str) -> GaussianHMMRegimeDetector:
        """Deserialize model bundle from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        bundle = joblib.load(path)
        detector = cls(
            n_states=bundle.get("n_states", 3),
            covariance_type=bundle.get("covariance_type", "full"),
        )
        detector.model = bundle["model"]
        detector.scaler = bundle["scaler"]
        detector.canonical_to_raw_ = bundle["canonical_to_raw_"]
        detector.raw_to_canonical_ = bundle["raw_to_canonical_"]
        detector.transition_matrix_ = bundle["transition_matrix_"]
        detector.stationary_distribution_ = bundle["stationary_distribution_"]
        detector.means_ = bundle.get("means_")
        detector.covars_ = bundle.get("covars_")
        detector.expected_durations_ = bundle.get("expected_durations_", {})
        detector.is_fitted_ = bundle.get("is_fitted_", True)
        return detector
