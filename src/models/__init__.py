"""Models package for Indian Market Portfolio Intelligence Platform.

Exports:
- GaussianHMMRegimeDetector: 3-State Gaussian Hidden Markov Model for market regime detection.
- XGBoostStrategyRecommender: 6-Strategy Calibrated XGBoost Recommender Classifier.
- PurgedTimeSeriesSplit: Time-series CV splitter with forward purge and embargo buffers.
- LSTMDNNBenchmarkModel: PyTorch implementation of the IEEE Access 2024 (Alam et al.) hybrid LSTM-DNN.
"""
from __future__ import annotations

from models.regime_detector import (
    GaussianHMMRegimeDetector,
    CANONICAL_REGIME_NAMES,
)
from models.recommender import (
    XGBoostStrategyRecommender,
    PurgedTimeSeriesSplit,
    RecommendationResult,
    STRATEGY_NAMES,
)
from models.lstm_benchmark import (
    LSTMDNNBenchmarkModel,
    evaluate_benchmark_pipeline,
    train_lstm_benchmark,
    TimeSeriesSequenceDataset,
    get_device,
)

__all__ = [
    "GaussianHMMRegimeDetector",
    "CANONICAL_REGIME_NAMES",
    "XGBoostStrategyRecommender",
    "PurgedTimeSeriesSplit",
    "RecommendationResult",
    "STRATEGY_NAMES",
    "LSTMDNNBenchmarkModel",
    "evaluate_benchmark_pipeline",
    "train_lstm_benchmark",
    "TimeSeriesSequenceDataset",
    "get_device",
]
