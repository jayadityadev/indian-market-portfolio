"""Unit and integration tests for PyTorch LSTM-DNN Academic Benchmark."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.lstm_benchmark import (
    EarlyStopping,
    LSTMDNNBenchmarkModel,
    construct_sliding_windows,
    evaluate_benchmark_pipeline,
    get_device,
    load_cached_benchmark,
    prepare_benchmark_dataloaders,
    profile_inference_latency,
    save_benchmark_cache,
    train_lstm_benchmark,
)


@pytest.fixture
def synthetic_benchmark_data() -> pd.DataFrame:
    """Generate synthetic historical price data for benchmark training."""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

    rets = np.random.normal(loc=0.0005, scale=0.015, size=n_days)
    price = 100.0 * np.exp(np.cumsum(rets))

    return pd.DataFrame(
        {
            "Open": price * 0.999,
            "High": price * 1.008,
            "Low": price * 0.992,
            "Close": price,
            "Volume": np.random.lognormal(14.0, 0.5, size=n_days),
        },
        index=dates,
    )


def test_lstm_dnn_model_shapes():
    """Verify LSTM-DNN forward pass tensor shapes and gradient backpropagation."""
    batch_size = 16
    seq_len = 30
    input_dim = 9
    num_classes = 6

    model = LSTMDNNBenchmarkModel(input_dim=input_dim, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

    logits = model(dummy_input)
    assert logits.shape == (batch_size, num_classes)

    loss = logits.sum()
    loss.backward()
    assert dummy_input.grad is not None
    assert dummy_input.grad.shape == dummy_input.shape


def test_single_sample_inference():
    """Verify single-sample inference (B=1) succeeds without normalization failures."""
    model = LSTMDNNBenchmarkModel(input_dim=9, num_classes=6)
    model.eval()
    dummy_single = torch.randn(1, 30, 9)

    with torch.no_grad():
        out = model(dummy_single)
    assert out.shape == (1, 6)


def test_sliding_window_dataset_no_leakage():
    """Verify sliding window construction enforces strict causality without future leakage."""
    n_samples = 100
    seq_len = 20
    n_feats = 4
    feats = np.arange(n_samples * n_feats, dtype=np.float32).reshape(n_samples, n_feats)
    targets = np.arange(n_samples, dtype=np.int64)

    seqs, tgts = construct_sliding_windows(feats, targets, seq_len=seq_len)

    assert len(seqs) == n_samples - seq_len + 1
    assert seqs.shape == (n_samples - seq_len + 1, seq_len, n_feats)
    # Check first sequence ends at time seq_len - 1
    np.testing.assert_array_equal(seqs[0][-1], feats[seq_len - 1])
    assert tgts[0] == targets[seq_len - 1]


def test_benchmark_partitions_have_disjoint_target_ranges():
    """Validation/test windows may use past context but never future targets."""
    features = pd.DataFrame(np.arange(100 * 2, dtype=np.float32).reshape(100, 2))
    targets = np.arange(100, dtype=np.int64)
    train, val, test, _, _ = prepare_benchmark_dataloaders(
        features, targets, seq_len=10, batch_size=16
    )
    train_targets = train.dataset.targets.numpy()
    val_targets = val.dataset.targets.numpy()
    test_targets = test.dataset.targets.numpy()
    assert train_targets.max() < val_targets.min()
    assert val_targets.max() < test_targets.min()


def test_early_stopping():
    """Verify EarlyStopping stops training when validation loss fails to improve."""
    model = LSTMDNNBenchmarkModel(input_dim=4, num_classes=3)
    early_stopping = EarlyStopping(patience=3, min_delta=1e-3)

    # Simulated monotonically increasing validation loss
    assert not early_stopping(1.0, model)
    assert not early_stopping(1.1, model)
    assert not early_stopping(1.2, model)
    assert early_stopping(1.3, model)  # Counter reached patience


def test_latency_profiler():
    """Verify profile_inference_latency computes valid timing metrics in milliseconds."""
    model = LSTMDNNBenchmarkModel(input_dim=6, num_classes=6)
    device = torch.device("cpu")
    latency = profile_inference_latency(model, (1, 30, 6), device=device, n_runs=10)

    assert "mean_ms" in latency
    assert "p50_ms" in latency
    assert "p95_ms" in latency
    assert "p99_ms" in latency
    assert latency["mean_ms"] > 0.0


def test_device_fallback():
    """Verify get_device returns a valid compute device."""
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_cache_save_and_load():
    """Verify benchmark cache serialization and deserialization."""
    summary_dummy = {
        "paper_reference": "Alam et al. (2024)",
        "xgboost_metrics": {"accuracy": 0.8},
        "lstm_metrics": {"accuracy": 0.75},
    }
    model = LSTMDNNBenchmarkModel(input_dim=5, num_classes=6)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.random.randn(50, 5))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_benchmark_cache(summary_dummy, model, scaler, cache_dir=tmpdir)
        loaded = load_cached_benchmark(cache_dir=tmpdir)
        assert loaded is not None
        assert loaded["paper_reference"] == summary_dummy["paper_reference"]
        assert (Path(tmpdir) / "lstm_dnn_model.pt").exists()
        assert (Path(tmpdir) / "lstm_scaler.pkl").exists()


def test_benchmark_pipeline_end_to_end(synthetic_benchmark_data: pd.DataFrame):
    """Verify end-to-end benchmark comparison pipeline between XGBoost and LSTM-DNN."""
    result = evaluate_benchmark_pipeline(
        synthetic_benchmark_data, force_retrain=True, epochs=5, batch_size=16, seq_len=15
    )

    assert "paper_reference" in result
    assert "xgboost_metrics" in result
    assert "lstm_metrics" in result
    assert "overfitting_gap" in result
    assert "latency_benchmark" in result
    assert "training_history" in result
    assert len(result["training_history"]) > 0

    assert "accuracy" in result["xgboost_metrics"]
    assert "accuracy" in result["lstm_metrics"]
    assert "loss_gap" in result["overfitting_gap"]
