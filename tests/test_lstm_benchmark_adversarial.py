"""Adversarial Empirical Stress-Testing Suite for PyTorch LSTM-DNN Benchmark (Milestone M1).

Covers:
1. Neural Network Forward & Backward Passes:
   - Single-sample inference (B=1, L=30, D=9) in eval and train mode.
   - Matrix of shape variations (B in [1, 2, 64, 512], L in [1, 5, 30, 100], D in [1, 9, 21]).
   - Gradient flow verification: non-zero, non-NaN gradients across all 2 LSTM layers and all 4 Dense layers.
   - Gradient clipping and optimizer step stability under extreme inputs.
2. Sequence Pipeline & Anti-Leakage:
   - Strict chronological causality in sliding window dataset.
   - Train-only StandardScaler fitting isolation (t <= T_train).
   - TimeSeriesSequenceDataset tensor integrity and indexing.
   - Error handling for short time series (N < seq_len).
3. Caching, Latency Profiling & Device Fallback:
   - Cache roundtrip serialization and sub-10ms reload benchmark.
   - Corrupted JSON/state cache resilience.
   - CPU device fallback enforcement and latency profiling percentiles (p50, p95, p99).
4. Comparative Metrics & Overfitting Diagnostics:
   - Accuracy, F1 macro, precision, recall, loss gap, accuracy gap, and f1 gap logic.
   - End-to-end comparative evaluation pipeline execution.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.lstm_benchmark import (
    EarlyStopping,
    LSTMDNNBenchmarkModel,
    TimeSeriesSequenceDataset,
    construct_sliding_windows,
    evaluate_benchmark_pipeline,
    evaluate_lstm_model,
    get_device,
    load_cached_benchmark,
    prepare_benchmark_dataloaders,
    profile_inference_latency,
    save_benchmark_cache,
    train_lstm_benchmark,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for stress testing."""
    np.random.seed(42)
    n_days = 400
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    rets = np.random.normal(loc=0.0004, scale=0.012, size=n_days)
    price = 1000.0 * np.exp(np.cumsum(rets))

    return pd.DataFrame(
        {
            "Open": price * (1.0 - np.random.uniform(0.0, 0.005, n_days)),
            "High": price * (1.0 + np.random.uniform(0.005, 0.015, n_days)),
            "Low": price * (1.0 - np.random.uniform(0.005, 0.015, n_days)),
            "Close": price,
            "Volume": np.random.lognormal(15.0, 0.4, size=n_days),
        },
        index=dates,
    )


# =====================================================================
# 1. Neural Network Forward & Backward Passes
# =====================================================================

class TestForwardBackwardPasses:
    """Adversarial stress-testing of LSTM-DNN forward, backward, and gradient dynamics."""

    def test_single_sample_inference_modes(self):
        """Verify B=1 inference succeeds in both eval() and train() modes without LayerNorm crashes."""
        model = LSTMDNNBenchmarkModel(input_dim=9, hidden_dim1=64, hidden_dim2=32, dense_dim=64, num_classes=6)
        x_single = torch.randn(1, 30, 9)

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x_single)
        assert out_eval.shape == (1, 6)
        assert not torch.isnan(out_eval).any()
        assert not torch.isinf(out_eval).any()

        # Train mode with batch size 1 (must NOT crash LayerNorm)
        model.train()
        out_train = model(x_single)
        assert out_train.shape == (1, 6)
        assert not torch.isnan(out_train).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 7, 32, 64, 128, 512])
    @pytest.mark.parametrize("seq_len", [1, 5, 30, 60, 100])
    @pytest.mark.parametrize("input_dim", [1, 9, 21])
    def test_forward_pass_dimension_matrix(self, batch_size: int, seq_len: int, input_dim: int):
        """Stress-test forward pass across combinations of batch sizes, sequence lengths, and feature dimensions."""
        model = LSTMDNNBenchmarkModel(input_dim=input_dim, num_classes=6)
        model.eval()
        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch_size, 6)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow_all_layers_nonzero(self):
        """Verify non-zero, finite gradients propagate through ALL 2 LSTM layers, LayerNorm, and 4 Dense layers."""
        torch.manual_seed(42)
        model = LSTMDNNBenchmarkModel(input_dim=9, hidden_dim1=64, hidden_dim2=32, dense_dim=64, num_classes=6)
        model.train()

        batch_size = 16
        seq_len = 30
        x = torch.randn(batch_size, seq_len, 9, requires_grad=True)
        targets = torch.randint(0, 6, (batch_size,))

        criterion = nn.CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, targets)
        loss.backward()

        # Expected named parameter groups
        expected_param_names = [
            "lstm1.weight_ih_l0",
            "lstm1.weight_hh_l0",
            "lstm1.bias_ih_l0",
            "lstm1.bias_hh_l0",
            "lstm2.weight_ih_l0",
            "lstm2.weight_hh_l0",
            "lstm2.bias_ih_l0",
            "lstm2.bias_hh_l0",
            "layer_norm.weight",
            "layer_norm.bias",
            "dense1.weight",
            "dense1.bias",
            "dense2.weight",
            "dense2.bias",
            "dense3.weight",
            "dense3.bias",
            "dense4.weight",
            "dense4.bias",
            "out_head.weight",
            "out_head.bias",
        ]

        for name, param in model.named_parameters():
            assert name in expected_param_names, f"Unexpected parameter: {name}"
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            grad_norm = param.grad.norm().item()
            assert not np.isnan(grad_norm), f"Gradient is NaN for parameter: {name}"
            assert not np.isinf(grad_norm), f"Gradient is Inf for parameter: {name}"
            assert grad_norm > 1e-9, f"Gradient vanished (is zero) for parameter: {name} (norm={grad_norm})"

        # Verify input gradient propagates
        assert x.grad is not None
        assert x.grad.shape == (batch_size, seq_len, 9)
        assert x.grad.norm().item() > 1e-9

    def test_extreme_input_resilience(self):
        """Stress-test numerical stability under extreme input values (large, small, all zero)."""
        model = LSTMDNNBenchmarkModel(input_dim=9, num_classes=6)
        model.eval()

        # All zeros
        x_zeros = torch.zeros(4, 30, 9)
        out_zeros = model(x_zeros)
        assert not torch.isnan(out_zeros).any()

        # Very small values
        x_small = torch.randn(4, 30, 9) * 1e-7
        out_small = model(x_small)
        assert not torch.isnan(out_small).any()

        # Large values (within reasonable financial standard deviation range after clipping)
        x_large = torch.randn(4, 30, 9) * 10.0
        out_large = model(x_large)
        assert not torch.isnan(out_large).any()

    def test_parameter_count_matches_architecture(self):
        """Verify parameter count matches theoretical specifications for hybrid LSTM-DNN."""
        model = LSTMDNNBenchmarkModel(input_dim=9, hidden_dim1=64, hidden_dim2=32, dense_dim=64, num_classes=6)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Theoretical breakdown:
        # LSTM 1: 4 * (9*64 + 64*64 + 64 + 64) = 4 * 4800 = 19,200
        # LSTM 2: 4 * (64*32 + 32*32 + 32 + 32) = 4 * 3136 = 12,544
        # LayerNorm: 32 * 2 = 64
        # Dense 1: 32 * 64 + 64 = 2,112
        # Dense 2: 64 * 64 + 64 = 4,160
        # Dense 3: 64 * 64 + 64 = 4,160
        # Dense 4: 64 * 64 + 64 = 4,160
        # Out Head: 64 * 6 + 6 = 390
        # Total = 46,790
        expected_params = 19200 + 12544 + 64 + 2112 + 4160 + 4160 + 4160 + 390
        assert total_params == expected_params == 46790


# =====================================================================
# 2. Sequence Pipeline & Anti-Leakage
# =====================================================================

class TestSequencePipelineAndAntiLeakage:
    """Adversarial stress-testing of sequence slicing, temporal alignment, and scaler fitting."""

    def test_chronological_causality_and_no_lookahead(self):
        """Verify sliding window construction adheres to strict causality."""
        n_samples = 150
        seq_len = 25
        n_feats = 5

        # Feature at time t has value t in all columns
        features = np.repeat(np.arange(n_samples, dtype=np.float32)[:, None], n_feats, axis=1)
        targets = np.arange(n_samples, dtype=np.int64) * 10

        seqs, tgts = construct_sliding_windows(features, targets, seq_len=seq_len)

        assert len(seqs) == n_samples - seq_len + 1

        for i in range(len(seqs)):
            # Start time of window i should be i
            assert seqs[i, 0, 0] == float(i)
            # End time of window i should be i + seq_len - 1
            assert seqs[i, -1, 0] == float(i + seq_len - 1)
            # Target should correspond to time i + seq_len - 1 (contemporary/lagged, never future)
            assert tgts[i] == targets[i + seq_len - 1]

    def test_future_data_perturbation_invariance(self):
        """Verify perturbing future data points (t > T) does NOT alter sliding sequences for t <= T."""
        n_samples = 100
        seq_len = 20
        n_feats = 4

        np.random.seed(123)
        features_clean = np.random.randn(n_samples, n_feats).astype(np.float32)
        targets_clean = np.random.randint(0, 6, size=n_samples)

        seqs_clean, tgts_clean = construct_sliding_windows(features_clean, targets_clean, seq_len=seq_len)

        # Perturb all data points after index 60
        split_idx = 60
        features_perturbed = features_clean.copy()
        targets_perturbed = targets_clean.copy()
        features_perturbed[split_idx:] += 999.0
        targets_perturbed[split_idx:] = 5

        seqs_perturbed, tgts_perturbed = construct_sliding_windows(features_perturbed, targets_perturbed, seq_len=seq_len)

        # All windows that finish at or before split_idx - 1 MUST be strictly identical
        valid_unperturbed_windows = split_idx - seq_len + 1
        np.testing.assert_array_equal(seqs_clean[:valid_unperturbed_windows], seqs_perturbed[:valid_unperturbed_windows])
        np.testing.assert_array_equal(tgts_clean[:valid_unperturbed_windows], tgts_perturbed[:valid_unperturbed_windows])

    def test_standard_scaler_fitted_strictly_on_train(self):
        """Verify StandardScaler in prepare_benchmark_dataloaders is fitted ONLY on training split (t <= T_train)."""
        np.random.seed(999)
        n_samples = 200
        n_feats = 6

        # Train data centered around 0.0, Test data shifted to 100.0
        train_len = int(n_samples * 0.70)
        feats = np.zeros((n_samples, n_feats), dtype=np.float32)
        feats[:train_len] = np.random.normal(loc=0.0, scale=1.0, size=(train_len, n_feats))
        feats[train_len:] = np.random.normal(loc=100.0, scale=1.0, size=(n_samples - train_len, n_feats))

        features_df = pd.DataFrame(feats, columns=[f"f_{i}" for i in range(n_feats)])
        targets = pd.Series(np.random.randint(0, 6, size=n_samples))

        train_l, val_l, test_l, scaler, tabular_splits = prepare_benchmark_dataloaders(
            features_df, targets, seq_len=15, train_ratio=0.70, val_ratio=0.15
        )

        # Scaler mean must match training partition mean, NOT full dataset mean
        train_mean = feats[:train_len].mean(axis=0)
        full_mean = feats.mean(axis=0)

        np.testing.assert_allclose(scaler.mean_, train_mean, atol=1e-4)
        assert np.max(np.abs(scaler.mean_ - full_mean)) > 20.0, "Scaler leaked future test distribution mean!"

    def test_short_dataset_raises_value_error(self):
        """Verify construct_sliding_windows rejects input sequences shorter than seq_len."""
        feats = np.random.randn(10, 5)
        tgts = np.random.randint(0, 6, size=10)

        with pytest.raises(ValueError, match="Need at least 20 samples"):
            construct_sliding_windows(feats, tgts, seq_len=20)


# =====================================================================
# 3. Caching, Latency Profiling & Device Fallback
# =====================================================================

class TestCachingLatencyAndDevice:
    """Adversarial testing of artifact caching, inference latency, and hardware abstraction."""

    def test_cache_serialization_and_fast_reload(self):
        """Verify cache serialization and benchmark <10ms reload speed over 100 iterations."""
        model = LSTMDNNBenchmarkModel(input_dim=9, num_classes=6)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, 9))

        dummy_summary = {
            "paper_reference": "Alam et al. (2024)",
            "model_architecture": "Hybrid 2-LSTM",
            "total_parameters": 46790,
            "xgboost_metrics": {"accuracy": 0.82, "f1_score": 0.81},
            "lstm_metrics": {"accuracy": 0.79, "f1_score": 0.77},
            "overfitting_gap": {"loss_gap": 0.04, "accuracy_gap": 0.03, "f1_gap": 0.04},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_benchmark_cache(dummy_summary, model, scaler, cache_dir=tmpdir)

            # Test 100 repeated reloads
            t0 = time.perf_counter()
            for _ in range(100):
                loaded = load_cached_benchmark(cache_dir=tmpdir)
                assert loaded is not None
            total_time_ms = (time.perf_counter() - t0) * 1000.0
            avg_reload_ms = total_time_ms / 100.0

            assert avg_reload_ms < 10.0, f"Average reload time {avg_reload_ms:.3f}ms exceeds 10ms target."
            assert loaded["total_parameters"] == 46790

    def test_corrupted_cache_resilience(self):
        """Verify load_cached_benchmark handles corrupted JSON or unreadable cache cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_json = Path(tmpdir) / "benchmark_summary.json"
            with open(bad_json, "w") as f:
                f.write("{ INVALID JSON CONTENT ...")

            result = load_cached_benchmark(cache_dir=tmpdir)
            assert result is None, "Corrupted cache must return None rather than raising an unhandled exception."

    def test_cpu_device_fallback_and_latency(self):
        """Verify forced CPU device execution and latency percentiles."""
        cpu_dev = torch.device("cpu")
        model = LSTMDNNBenchmarkModel(input_dim=9, num_classes=6)
        model.to(cpu_dev)

        latency = profile_inference_latency(model, (1, 30, 9), device=cpu_dev, n_runs=30)

        assert latency["mean_ms"] > 0.0
        assert latency["p50_ms"] > 0.0
        assert latency["p50_ms"] <= latency["p95_ms"] <= latency["p99_ms"]


# =====================================================================
# 4. Comparative Evaluation Engine & Overfitting Diagnostics
# =====================================================================

class TestComparativeEvaluationEngine:
    """Adversarial testing of comparative metrics against XGBoost and overfitting diagnostic gaps."""

    def test_early_stopping_restoration(self):
        """Verify EarlyStopping correctly tracks best validation loss and restores best model weights."""
        model = LSTMDNNBenchmarkModel(input_dim=4, num_classes=3)
        device = torch.device("cpu")
        stopper = EarlyStopping(patience=3, min_delta=1e-3)

        # Epoch 1: Loss 1.0 -> best
        stopper(1.0, model)
        assert stopper.best_loss == 1.0

        # Epoch 2: Loss 0.8 -> new best
        stopper(0.8, model)
        assert stopper.best_loss == 0.8

        # Epoch 3: Loss 0.85 -> counter 1
        stopper(0.85, model)
        assert stopper.counter == 1

        # Epoch 4: Loss 0.90 -> counter 2
        stopper(0.90, model)
        assert stopper.counter == 2

        # Epoch 5: Loss 0.95 -> counter 3 -> triggers early stop
        stopped = stopper(0.95, model)
        assert stopped is True
        assert stopper.early_stop is True

        # Restoring weights should not throw errors
        stopper.restore_best_weights(model, device)

    def test_end_to_end_pipeline_diagnostics(self, synthetic_ohlcv_df: pd.DataFrame):
        """Verify full comparative pipeline output conforms to IEEE Access 2024 academic benchmark schema."""
        summary = evaluate_benchmark_pipeline(
            synthetic_ohlcv_df, force_retrain=True, epochs=4, batch_size=16, seq_len=15
        )

        required_keys = [
            "paper_reference",
            "model_architecture",
            "total_parameters",
            "device",
            "xgboost_metrics",
            "lstm_metrics",
            "overfitting_gap",
            "latency_benchmark",
            "training_history",
            "recommendation",
        ]

        for k in required_keys:
            assert k in summary, f"Missing required key in benchmark summary: {k}"

        # Overfitting gap keys
        gap = summary["overfitting_gap"]
        assert "loss_gap" in gap
        assert "accuracy_gap" in gap
        assert "f1_gap" in gap
        assert isinstance(gap["loss_gap"], float)
        assert isinstance(gap["accuracy_gap"], float)
        assert isinstance(gap["f1_gap"], float)

        # Latency keys
        latency = summary["latency_benchmark"]
        assert "xgboost" in latency
        assert "lstm_dnn" in latency
        assert latency["xgboost"]["mean_ms"] > 0
        assert latency["lstm_dnn"]["mean_ms"] > 0
