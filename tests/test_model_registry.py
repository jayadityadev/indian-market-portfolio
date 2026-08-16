from __future__ import annotations

import json
from pathlib import Path

from model_registry import build_registry, verify_registry


def test_registry_tracks_current_artifact_hashes(tmp_path: Path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    for name in ("nifty50.parquet", "nifty50_regimes.parquet", "labeled_data.parquet"):
        (data_dir / name).write_bytes(name.encode())
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps({"generated_at": "2026-01-01", "ticker": "^NSEI", "price_start": "2020", "price_end": "2021", "price_rows": 10}),
        encoding="utf-8",
    )
    (models_dir / "regime_model.pkl").write_bytes(b"regime")
    registry = build_registry(data_dir, models_dir)
    assert registry["frozen"] is True
    assert registry["recommendation"]["status"] == "historical_fallback"
    assert verify_registry(registry) == []


def test_registry_detects_artifact_mutation(tmp_path: Path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    for name in ("nifty50.parquet", "nifty50_regimes.parquet", "labeled_data.parquet"):
        (data_dir / name).write_bytes(b"artifact")
    (data_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    registry = build_registry(data_dir, models_dir)
    (data_dir / "nifty50.parquet").write_bytes(b"changed")
    errors = verify_registry(registry)
    assert any("hash mismatch" in error for error in errors)
