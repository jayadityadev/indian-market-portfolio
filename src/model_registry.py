"""Frozen dataset/model lineage and recommendation state helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


RECOMMENDATION_STATES = {
    "validated_ml",
    "experimental_ml",
    "historical_fallback",
    "unavailable",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def recommendation_state(
    models_dir: Path,
    minimum_macro_f1: float = 0.30,
) -> dict[str, Any]:
    """Return explicit recommendation status from persisted model metadata."""
    model_path = models_dir / "xgboost_recommender.joblib"
    if not model_path.exists():
        return {
            "status": "historical_fallback",
            "source": "historical_sharpe",
            "reason": "No trained XGBoost recommender artifact is available.",
            "model_path": None,
            "cv_metrics": {},
            "target_mode": "unknown",
        }

    try:
        bundle = joblib.load(model_path)
        metrics = bundle.get("cv_metrics_", {}) if isinstance(bundle, dict) else {}
        target_mode = bundle.get("target_mode", "winner") if isinstance(bundle, dict) else "winner"
    except Exception as exc:
        return {
            "status": "historical_fallback",
            "source": "historical_sharpe",
            "reason": f"XGBoost artifact could not be loaded: {exc}",
            "model_path": str(model_path),
            "cv_metrics": {},
            "target_mode": "unknown",
        }

    macro_f1 = float(metrics.get("macro_f1", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))
    majority_accuracy = float(metrics.get("majority_accuracy", 0.0))
    baseline_pass = not majority_accuracy or accuracy >= majority_accuracy
    if macro_f1 >= minimum_macro_f1 and baseline_pass:
        status = "validated_ml"
        reason = (
            f"XGBoost cross-validation Macro F1 passed gate ({macro_f1:.3f}) "
            f"and beat majority baseline ({accuracy:.3f} vs {majority_accuracy:.3f})."
        )
        source = "ml_classifier"
    else:
        status = "historical_fallback"
        if macro_f1 < minimum_macro_f1:
            reason = (
                f"XGBoost cross-validation Macro F1 ({macro_f1:.3f}) is below "
                f"gate ({minimum_macro_f1:.3f})."
            )
        else:
            reason = (
                f"XGBoost accuracy ({accuracy:.3f}) does not beat majority baseline "
                f"({majority_accuracy:.3f})."
            )
        source = "historical_sharpe"
    return {
        "status": status,
        "source": source,
        "reason": reason,
        "model_path": str(model_path),
        "cv_metrics": metrics,
        "target_mode": target_mode,
    }


def build_registry(
    data_dir: Path,
    models_dir: Path,
    minimum_macro_f1: float = 0.30,
) -> dict[str, Any]:
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_names = ("nifty50.parquet", "nifty50_regimes.parquet", "labeled_data.parquet")
    model_names = ("regime_model.pkl", "xgboost_recommender.joblib")
    data_artifacts = {
        name: _artifact_record(data_dir / name) for name in data_names if (data_dir / name).exists()
    }
    model_artifacts = {
        name: _artifact_record(models_dir / name) for name in model_names if (models_dir / name).exists()
    }
    state = recommendation_state(models_dir, minimum_macro_f1=minimum_macro_f1)
    return {
        "registry_version": 1,
        "frozen": True,
        "frozen_at": manifest.get("generated_at"),
        "ticker": manifest.get("ticker"),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset": {
            "start": manifest.get("price_start"),
            "end": manifest.get("price_end"),
            "rows": manifest.get("price_rows"),
            "artifacts": data_artifacts,
        },
        "models": model_artifacts,
        "recommendation": state,
    }


def verify_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    artifact_groups = {
        "dataset": registry.get("dataset", {}).get("artifacts", {}),
        "models": registry.get("models", {}),
    }
    for records in artifact_groups.values():
        for name, record in records.items():
            path = Path(record["path"])
            if not path.exists():
                errors.append(f"missing artifact: {path}")
            elif sha256_file(path) != record["sha256"]:
                errors.append(f"hash mismatch: {path}")
    status = registry.get("recommendation", {}).get("status")
    if status not in RECOMMENDATION_STATES:
        errors.append(f"invalid recommendation state: {status}")
    return errors
