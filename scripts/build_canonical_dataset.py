"""Fetch and rebuild all derived NIFTY 50 artifacts from one price dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data_pipeline import run_pipeline
from label_generator import generate_labels, validate_and_save
from regime_detector import fit_regimes, fit_regimes_walk_forward
from data_contracts import validate_label_frame, validate_price_frame, validate_regime_frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    data_dir: Path,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    label_step: int = 14,
) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    resolved_end = end_date or (date.today() + timedelta(days=1)).isoformat()
    prices_path = data_dir / "nifty50.parquet"
    regimes_path = data_dir / "nifty50_regimes.parquet"
    labels_path = data_dir / "labeled_data.parquet"

    prices = run_pipeline(
        ticker="^NSEI",
        start=start_date,
        end=resolved_end,
        output_path=prices_path,
    )
    # Persist causal labels for historical analysis; fit full data separately
    # so API can classify the current tail using all information available today.
    regimes = fit_regimes_walk_forward(prices)
    fit_regimes(prices, persist=True)
    regimes.to_parquet(data_dir / "nifty50_regimes.parquet")
    labels = generate_labels(prices, lookback=252, forward=63, step=label_step)
    validate_and_save(labels, labels_path)

    results = [
        validate_price_frame(prices),
        validate_regime_frame(regimes, prices.index),
        validate_label_frame(labels),
    ]
    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": "^NSEI",
        "source": "yfinance",
        "requested_start_date": start_date,
        "requested_end_date_exclusive": resolved_end,
        "price_rows": len(prices),
        "price_start": str(prices.index.min().date()),
        "price_end": str(prices.index.max().date()),
        "regime_model": "GaussianHMM-3-state",
        "label_lookback_days": 252,
        "label_forward_days": 63,
        "label_step_days": label_step,
        "schema_status": "valid" if all(result.valid for result in results) else "invalid",
        "research_validation_status": "pending_walk_forward_strategy_evaluation",
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (prices_path, regimes_path, labels_path)
        },
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--label-step", type=int, default=14)
    args = parser.parse_args()
    manifest = build_dataset(args.data_dir, args.start_date, args.end_date, args.label_step)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["schema_status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
