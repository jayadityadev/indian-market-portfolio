"""Audit persisted market artifacts against canonical data contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_contracts import validate_label_frame, validate_price_frame, validate_regime_frame


def audit_dataset(data_dir: Path) -> dict:
    prices = pd.read_parquet(data_dir / "nifty50.parquet")
    regimes = pd.read_parquet(data_dir / "nifty50_regimes.parquet")
    labels = pd.read_parquet(data_dir / "labeled_data.parquet")
    results = [
        validate_price_frame(prices),
        validate_regime_frame(regimes, prices.index),
        validate_label_frame(labels),
    ]
    return {
        "data_dir": str(data_dir),
        "valid": all(result.valid for result in results),
        "contracts": [
            {
                "name": result.name,
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "facts": result.facts,
            }
            for result in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--strict", action="store_true", help="exit 1 when any contract fails")
    args = parser.parse_args()
    report = audit_dataset(args.data_dir)
    print(json.dumps(report, indent=2, default=str))
    return 1 if args.strict and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
