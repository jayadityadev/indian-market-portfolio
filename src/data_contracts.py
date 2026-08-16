"""Validation contracts for canonical market, regime, and label datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
REGIME_COLUMNS = ("regime_id", "regime", "prob_bear", "prob_sideways", "prob_bull")
LABEL_COLUMNS = ("window_end_date", "strategy", "label")
EXPECTED_STRATEGIES = (
    "Buy & Hold",
    "MA Crossover",
    "RSI",
    "Momentum",
    "Bollinger Bands",
    "Dual Momentum",
)


@dataclass
class ContractResult:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors


def _date_facts(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": int(len(frame)),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "unique_index": bool(frame.index.is_unique),
    }


def validate_price_frame(frame: pd.DataFrame, min_rows: int = 252) -> ContractResult:
    result = ContractResult("prices", facts=_date_facts(frame))
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        result.errors.append(f"missing required columns: {', '.join(missing)}")
        return result
    if not isinstance(frame.index, pd.DatetimeIndex):
        result.errors.append("index must be DatetimeIndex")
    else:
        if not frame.index.is_monotonic_increasing:
            result.errors.append("index must be sorted ascending")
        if not frame.index.is_unique:
            result.errors.append("index must contain unique dates")
    if len(frame) < min_rows:
        result.errors.append(f"requires at least {min_rows} rows, found {len(frame)}")
    if int(frame[list(PRICE_COLUMNS)].isna().sum().sum()) > 0:
        result.errors.append("OHLCV contains null values")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        result.errors.append("OHLC values must be positive")
    if (frame["Volume"] < 0).any():
        result.errors.append("Volume cannot be negative")
    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        result.errors.append("High must be at least Open and Close")
    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        result.errors.append("Low must be at most Open and Close")
    zero_volume = int((frame["Volume"] == 0).sum())
    result.facts["zero_volume_rows"] = zero_volume
    if zero_volume:
        result.warnings.append(f"{zero_volume} rows have zero volume; verify index-data semantics")
    return result


def validate_regime_frame(
    frame: pd.DataFrame, price_index: pd.Index, min_coverage: float = 0.95
) -> ContractResult:
    result = ContractResult("regimes", facts=_date_facts(frame))
    missing = [column for column in REGIME_COLUMNS if column not in frame.columns]
    if missing:
        result.errors.append(f"missing required columns: {', '.join(missing)}")
        return result
    if not isinstance(frame.index, pd.DatetimeIndex):
        result.errors.append("index must be DatetimeIndex")
    overlap = len(frame.index.intersection(price_index))
    coverage = overlap / max(1, len(price_index))
    result.facts.update({"overlap_rows": overlap, "price_coverage": round(coverage, 4)})
    if coverage < min_coverage:
        result.errors.append(
            f"regime coverage {coverage:.1%} is below required {min_coverage:.1%}"
        )
    probability_sum = frame[["prob_bear", "prob_sideways", "prob_bull"]].sum(axis=1)
    if ((probability_sum - 1.0).abs() > 1e-5).any():
        result.errors.append("regime probabilities must sum to 1")
    allowed = {"Bear", "Sideways", "Bull"}
    unknown = sorted(set(frame["regime"].dropna().astype(str)) - allowed)
    if unknown:
        result.errors.append(f"unknown regime labels: {', '.join(unknown)}")
    result.facts["regime_counts"] = frame["regime"].value_counts().astype(int).to_dict()
    if result.facts["regime_counts"].get("Bull", 0) < 30:
        result.warnings.append("Bull regime has fewer than 30 observations")
    return result


def validate_label_frame(frame: pd.DataFrame, min_unique_dates: int = 252) -> ContractResult:
    result = ContractResult("labels", facts={"rows": int(len(frame))})
    missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
    if missing:
        result.errors.append(f"missing required columns: {', '.join(missing)}")
        return result
    dates = pd.to_datetime(frame["window_end_date"], errors="coerce")
    if dates.isna().any():
        result.errors.append("window_end_date contains invalid dates")
    unique_dates = int(dates.nunique())
    result.facts.update(
        {
            "unique_dates": unique_dates,
            "date_start": str(dates.min()),
            "date_end": str(dates.max()),
            "strategy_counts": frame["strategy"].value_counts().astype(int).to_dict(),
            "label_counts": frame["label"].value_counts().astype(int).to_dict(),
        }
    )
    if unique_dates < min_unique_dates:
        result.errors.append(
            f"requires at least {min_unique_dates} unique dates, found {unique_dates}"
        )
    missing_strategies = sorted(set(EXPECTED_STRATEGIES) - set(frame["strategy"].astype(str)))
    if missing_strategies:
        result.warnings.append(
            f"missing expected strategy labels: {', '.join(missing_strategies)}"
        )
    if frame.duplicated(["window_end_date", "strategy"]).any():
        result.errors.append("duplicate date/strategy label pairs")
    return result
