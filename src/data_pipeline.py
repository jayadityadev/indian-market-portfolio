"""Data Pipeline Module — yfinance OHLC fetch + feature engineering.

Purpose:
    Handles all data sourcing and feature engineering for the Indian market intelligence platform.
    Fetches historical price data from yfinance and computes technical features used by backtester
    and regime detector.

Inputs:
    - Ticker symbol (str): e.g., '^NSEI' for NIFTY 50
    - Date range (str): YYYY-MM-DD format
    - Optional output path for parquet cache

Outputs:
    - DataFrame with OHLCV + engineered features (returns, volatility, momentum, drawdown)
    - Index: DatetimeIndex
    - Cached to parquet for fast subsequent loads

Key Functions:
    - fetch_data(ticker, start, end) → raw OHLCV DataFrame
    - engineer_features(df) → add technical features
    - run_pipeline(ticker, start, end, output_path) → full end-to-end process
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import time


DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "nifty50.parquet"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download raw OHLCV data from yfinance with retry logic."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if frame.empty:
                raise RuntimeError(f"No data returned for {ticker} between {start} and {end}.")

            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)

            missing_columns = [column for column in PRICE_COLUMNS if column not in frame.columns]
            if missing_columns:
                raise RuntimeError(
                    f"Missing expected columns from yfinance: {', '.join(missing_columns)}"
                )

            frame = frame.loc[:, PRICE_COLUMNS].copy()
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            frame.index.name = "Date"
            frame.columns.name = None
            return frame.dropna(how="any")

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

    raise RuntimeError(
        f"yfinance fetch failed after {MAX_RETRIES} attempts: {last_error}"
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the Day 1 feature set to a clean OHLCV frame."""
    featured = frame.copy()
    featured["returns"] = featured["Close"].pct_change()
    featured["volatility"] = featured["returns"].rolling(window=20, min_periods=20).std()
    featured["momentum"] = featured["Close"].pct_change(periods=60)
    featured["drawdown"] = featured["Close"] / featured["Close"].cummax() - 1.0
    featured = featured.replace([np.inf, -np.inf], np.nan).dropna().copy()
    featured.index.name = "Date"
    featured.columns.name = None
    return featured


def validate_pipeline_output(frame: pd.DataFrame) -> None:
    """Fail fast when the Day 1 contract is violated."""
    expected_columns = PRICE_COLUMNS + ["returns", "volatility", "momentum", "drawdown"]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"Missing expected output columns: {', '.join(missing_columns)}")

    null_counts = frame.isnull().sum()
    if int(null_counts.sum()) != 0:
        raise RuntimeError(f"Pipeline output still has nulls: {null_counts[null_counts > 0].to_dict()}")

    if frame.shape[1] != 9:
        raise RuntimeError(f"Expected 9 columns, got {frame.shape[1]}.")

    if frame["drawdown"].max() > 1e-9:
        raise RuntimeError("Drawdown should never be positive.")


def save_data(frame: pd.DataFrame, output_path: Path | str) -> Path:
    """Write the featured dataset to parquet."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination)
    return destination


def load_data(input_path: Path | str = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Read the featured dataset back from parquet."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Parquet file not found: {source}")

    frame = pd.read_parquet(source)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    frame.columns.name = None
    return frame


def run_pipeline(
    ticker: str = "^NSEI",
    start: str = "2015-01-01",
    end: str = "2024-01-01",
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Fetch, feature-engineer, validate, and persist the Day 1 dataset."""
    raw_frame = fetch_data(ticker=ticker, start=start, end=end)
    featured_frame = engineer_features(raw_frame)
    validate_pipeline_output(featured_frame)
    save_data(featured_frame, output_path)
    return featured_frame