"""Train a staged recommender candidate without touching production artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from models.recommender import XGBoostStrategyRecommender


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("data/nifty50.parquet"))
    parser.add_argument("--regimes", type=Path, default=Path("data/nifty50_regimes.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-mode", choices=("winner", "suitability"), default="suitability")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--purge-window-samples", type=int, default=5)
    parser.add_argument("--embargo-window-samples", type=int, default=1)
    parser.add_argument("--suitability-margin", type=float, default=0.0)
    parser.add_argument("--commission-pct", type=float, default=0.0005)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--utility-metric", choices=("CAGR", "Sharpe", "Sortino", "Calmar"), default="Sharpe")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices.index = pd.to_datetime(prices.index)
    regimes = pd.read_parquet(args.regimes)
    regimes.index = pd.to_datetime(regimes.index)

    recommender = XGBoostStrategyRecommender(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    X, y, utility = recommender.build_training_dataset(
        prices,
        regimes,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct,
        utility_metric=args.utility_metric,
    )
    if args.target_mode == "suitability":
        recommender.fit_suitability(
            X,
            utility,
            cv_splits=args.cv_splits,
            purge_window=args.purge_window_samples,
            embargo_window=args.embargo_window_samples,
            margin=args.suitability_margin,
        )
    else:
        recommender.fit(
            X,
            y,
            cv_splits=args.cv_splits,
            purge_window=args.purge_window_samples,
            embargo_window=args.embargo_window_samples,
        )
    recommender.cv_config_.update(
        {
            "utility_commission_pct": args.commission_pct,
            "utility_slippage_pct": args.slippage_pct,
            "utility_metric": args.utility_metric,
        }
    )

    recommender.save(args.output)
    summary = {
        "artifact": str(args.output),
        "target_mode": recommender.target_mode,
        "rows": len(X),
        "cv_metrics": recommender.cv_metrics_,
        "cv_config": recommender.cv_config_,
        "utility_costs": {
            "commission_pct": args.commission_pct,
            "slippage_pct": args.slippage_pct,
            "utility_metric": args.utility_metric,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
