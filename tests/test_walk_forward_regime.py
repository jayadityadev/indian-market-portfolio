from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detector import fit_regimes_walk_forward


def test_walk_forward_regimes_mark_causal_source():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, 150)
    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1000,
        },
        index=pd.date_range("2020-01-01", periods=150, freq="D"),
    )
    result = fit_regimes_walk_forward(frame, min_train=60, test_size=15)
    assert result["regime"].notna().all()
    assert set(result["regime_source"].dropna().unique()) == {"in_sample", "walk_forward"}
    assert (result.loc[result["regime_source"] == "walk_forward", ["prob_bear", "prob_sideways", "prob_bull"]].sum(axis=1) - 1).abs().max() < 1e-5
