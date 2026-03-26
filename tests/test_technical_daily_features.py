from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.technical.features import DAILY_FEATURE_COLUMNS, engineer_features


def _sample_daily_bars(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(100.0, 160.0, rows)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["TEST.NS"] * rows,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
        }
    )


def test_engineer_features_adds_daily_columns():
    features = engineer_features(_sample_daily_bars())
    for column in DAILY_FEATURE_COLUMNS:
        assert column in features.columns


def test_daily_ma_50_is_shifted_one_day_to_avoid_lookahead():
    bars = _sample_daily_bars(rows=120)
    features = engineer_features(bars)

    expected = bars["close"].rolling(50, min_periods=50).mean().shift(1)
    mask = expected.notna() & features["daily_ma_50"].notna()

    assert mask.any()
    assert np.allclose(
        features.loc[mask, "daily_ma_50"].to_numpy(dtype=float),
        expected.loc[mask].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0.0,
    )
