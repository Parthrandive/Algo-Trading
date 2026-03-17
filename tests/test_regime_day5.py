from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.regime.models.hmm_regime import HMMRegimeModel


def _historical_break_frame() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2007-01-01", "2021-12-31", freq="D", tz="UTC")
    ret_z = rng.normal(0.15, 0.25, size=len(dates))
    vol = rng.normal(0.008, 0.0015, size=len(dates))
    macro = np.zeros(len(dates))

    break_events = [
        pd.Timestamp("2008-09-15", tz="UTC"),  # GFC
        pd.Timestamp("2013-06-01", tz="UTC"),  # taper tantrum
        pd.Timestamp("2020-03-15", tz="UTC"),  # covid crash
    ]
    for event in break_events:
        mask = (dates >= event - pd.Timedelta(days=20)) & (dates <= event + pd.Timedelta(days=25))
        ret_z[mask] = rng.normal(-3.0, 0.7, size=mask.sum())
        vol[mask] = rng.normal(0.04, 0.007, size=mask.sum())
        macro[mask] = rng.choice([-1.0, 0.0, 1.0], size=mask.sum(), p=[0.35, 0.3, 0.35])

    close_log_return = ret_z * 0.004
    close = 100 * np.exp(np.cumsum(close_log_return))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "close": close,
            "close_log_return": close_log_return,
            "close_log_return_zscore": ret_z,
            "rolling_vol_20": vol,
            "macro_directional_flag": macro,
        }
    )


def test_transition_breaks_detected_near_known_events():
    df = _historical_break_frame()
    model = HMMRegimeModel(n_components=5, random_state=42)
    model.fit(df)
    decoded = pd.Series(model.decode_regimes(df), index=df["timestamp"])
    changes = decoded.ne(decoded.shift(1))

    for event_str in ("2008-09-15", "2013-06-01", "2020-03-15"):
        event = pd.Timestamp(event_str, tz="UTC")
        window = changes.loc[event - pd.Timedelta(days=25) : event + pd.Timedelta(days=25)]
        assert window.sum() >= 1, f"Expected transition near {event_str}"


def test_transition_rate_higher_in_break_windows_than_calm_windows():
    df = _historical_break_frame()
    model = HMMRegimeModel(n_components=5, random_state=91)
    model.fit(df)
    decoded = pd.Series(model.decode_regimes(df), index=df["timestamp"])
    changes = decoded.ne(decoded.shift(1)).astype(float)

    break_windows = [
        ("2008-09-01", "2008-10-15"),
        ("2013-05-15", "2013-07-15"),
        ("2020-03-01", "2020-04-15"),
    ]
    calm_windows = [
        ("2015-01-01", "2015-06-30"),
        ("2017-01-01", "2017-06-30"),
    ]

    break_rate = np.mean([changes.loc[start:end].mean() for start, end in break_windows])
    calm_rate = np.mean([changes.loc[start:end].mean() for start, end in calm_windows])
    assert break_rate > calm_rate
