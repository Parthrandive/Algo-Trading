from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.regime.models.hmm_regime import HMMRegimeModel
from src.agents.regime.schemas import RegimeState


def _make_day2_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", "2020-10-31", freq="D", tz="UTC")
    ret_z = []
    vol = []
    macro = []

    for ts in dates:
        if ts < pd.Timestamp("2020-03-10", tz="UTC"):
            ret_z.append(rng.normal(1.1, 0.25))
            vol.append(rng.normal(0.007, 0.001))
            macro.append(0)
        elif ts < pd.Timestamp("2020-05-01", tz="UTC"):
            ret_z.append(rng.normal(-3.3, 0.65))
            vol.append(rng.normal(0.045, 0.006))
            macro.append(0)
        elif ts < pd.Timestamp("2020-07-15", tz="UTC"):
            ret_z.append(rng.normal(-1.2, 0.3))
            vol.append(rng.normal(0.015, 0.002))
            macro.append(0)
        else:
            ret_z.append(rng.normal(0.1, 0.2))
            vol.append(rng.normal(0.008, 0.001))
            macro.append(1 if rng.random() < 0.15 else 0)

    ret_z_arr = np.array(ret_z)
    close_log_return = ret_z_arr * 0.005
    close = 100 * np.exp(np.cumsum(close_log_return))

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "RELIANCE.NS",
            "close": close,
            "close_log_return": close_log_return,
            "close_log_return_zscore": ret_z_arr,
            "rolling_vol_20": np.array(vol),
            "macro_directional_flag": np.array(macro, dtype=float),
        }
    )


def test_hmm_baseline_train_and_predict():
    df = _make_day2_frame()
    model = HMMRegimeModel(n_components=4, random_state=7)
    model.fit(df)
    pred = model.predict(df)

    assert pred.regime_state in {
        RegimeState.BULL,
        RegimeState.BEAR,
        RegimeState.SIDEWAYS,
        RegimeState.CRISIS,
        RegimeState.RBI_BAND_TRANSITION,
    }
    assert 0.0 <= pred.transition_probability <= 1.0
    assert 0.0 <= pred.confidence <= 1.0


def test_hmm_transition_detects_covid_break():
    df = _make_day2_frame()
    model = HMMRegimeModel(n_components=4, random_state=11)
    model.fit(df)
    decoded = model.decode_regimes(df)

    break_date = pd.Timestamp("2020-03-15", tz="UTC")
    decoded_series = pd.Series(decoded, index=df["timestamp"])
    changes = decoded_series[decoded_series.ne(decoded_series.shift(1))]
    assert not changes.empty

    in_window = changes.loc[(changes.index >= break_date - pd.Timedelta(days=12)) & (changes.index <= break_date + pd.Timedelta(days=12))]
    assert len(in_window) >= 1

    crisis_window = decoded_series.loc["2020-03-10":"2020-04-15"]
    crisis_ratio = (crisis_window == RegimeState.CRISIS).mean()
    assert crisis_ratio >= 0.5

