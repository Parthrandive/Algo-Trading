from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.regime.models.pearl_meta import PearlMetaModel
from src.agents.regime.schemas import RegimeState


def _make_train_frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    ret_z = rng.normal(1.2, 0.25, size=rows)
    vol = rng.normal(0.008, 0.001, size=rows)
    macro = np.zeros(rows)
    close_log_return = ret_z * 0.005
    close = 100 * np.exp(np.cumsum(close_log_return))
    return pd.DataFrame(
        {
            "close": close,
            "close_log_return": close_log_return,
            "close_log_return_zscore": ret_z,
            "rolling_vol_20": vol,
            "macro_directional_flag": macro,
        }
    )


def _append_bearish_context(df: pd.DataFrame, rows: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    ret_z = rng.normal(-1.8, 0.35, size=rows)
    vol = rng.normal(0.018, 0.002, size=rows)
    macro = -1.0 * np.ones(rows)
    close_log_return = ret_z * 0.005
    base_close = float(df["close"].iloc[-1])
    close = base_close * np.exp(np.cumsum(close_log_return))
    tail = pd.DataFrame(
        {
            "close": close,
            "close_log_return": close_log_return,
            "close_log_return_zscore": ret_z,
            "rolling_vol_20": vol,
            "macro_directional_flag": macro,
        }
    )
    return pd.concat([df, tail], ignore_index=True)


def test_pearl_adapts_to_recent_context():
    base = _make_train_frame()
    infer = _append_bearish_context(base, rows=50)
    model = PearlMetaModel(adaptation_window=60, max_adaptation_weight=0.75)
    model.fit(base)
    pred = model.predict(infer)

    assert pred.regime_state in {RegimeState.BEAR, RegimeState.CRISIS}
    assert pred.confidence >= 0.3


def test_pearl_probability_contract():
    base = _make_train_frame()
    model = PearlMetaModel(adaptation_window=40)
    pred = model.predict(base)
    probs = pred.probabilities

    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert set(probs.keys()) == {
        RegimeState.BULL.value,
        RegimeState.BEAR.value,
        RegimeState.SIDEWAYS.value,
        RegimeState.CRISIS.value,
        RegimeState.RBI_BAND_TRANSITION.value,
    }
    assert 0.0 <= pred.transition_probability <= 1.0

