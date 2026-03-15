from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.regime.models.ood_detector import OODDetector
from src.agents.regime.schemas import RiskLevel


def _normal_frame(rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "close_log_return_zscore": rng.normal(0.0, 0.35, size=rows),
            "rolling_vol_20": rng.normal(0.008, 0.0015, size=rows),
            "macro_directional_flag": rng.normal(0.0, 0.2, size=rows),
        }
    )


def test_ood_normal_window_is_not_flagged():
    frame = _normal_frame()
    detector = OODDetector(
        warning_mahalanobis=3.0,
        alien_mahalanobis=5.0,
        warning_kl=1.5,
        alien_kl=6.0,
    )
    detector.fit(frame)
    result = detector.detect(frame.tail(40))

    assert result.is_warning is False
    assert result.is_alien is False
    assert result.risk_level == RiskLevel.FULL_RISK


def test_ood_warning_and_alien_thresholds():
    frame = _normal_frame()
    detector = OODDetector(
        warning_mahalanobis=3.0,
        alien_mahalanobis=8.0,
        warning_kl=1.2,
        alien_kl=8.0,
    )
    detector.fit(frame)

    warning_frame = frame.tail(40).copy()
    warning_frame.iloc[-1, warning_frame.columns.get_loc("close_log_return_zscore")] = 0.9
    warning_frame.iloc[-1, warning_frame.columns.get_loc("rolling_vol_20")] = 0.011
    warning_result = detector.detect(warning_frame)
    assert warning_result.is_warning is True
    assert warning_result.is_alien is False
    assert warning_result.risk_level == RiskLevel.REDUCED_RISK

    alien_frame = frame.tail(40).copy()
    alien_frame.iloc[-1, alien_frame.columns.get_loc("close_log_return_zscore")] = 8.0
    alien_frame.iloc[-1, alien_frame.columns.get_loc("rolling_vol_20")] = 0.09
    alien_frame.iloc[-1, alien_frame.columns.get_loc("macro_directional_flag")] = 3.0
    alien_result = detector.detect(alien_frame)
    assert alien_result.is_alien is True
    assert alien_result.risk_level == RiskLevel.NEUTRAL_CASH
