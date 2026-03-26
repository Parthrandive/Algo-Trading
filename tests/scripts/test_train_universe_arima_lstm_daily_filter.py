from __future__ import annotations

import numpy as np

from src.agents.technical.features import apply_daily_trend_confirmation

CLASS_NEUTRAL = 1
CLASS_UP = 2


def test_apply_daily_trend_confirmation_neutralizes_up_without_bullish_trend():
    labels = np.array([CLASS_UP, CLASS_NEUTRAL, CLASS_UP], dtype=np.int64)
    probs = np.array(
        [
            [0.05, 0.15, 0.80],
            [0.20, 0.60, 0.20],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    trend = np.array([0.0, 1.0, 1.0], dtype=np.float64)

    adjusted_labels, adjusted_probs, neutralized_mask = apply_daily_trend_confirmation(labels, probs, trend)

    assert adjusted_labels.tolist() == [CLASS_NEUTRAL, CLASS_NEUTRAL, CLASS_UP]
    assert neutralized_mask.tolist() == [True, False, False]
    assert adjusted_probs[0, CLASS_UP] == 0.0
    assert adjusted_probs[0, CLASS_NEUTRAL] > probs[0, CLASS_NEUTRAL]


def test_apply_daily_trend_confirmation_noops_on_misaligned_lengths():
    labels = np.array([CLASS_UP], dtype=np.int64)
    probs = np.array([[0.1, 0.2, 0.7]], dtype=np.float64)
    trend = np.array([], dtype=np.float64)

    adjusted_labels, adjusted_probs, neutralized_mask = apply_daily_trend_confirmation(labels, probs, trend)

    assert adjusted_labels.tolist() == labels.tolist()
    assert np.array_equal(adjusted_probs, probs)
    assert neutralized_mask.tolist() == [False]
