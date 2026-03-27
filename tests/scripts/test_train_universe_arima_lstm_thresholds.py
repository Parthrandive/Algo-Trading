from __future__ import annotations

import numpy as np

from src.agents.technical.thresholding import choose_symbol_threshold


def test_choose_symbol_threshold_can_select_below_005_when_allowed():
    train_scores = np.array([0.0012, -0.0013, 0.0004, -0.0003, 0.0022, -0.0025], dtype=np.float64)
    train_actual_returns = np.array([0.0014, -0.0011, 0.0002, -0.0001, 0.0021, -0.0021], dtype=np.float64)

    threshold, neutral_ratio = choose_symbol_threshold(
        train_scores=train_scores,
        train_actual_returns=train_actual_returns,
        requested_threshold=0.005,
        class_threshold_min=0.001,
        min_neutral_ratio=0.10,
        max_neutral_ratio=0.90,
        target_neutral_ratio=0.30,
    )

    assert threshold < 0.005
    assert threshold >= 0.001
    assert 0.0 <= neutral_ratio <= 1.0


def test_choose_symbol_threshold_respects_min_floor():
    train_scores = np.array([0.0012, -0.0013, 0.0004, -0.0003, 0.0022, -0.0025], dtype=np.float64)
    train_actual_returns = np.array([0.0014, -0.0011, 0.0002, -0.0001, 0.0021, -0.0021], dtype=np.float64)

    threshold, _ = choose_symbol_threshold(
        train_scores=train_scores,
        train_actual_returns=train_actual_returns,
        requested_threshold=0.005,
        class_threshold_min=0.002,
        min_neutral_ratio=0.10,
        max_neutral_ratio=0.90,
        target_neutral_ratio=0.30,
    )

    assert threshold >= 0.002
