from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_universe_xgboost import (
    apply_daily_regime_filter_to_labels,
    apply_daily_regime_filter_to_probs,
    build_daily_regime_block_mask,
    resolve_symbol_artifact_dir,
    validate_artifact_contract,
)


def test_resolve_symbol_artifact_dir_rejects_mixed_collision_in_strict_mode(tmp_path: Path):
    base = tmp_path / "cnn_pattern"
    (base / "RELIANCE_NS").mkdir(parents=True)
    (base / "RELIANCE.NS").mkdir(parents=True)

    with pytest.raises(ValueError, match="Mixed artifact collision"):
        resolve_symbol_artifact_dir(base, "RELIANCE.NS", strict=True)


def test_validate_artifact_contract_rejects_interval_and_run_id_mismatch():
    meta = {
        "run_id": "run_a",
        "interval": "1d",
        "symbol_canonical": "RELIANCE.NS",
        "source_script": "train_universe_cnn.py",
        "feature_schema_version": "technical_features_v1",
    }

    ok, reason = validate_artifact_contract(
        symbol="RELIANCE.NS",
        meta=meta,
        strict=True,
        expected_interval="1h",
        expected_run_id="run_b",
    )
    assert not ok
    assert "interval mismatch" in reason or "run_id mismatch" in reason


def test_validate_artifact_contract_accepts_matching_metadata():
    meta = {
        "run_id": "run_123",
        "interval": "1h",
        "symbol_canonical": "RELIANCE.NS",
        "source_script": "train_universe_cnn.py",
        "feature_schema_version": "technical_features_v1",
    }

    ok, reason = validate_artifact_contract(
        symbol="RELIANCE.NS",
        meta=meta,
        strict=True,
        expected_interval="1h",
        expected_run_id="run_123",
    )
    assert ok
    assert reason == ""


def test_daily_regime_filter_blocks_up_when_below_ma200():
    frame = pd.DataFrame(
        {
            "close": [101.0, 99.0, 120.0],
            "daily_ma_200": [100.0, 100.0, 110.0],
        }
    )
    mask = build_daily_regime_block_mask(frame, mode="long_only_above_ma200")
    assert mask.tolist() == [False, True, False]

    preds = np.array([2, 2, 0], dtype=np.int64)
    adjusted = apply_daily_regime_filter_to_labels(preds, mask)
    assert adjusted.tolist() == [2, 1, 0]


def test_daily_regime_filter_adjusts_probabilities_for_blocked_rows():
    probs = np.array(
        [
            [0.2, 0.2, 0.6],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    mask = np.array([False, True], dtype=bool)
    adjusted = apply_daily_regime_filter_to_probs(probs, mask)
    assert np.allclose(adjusted[0], probs[0])
    assert adjusted[1, 2] == pytest.approx(0.0)
    assert adjusted[1, 1] > probs[1, 1]
    assert np.isclose(float(adjusted[1].sum()), 1.0)
