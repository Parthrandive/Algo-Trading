from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train_xgboost_meta_layer import (
    apply_confidence_threshold,
    compute_backtest_metrics,
    generate_weight_grid,
    scores_to_soft_probs,
    validate_symbol_artifact,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ok")


def test_generate_weight_grid_respects_constraints():
    weights = generate_weight_grid(min_weight=0.1, max_weight=0.7, step=0.1)
    assert weights, "expected at least one weight combination"
    for w_cnn, w_lstm, w_xgb in weights:
        assert 0.1 <= w_cnn <= 0.7
        assert 0.1 <= w_lstm <= 0.7
        assert 0.1 <= w_xgb <= 0.7
        assert abs((w_cnn + w_lstm + w_xgb) - 1.0) < 1e-9


def test_scores_to_soft_probs_mapping():
    scores = np.array([-0.02, 0.0, 0.03], dtype=np.float64)
    probs = scores_to_soft_probs(scores, threshold=0.01)
    
    logits = np.array([
        [2.0, -2.0, -6.0],
        [-2.0, 2.0, -2.0],
        [-8.0, -4.0, 4.0],
    ])
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    expected = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    np.testing.assert_allclose(probs, expected, atol=1e-12)


def test_apply_confidence_threshold_sets_neutral():
    probs = np.array(
        [
            [0.60, 0.20, 0.20],  # down
            [0.30, 0.40, 0.30],  # neutral by argmax and confidence
            [0.20, 0.30, 0.50],  # up
            [0.34, 0.33, 0.33],  # low confidence => neutral
        ],
        dtype=np.float64,
    )
    preds = apply_confidence_threshold(probs, threshold=0.45)
    assert preds.tolist() == [0, 1, 2, 1]


def test_compute_backtest_metrics_returns_expected_fields():
    probs = np.array(
        [
            [0.10, 0.10, 0.80],  # up
            [0.80, 0.10, 0.10],  # down
            [0.20, 0.70, 0.10],  # neutral
            [0.10, 0.10, 0.80],  # up
        ],
        dtype=np.float64,
    )
    returns = np.array([0.01, -0.005, 0.001, 0.004], dtype=np.float64)
    metrics = compute_backtest_metrics(probs, returns, threshold=0.4)
    for key in (
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
        "coverage",
        "status",
    ):
        assert key in metrics
    assert metrics["total_trades"] >= 0
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_validate_symbol_artifact_passes_with_compatible_fallback_paths(tmp_path):
    artifacts_dir = tmp_path / "run"
    symbol = "RELIANCE.NS"
    symbol_dir = artifacts_dir / "RELIANCE_NS"
    _touch(symbol_dir / "cnn_pattern" / "best_cnn_checkpoint.pt")
    _touch(symbol_dir / "arima_lstm" / "best_lstm_checkpoint.pt")
    _touch(symbol_dir / "arima_lstm" / "feature_scaler.pkl")
    summary = {
        "symbol": symbol,
        "rows": 450,
        "interval": "1h",
    }
    (symbol_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    valid, skip = validate_symbol_artifact(
        symbol=symbol,
        artifacts_dir=artifacts_dir,
        manifest=None,
        min_rows=300,
        expected_freq="1h",
    )
    assert skip is None
    assert valid is not None
    assert valid.symbol == symbol


def test_validate_symbol_artifact_rejects_forex_symbol(tmp_path):
    valid, skip = validate_symbol_artifact(
        symbol="USDINR=X",
        artifacts_dir=tmp_path,
        manifest=None,
        min_rows=300,
        expected_freq="1h",
    )
    assert valid is None
    assert skip is not None
    assert "Check 6 failed" in skip.reason
