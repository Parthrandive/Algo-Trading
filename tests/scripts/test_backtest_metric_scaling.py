from __future__ import annotations

import numpy as np

from scripts.train_universe_xgboost import compute_backtest_metrics


def test_xgboost_backtest_uses_sqrt_annualization_scale():
    n = 200
    probs = np.tile(np.array([[0.05, 0.05, 0.90]], dtype=np.float64), (n, 1))
    raw_returns = np.tile(np.array([0.0020, -0.0010, 0.0030, -0.0005], dtype=np.float64), n // 4 + 1)[:n]
    metrics = compute_backtest_metrics(probs, np.log1p(raw_returns), threshold=0.0)

    # Linear annualization would produce unrealistically large values (> 200) for this synthetic sample.
    assert metrics["sharpe"] < 200.0
    assert metrics["sortino"] < 400.0
