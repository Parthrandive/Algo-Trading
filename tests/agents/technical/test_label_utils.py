"""Unit tests for src.agents.technical.label_utils."""
import numpy as np
import pytest

from src.agents.technical.label_utils import (
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
    build_atr_scaled_labels,
    build_fixed_labels,
    build_labels,
    build_percentile_labels,
    class_balance_report,
    compute_atr,
    directional_coverage,
    recall_balance,
    atr_effective_threshold,
)


# ---------------------------------------------------------------------------
# compute_atr
# ---------------------------------------------------------------------------

class TestComputeAtr:
    def test_basic_atr(self):
        high = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
                         22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype=np.float64)
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                        20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=np.float64)
        close = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                          21.0, 22.0, 23.0, 24.0, 25.0, 26.0], dtype=np.float64)
        atr = compute_atr(high, low, close, period=3)
        # First bar has NaN prev_close → TR[0] is NaN → first `period` values are NaN
        assert np.isnan(atr[0])
        assert np.isnan(atr[1])
        assert np.isnan(atr[2])
        # From index 3 onward, values should be finite and positive
        assert np.all(np.isfinite(atr[3:]))
        assert np.all(atr[3:] > 0)

    def test_period_determines_nan_count(self):
        n = 30
        high = np.random.uniform(10, 12, n)
        low = np.random.uniform(8, 10, n)
        close = np.random.uniform(9, 11, n)
        for period in [5, 10, 14]:
            atr = compute_atr(high, low, close, period=period)
            nan_count = np.sum(np.isnan(atr))
            # NaN count is `period` because TR[0] is also NaN (no prev_close)
            assert nan_count == period, f"Expected {period} NaNs, got {nan_count}"


# ---------------------------------------------------------------------------
# build_fixed_labels
# ---------------------------------------------------------------------------

class TestBuildFixedLabels:
    def test_three_class_classification(self):
        returns = np.array([0.02, -0.02, 0.001, 0.0, -0.001])
        labels = build_fixed_labels(returns, threshold=0.005)
        assert labels[0] == CLASS_UP       # 0.02 > 0.005
        assert labels[1] == CLASS_DOWN     # -0.02 < -0.005
        assert labels[2] == CLASS_NEUTRAL  # |0.001| <= 0.005
        assert labels[3] == CLASS_NEUTRAL  # 0.0 <= 0.005
        assert labels[4] == CLASS_NEUTRAL  # |-0.001| <= 0.005

    def test_empty_returns(self):
        labels = build_fixed_labels(np.array([]), threshold=0.01)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# build_atr_scaled_labels
# ---------------------------------------------------------------------------

class TestBuildAtrScaledLabels:
    def test_basic_functionality(self):
        n = 30
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0.5, 1.5, n)
        low = close - np.random.uniform(0.5, 1.5, n)
        returns = np.random.randn(n) * 0.02

        labels, thresholds = build_atr_scaled_labels(
            returns, high, low, close, k=0.5, atr_period=5,
        )
        assert labels.dtype == np.int64
        assert len(labels) == n
        assert len(thresholds) == n
        assert set(np.unique(labels)).issubset({CLASS_UP, CLASS_NEUTRAL, CLASS_DOWN})

    def test_extreme_returns_classified_correctly(self):
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        # ATR will be ~2.0, ATR/close ~0.02, k*ATR = ~0.01
        returns = np.zeros(n)
        returns[0] = 0.10   # Very large UP
        returns[1] = -0.10  # Very large DOWN
        returns[2] = 0.0    # Neutral

        labels, thresholds = build_atr_scaled_labels(
            returns, high, low, close, k=0.5, atr_period=5,
        )
        assert labels[0] == CLASS_UP
        assert labels[1] == CLASS_DOWN
        assert labels[2] == CLASS_NEUTRAL


# ---------------------------------------------------------------------------
# build_percentile_labels
# ---------------------------------------------------------------------------

class TestBuildPercentileLabels:
    def test_balanced_distribution(self):
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        labels = build_percentile_labels(returns, up_percentile=70, down_percentile=30)

        up_pct = (labels == CLASS_UP).mean()
        down_pct = (labels == CLASS_DOWN).mean()
        neutral_pct = (labels == CLASS_NEUTRAL).mean()

        # Should be approximately 30/40/30
        assert 0.25 <= up_pct <= 0.35
        assert 0.25 <= down_pct <= 0.35
        assert 0.35 <= neutral_pct <= 0.45


# ---------------------------------------------------------------------------
# build_labels (unified)
# ---------------------------------------------------------------------------

class TestBuildLabelsUnified:
    def test_fixed_mode(self):
        returns = np.array([0.02, -0.02, 0.001])
        labels, eff_th = build_labels(returns, mode="fixed", threshold=0.005)
        assert labels[0] == CLASS_UP
        assert labels[1] == CLASS_DOWN
        assert labels[2] == CLASS_NEUTRAL
        assert eff_th == 0.005

    def test_atr_mode_requires_ohlc(self):
        returns = np.array([0.01, -0.01])
        with pytest.raises(ValueError, match="requires high, low, close"):
            build_labels(returns, mode="atr")

    def test_atr_mode_with_ohlc(self):
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        returns = np.random.randn(n) * 0.01
        labels, eff_th = build_labels(
            returns, mode="atr", high=high, low=low, close=close,
            k=0.5, atr_period=5,
        )
        assert len(labels) == n
        assert eff_th > 0.0

    def test_binary_mode_overrides(self):
        returns = np.array([0.01, -0.01, 0.0])
        labels, _ = build_labels(returns, mode="atr", use_binary=True)
        # Binary: 0=up, 1=down
        assert set(np.unique(labels)).issubset({0, 1})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown label mode"):
            build_labels(np.array([0.01]), mode="bogus")


# ---------------------------------------------------------------------------
# recall_balance
# ---------------------------------------------------------------------------

class TestRecallBalance:
    def test_perfect_balance(self):
        y_true = np.array([CLASS_UP, CLASS_UP, CLASS_DOWN, CLASS_DOWN])
        y_pred = np.array([CLASS_UP, CLASS_UP, CLASS_DOWN, CLASS_DOWN])
        # Both directions have 100% recall → balance = 0.0
        assert recall_balance(y_true, y_pred) == pytest.approx(0.0)

    def test_complete_imbalance(self):
        y_true = np.array([CLASS_UP, CLASS_UP, CLASS_DOWN, CLASS_DOWN])
        y_pred = np.array([CLASS_UP, CLASS_UP, CLASS_NEUTRAL, CLASS_NEUTRAL])
        # Up recall = 1.0, Down recall = 0.0 → balance = 1.0
        assert recall_balance(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# directional_coverage
# ---------------------------------------------------------------------------

class TestDirectionalCoverage:
    def test_all_directional(self):
        preds = np.array([CLASS_UP, CLASS_DOWN, CLASS_UP, CLASS_DOWN])
        assert directional_coverage(preds) == pytest.approx(1.0)

    def test_all_neutral(self):
        preds = np.array([CLASS_NEUTRAL, CLASS_NEUTRAL])
        assert directional_coverage(preds) == pytest.approx(0.0)

    def test_mixed(self):
        preds = np.array([CLASS_UP, CLASS_NEUTRAL, CLASS_DOWN, CLASS_NEUTRAL])
        assert directional_coverage(preds) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# class_balance_report
# ---------------------------------------------------------------------------

class TestClassBalanceReport:
    def test_report_sums_to_100(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        report = class_balance_report(labels)
        total = report["up_pct"] + report["neutral_pct"] + report["down_pct"]
        assert total == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# atr_effective_threshold
# ---------------------------------------------------------------------------

class TestAtrEffectiveThreshold:
    def test_returns_positive_value(self):
        close = np.full(20, 100.0)
        high = np.full(20, 102.0)
        low = np.full(20, 98.0)
        th = atr_effective_threshold(high, low, close, k=0.5, atr_period=5)
        assert th > 0.0

    def test_fallback_when_insufficient_data(self):
        # Only 2 data points, period=14 → all NaN
        close = np.array([100.0, 101.0])
        high = np.array([102.0, 103.0])
        low = np.array([98.0, 99.0])
        th = atr_effective_threshold(high, low, close, k=0.5, atr_period=14)
        assert th == 0.005  # conservative fallback
