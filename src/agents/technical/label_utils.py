"""
Label construction utilities for trading model training pipelines.

Supports three labeling strategies:
- fixed: static absolute-return threshold (legacy behavior)
- atr: volatility-adjusted threshold via k × ATR(period)
- percentile: distribution-balanced via return percentile cutoffs

All functions produce int64 labels: 0=UP, 1=NEUTRAL, 2=DOWN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_UP = 0
CLASS_NEUTRAL = 1
CLASS_DOWN = 2

LABEL_MODES = ("fixed", "atr", "percentile")


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------

def compute_atr(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    period: int = 14,
) -> np.ndarray:
    """
    Compute Average True Range.

    Returns an ndarray the same length as the inputs.  The first
    ``period - 1`` values will be NaN (insufficient lookback).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().to_numpy()
    return atr


# ---------------------------------------------------------------------------
# Fixed labels (legacy)
# ---------------------------------------------------------------------------

def build_fixed_labels(
    forward_returns: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Classify forward returns using a fixed symmetric threshold.

    - return > threshold  → UP   (0)
    - return < -threshold → DOWN (2)
    - otherwise           → NEUTRAL (1)
    """
    returns = np.asarray(forward_returns, dtype=np.float64)
    labels = np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)
    labels[returns > threshold] = CLASS_UP
    labels[returns < -threshold] = CLASS_DOWN
    return labels


# ---------------------------------------------------------------------------
# ATR-scaled labels
# ---------------------------------------------------------------------------

def build_atr_scaled_labels(
    forward_returns: np.ndarray,
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    k: float = 0.5,
    atr_period: int = 14,
    min_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify forward returns using per-bar volatility-scaled thresholds.

    threshold[t] = k × ATR(period)[t]

    Returns
    -------
    labels : ndarray[int64]
        0=UP, 1=NEUTRAL, 2=DOWN
    threshold_series : ndarray[float64]
        Per-bar effective threshold used.
    """
    returns = np.asarray(forward_returns, dtype=np.float64)
    atr = compute_atr(high, low, close, period=atr_period)

    # Convert ATR (price space) to return space: ATR / close
    close_arr = np.asarray(close, dtype=np.float64)
    close_safe = np.where(np.abs(close_arr) < 1e-12, 1.0, close_arr)
    atr_pct = atr / close_safe

    threshold_series = np.clip(k * atr_pct, min_threshold, None)

    # For bars where ATR is NaN (insufficient warmup), use the first valid value
    first_valid_idx = np.argmax(np.isfinite(threshold_series))
    if first_valid_idx > 0 and np.isfinite(threshold_series[first_valid_idx]):
        threshold_series[:first_valid_idx] = threshold_series[first_valid_idx]
    # If still NaN, fallback to a sensible default
    threshold_series = np.where(
        np.isfinite(threshold_series), threshold_series, min_threshold
    )

    labels = np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)
    labels[returns > threshold_series] = CLASS_UP
    labels[returns < -threshold_series] = CLASS_DOWN

    return labels, threshold_series


def atr_effective_threshold(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    close: np.ndarray | pd.Series,
    k: float = 0.5,
    atr_period: int = 14,
) -> float:
    """
    Compute a single representative threshold for reporting purposes.
    Returns the median ATR-scaled threshold from the provided window.
    """
    atr = compute_atr(high, low, close, period=atr_period)
    close_arr = np.asarray(close, dtype=np.float64)
    close_safe = np.where(np.abs(close_arr) < 1e-12, 1.0, close_arr)
    atr_pct = atr / close_safe
    valid = k * atr_pct[np.isfinite(atr_pct)]
    if len(valid) == 0:
        return 0.005  # conservative fallback
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Percentile-based labels
# ---------------------------------------------------------------------------

def build_percentile_labels(
    forward_returns: np.ndarray,
    up_percentile: float = 70.0,
    down_percentile: float = 30.0,
) -> np.ndarray:
    """
    Classify forward returns by percentile cutoffs.

    - return >= up_percentile      → UP   (0)
    - return <= down_percentile    → DOWN (2)
    - otherwise                    → NEUTRAL (1)

    This guarantees balanced class distribution regardless of the return
    distribution shape.
    """
    returns = np.asarray(forward_returns, dtype=np.float64)
    finite_mask = np.isfinite(returns)
    finite_returns = returns[finite_mask]

    if len(finite_returns) == 0:
        return np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)

    up_cutoff = float(np.percentile(finite_returns, up_percentile))
    down_cutoff = float(np.percentile(finite_returns, down_percentile))

    labels = np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)
    labels[returns >= up_cutoff] = CLASS_UP
    labels[returns <= down_cutoff] = CLASS_DOWN
    return labels


# ---------------------------------------------------------------------------
# Unified label builder
# ---------------------------------------------------------------------------

def choose_neutral_threshold(
    train_forward_returns: np.ndarray,
    requested_threshold: float = 0.005,
    min_neutral_ratio: float = 0.15,
    max_neutral_ratio: float = 0.25,
    target_neutral_ratio: float = 0.20,
) -> tuple[float, float]:
    """
    Search for a threshold that yields a neutral class ratio within the desired band.
    Optimizes for proximity to target_neutral_ratio.
    """
    returns = np.abs(np.asarray(train_forward_returns, dtype=np.float64))
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return float(requested_threshold), 0.0

    # Candidate thresholds: combine requested_threshold with data-driven quantiles
    # We want a fine-grained search between low and high coverage
    quantiles = np.linspace(0.01, 0.50, 50) 
    candidates = np.unique(np.quantile(returns, quantiles))
    candidates = np.sort(np.unique(np.append(candidates, [requested_threshold, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])))
    
    best_threshold = float(requested_threshold)
    best_ratio = 0.0
    min_diff = float("inf")
    
    found_in_band = False
    
    for t in candidates:
        if t <= 0: continue
        ratio = float(np.mean(returns <= t))
        if min_neutral_ratio <= ratio <= max_neutral_ratio:
            found_in_band = True
            diff = abs(ratio - target_neutral_ratio)
            if diff < min_diff:
                min_diff = diff
                best_threshold = float(t)
                best_ratio = ratio
    
    if not found_in_band:
        # Fallback to closest to target
        for t in candidates:
            if t <= 0: continue
            ratio = float(np.mean(returns <= t))
            diff = abs(ratio - target_neutral_ratio)
            if diff < min_diff:
                min_diff = diff
                best_threshold = float(t)
                best_ratio = ratio

    return best_threshold, best_ratio


def build_labels(
    forward_returns: np.ndarray,
    *,
    mode: str = "atr",
    # Fixed mode params
    threshold: float = 0.005,
    # ATR mode params
    high: np.ndarray | pd.Series | None = None,
    low: np.ndarray | pd.Series | None = None,
    close: np.ndarray | pd.Series | None = None,
    k: float = 0.5,
    atr_period: int = 14,
    # Percentile mode params
    up_percentile: float = 70.0,
    down_percentile: float = 30.0,
    # Binary fallback
    use_binary: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Unified label construction with mode selection.

    Parameters
    ----------
    forward_returns : array
        Next-period returns to classify.
    mode : str
        One of "fixed", "atr", "percentile".
    threshold : float
        Used in "fixed" mode.
    high, low, close : array-like
        OHLC data required for "atr" mode.
    k : float
        ATR multiplier for "atr" mode.
    atr_period : int
        Lookback period for ATR.
    up_percentile, down_percentile : float
        Cutoffs for "percentile" mode.
    use_binary : bool
        If True, produce binary labels (0=up, 1=down) regardless of mode.

    Returns
    -------
    labels : ndarray[int64]
        Class labels.
    effective_threshold : float
        Representative threshold used (for reporting/logging).
    """
    returns = np.asarray(forward_returns, dtype=np.float64)

    if use_binary:
        labels = np.where(returns >= 0.0, 0, 1).astype(np.int64)
        return labels, 0.0

    mode = str(mode).strip().lower()
    if mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode '{mode}'. Use one of {LABEL_MODES}")

    if mode == "fixed":
        labels = build_fixed_labels(returns, threshold)
        return labels, float(threshold)

    if mode == "atr":
        if high is None or low is None or close is None:
            raise ValueError("ATR label mode requires high, low, close arrays.")
        labels, threshold_series = build_atr_scaled_labels(
            returns, high, low, close, k=k, atr_period=atr_period,
        )
        effective = float(np.nanmedian(threshold_series))
        return labels, effective

    if mode == "percentile":
        labels = build_percentile_labels(returns, up_percentile, down_percentile)
        finite = returns[np.isfinite(returns)]
        if len(finite) > 0:
            effective = float(np.percentile(finite, up_percentile))
        else:
            effective = 0.005
        return labels, effective

    # Should never reach here
    raise ValueError(f"Unhandled label mode: {mode}")


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def recall_balance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute |Up Recall − Down Recall|.

    A value close to 0 means balanced directional recall.
    A value close to 1 means the model is biased to one direction.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    up_true_mask = y_true == CLASS_UP
    down_true_mask = y_true == CLASS_DOWN

    up_recall = float(np.mean(y_pred[up_true_mask] == CLASS_UP)) if up_true_mask.any() else 0.0
    down_recall = float(np.mean(y_pred[down_true_mask] == CLASS_DOWN)) if down_true_mask.any() else 0.0

    return abs(up_recall - down_recall)


def directional_coverage(predictions: np.ndarray) -> float:
    """Fraction of predictions that are UP or DOWN (not NEUTRAL)."""
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(predictions) == 0:
        return 0.0
    directional = np.isin(predictions, [CLASS_UP, CLASS_DOWN])
    return float(np.mean(directional))


def class_balance_report(labels: np.ndarray) -> dict[str, float]:
    """Return per-class percentage breakdown."""
    labels = np.asarray(labels, dtype=np.int64)
    total = max(len(labels), 1)
    return {
        "up_pct": float((labels == CLASS_UP).sum() / total * 100),
        "neutral_pct": float((labels == CLASS_NEUTRAL).sum() / total * 100),
        "down_pct": float((labels == CLASS_DOWN).sum() / total * 100),
    }


def validate_label_distribution(labels: np.ndarray, symbol: str) -> tuple[bool, list[str]]:
    """
    Validate that label distribution meets training gates.
    - UP and DOWN classes should be roughly 25-45%
    - No class should be <15% or >55%
    """
    report = class_balance_report(labels)
    messages = []
    passed = True

    if report["up_pct"] < 15.0 or report["down_pct"] < 15.0 or report["neutral_pct"] < 15.0:
        messages.append(f"[{symbol}] Distribution failure: A class has < 15% representation. UP={report['up_pct']:.1f}%, NEUTRAL={report['neutral_pct']:.1f}%, DOWN={report['down_pct']:.1f}%")
        passed = False
    
    if report["up_pct"] > 55.0 or report["down_pct"] > 55.0 or report["neutral_pct"] > 55.0:
        messages.append(f"[{symbol}] Distribution warning/failure: A class has > 55% representation. UP={report['up_pct']:.1f}%, NEUTRAL={report['neutral_pct']:.1f}%, DOWN={report['down_pct']:.1f}%")
        # In some cases we might just warn, but let's flag it

    return passed, messages
