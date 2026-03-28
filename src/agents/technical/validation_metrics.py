"""
Validation metrics for trading model evaluation.

Provides post-cost Sharpe, calibration metrics (ECE, Brier),
and trading-specific diagnostics beyond vanilla accuracy.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Post-cost Sharpe
# ---------------------------------------------------------------------------

def post_cost_sharpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actual_returns: np.ndarray,
    *,
    brokerage_bps: float = 3.0,
    impact_bps: float = 5.0,
    stt_bps: float = 1.0,
    annualization_factor: float | None = None,
) -> float:
    """
    Compute Sharpe ratio after realistic transaction costs.
    Signals: 0=UP (+1), 1=NEUTRAL (0), 2=DOWN (-1).

    Parameters
    ----------
    y_true : array
        True class labels (0, 1, 2).
    y_pred : array
        Predicted class labels (0, 1, 2).
    actual_returns : array
        Next-period log returns.
    """
    y_pred = np.asarray(y_pred, dtype=np.int64)
    actual_returns = np.asarray(actual_returns, dtype=np.float64)

    if len(y_pred) < 2 or len(y_pred) != len(actual_returns):
        return float("nan")

    # Map labels to position signals: 0(UP)->1, 1(NEUTRAL)->0, 2(DOWN)->-1
    signals = np.zeros(len(y_pred), dtype=np.float64)
    signals[y_pred == 0] = 1.0
    signals[y_pred == 2] = -1.0

    # Pre-cost strategy returns
    strategy_returns = signals * actual_returns

    # Total cost per round-trip in decimal
    entry_cost = (brokerage_bps + impact_bps) / 10_000.0
    exit_cost = (brokerage_bps + impact_bps + stt_bps) / 10_000.0
    round_trip_cost = entry_cost + exit_cost

    # Identify trade transitions (signal changes = new round-trip)
    signal_changes = np.zeros(len(signals), dtype=np.float64)
    signal_changes[1:] = np.abs(np.diff(signals))
    
    # Cost proportional to signal change magnitude
    # Example: 0 -> 1 costs 1 leg; 1 -> -1 costs 2 legs.
    trade_cost = signal_changes * (round_trip_cost / 2.0)

    adjusted_returns = strategy_returns - trade_cost

    mean_ret = float(np.mean(adjusted_returns))
    vol = float(np.std(adjusted_returns, ddof=0))

    if vol < 1e-12:
        return 0.0

    if annualization_factor is None:
        # Default to hourly (252 days * 6.5 hours)
        annualization_factor = math.sqrt(252.0 * 6.5)

    return float(mean_ret / vol * annualization_factor)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for multi-class classifier.

    Uses the maximum probability (confidence) and checks whether it
    matches the observed accuracy in each confidence bin.

    Parameters
    ----------
    probs : array, shape (n_samples, n_classes)
        Predicted probability distributions.
    labels : array, shape (n_samples,)
        True class labels (integer).
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        ECE ∈ [0, 1]. Lower is better calibrated.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2 or len(probs) == 0:
        return float("nan")

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(labels))

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)

        bin_count = float(mask.sum())
        if bin_count == 0:
            continue

        avg_confidence = float(np.mean(confidences[mask]))
        avg_accuracy = float(np.mean(correct[mask]))
        ece += (bin_count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# Brier Score (multi-class)
# ---------------------------------------------------------------------------

def brier_score_multiclass(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute multi-class Brier score.

    Brier = (1/N) Σ Σ (p_ij - y_ij)²

    where y_ij is the one-hot encoding of the true label.

    Returns
    -------
    float
        Brier score ∈ [0, 2]. Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2 or len(probs) == 0:
        return float("nan")

    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    valid_mask = (labels >= 0) & (labels < n_classes)
    one_hot[valid_mask, labels[valid_mask]] = 1.0

    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """
    Compute data for a reliability (calibration) diagram.

    Returns dict with 'bin_midpoints', 'observed_frequency',
    'mean_confidence', and 'bin_counts'.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2 or len(probs) == 0:
        return {
            "bin_midpoints": [],
            "observed_frequency": [],
            "mean_confidence": [],
            "bin_counts": [],
        }

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    midpoints: list[float] = []
    observed: list[float] = []
    mean_conf: list[float] = []
    counts: list[float] = []

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)

        bin_count = float(mask.sum())
        midpoints.append(float((lo + hi) / 2.0))
        counts.append(bin_count)

        if bin_count > 0:
            observed.append(float(np.mean(correct[mask])))
            mean_conf.append(float(np.mean(confidences[mask])))
        else:
            observed.append(0.0)
            mean_conf.append(float((lo + hi) / 2.0))

    return {
        "bin_midpoints": midpoints,
        "observed_frequency": observed,
        "mean_confidence": mean_conf,
        "bin_counts": counts,
    }
