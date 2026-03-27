from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2


def returns_to_labels(returns: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)
    labels[np.asarray(returns) < -float(threshold)] = CLASS_DOWN
    labels[np.asarray(returns) > float(threshold)] = CLASS_UP
    return labels


def directional_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "test_accuracy": 0.0,
            "directional_accuracy": 0.0,
            "up_precision": 0.0,
            "up_recall": 0.0,
            "up_f1": 0.0,
            "up_support": 0,
            "down_precision": 0.0,
            "down_recall": 0.0,
            "down_f1": 0.0,
            "down_support": 0,
            "test_confusion_matrix": [],
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[CLASS_UP, CLASS_DOWN],
        average=None,
        zero_division=0,
    )
    directional_mask = np.isin(y_true, [CLASS_UP, CLASS_DOWN])
    directional_accuracy = (
        float(accuracy_score(y_true[directional_mask], y_pred[directional_mask]))
        if directional_mask.any()
        else 0.0
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy,
        "up_precision": float(precision[0]),
        "up_recall": float(recall[0]),
        "up_f1": float(f1[0]),
        "up_support": int(support[0]),
        "down_precision": float(precision[1]),
        "down_recall": float(recall[1]),
        "down_f1": float(f1[1]),
        "down_support": int(support[1]),
        "test_confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP],
        ).tolist(),
    }


def choose_symbol_threshold(
    train_scores: np.ndarray,
    train_actual_returns: np.ndarray,
    requested_threshold: float,
    class_threshold_min: float,
    min_neutral_ratio: float,
    max_neutral_ratio: float,
    target_neutral_ratio: float,
) -> tuple[float, float]:
    abs_scores = np.abs(np.asarray(train_scores, dtype=np.float64))
    abs_scores = abs_scores[np.isfinite(abs_scores)]
    if len(abs_scores) == 0:
        return float(max(requested_threshold, class_threshold_min)), 0.0

    quantile_candidates = [
        float(np.quantile(abs_scores, q))
        for q in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
    ]
    static_candidates = [
        0.0002,
        0.0005,
        0.0010,
        0.0015,
        0.0020,
        0.0025,
        0.0030,
        0.0040,
        0.0050,
        0.0075,
        0.0100,
        0.0150,
        0.0200,
        0.0250,
        0.0300,
    ]
    candidates = sorted({float(requested_threshold), *quantile_candidates, *static_candidates})
    min_threshold = max(float(class_threshold_min), 1e-6)
    candidates = [c for c in candidates if c >= min_threshold]

    actual_returns = np.asarray(train_actual_returns, dtype=np.float64)
    has_actuals = len(actual_returns) == len(train_scores) and len(actual_returns) > 0
    if has_actuals:
        actual_returns = np.where(np.isfinite(actual_returns), actual_returns, 0.0)

    candidate_stats: list[tuple[float, float, float, float, float]] = []
    for threshold in candidates:
        threshold = max(float(threshold), 1e-6)
        neutral_ratio = float((abs_scores <= threshold).mean())
        directional_accuracy = 0.0
        mean_directional_recall = 0.0
        directional_coverage = 0.0
        if has_actuals:
            y_pred = returns_to_labels(train_scores, threshold=threshold)
            y_true = returns_to_labels(actual_returns, threshold=threshold)
            cls_metrics = directional_classification_metrics(y_true=y_true, y_pred=y_pred)
            directional_accuracy = float(cls_metrics["directional_accuracy"])
            mean_directional_recall = float((cls_metrics["up_recall"] + cls_metrics["down_recall"]) / 2.0)
            directional_coverage = float(np.mean(np.isin(y_pred, [CLASS_UP, CLASS_DOWN])))
        quality_score = (0.60 * directional_accuracy) + (0.30 * mean_directional_recall) + (0.10 * directional_coverage)
        candidate_stats.append((threshold, neutral_ratio, quality_score, directional_accuracy, mean_directional_recall))

    in_band = [
        (threshold, ratio, quality_score, directional_accuracy, mean_directional_recall)
        for threshold, ratio, quality_score, directional_accuracy, mean_directional_recall in candidate_stats
        if float(min_neutral_ratio) <= ratio <= float(max_neutral_ratio)
    ]
    if in_band:
        best_threshold, best_ratio, _, _, _ = max(
            in_band,
            key=lambda item: (
                item[2],
                -abs(item[1] - float(target_neutral_ratio)),
                -item[3],
                -item[4],
                -item[0],
            ),
        )
        return float(best_threshold), float(best_ratio)

    best_threshold, best_ratio, _, _, _ = max(
        candidate_stats,
        key=lambda item: (
            item[2],
            -abs(item[1] - float(target_neutral_ratio)),
            -item[3],
            -item[4],
            -item[0],
        ),
    )
    return float(best_threshold), float(best_ratio)
