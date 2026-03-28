import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, use_binary: bool = False) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class or binary probabilities.
    For multi-class, computes ECE on the maximum probability.
    """
    if len(probs) == 0 or len(labels) == 0:
        return float('nan')
        
    if use_binary or probs.ndim == 1 or probs.shape[1] == 2:
        # Binary case: assume probs represents probability of class 1.
        p = probs if probs.ndim == 1 else probs[:, 1]
        preds = (p >= 0.5).astype(int)
        confidences = np.where(preds == 1, p, 1 - p)
        accuracies = (preds == labels).astype(float)
    else:
        # Multi-class case: use max confidence
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        accuracies = (preds == labels).astype(float)
        
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(probs)) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def tune_directional_thresholds(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    min_thresh: float = 0.30,
    max_thresh: float = 0.65,
    step: float = 0.05,
    min_coverage: float = 0.30,
    min_dir_acc: float = 0.35,
    use_binary: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Independently tunes UP/DOWN prediction confidence thresholds.
    Optimizes for min(|up_recall - down_recall|) subject to coverage and directional accuracy constraints.
    Returns (thresh_up, thresh_down, stats)
    """
    if use_binary or probs_val.ndim == 1 or probs_val.shape[1] == 2:
        # Binary
        p1 = probs_val if probs_val.ndim == 1 else probs_val[:, 1]
        p0 = 1.0 - p1
        preds_up = p0
        preds_down = p1
        label_up = 0
        label_down = 1
    else:
        # Multiclass assume 0=UP, 1=NEUTRAL, 2=DOWN
        preds_up = probs_val[:, 0]
        preds_down = probs_val[:, 2]
        label_up = 0
        label_down = 2

    best_thresh_up = 0.50
    best_thresh_down = 0.50
    best_imbalance = float('inf')
    best_stats = {}
    
    thresholds = np.arange(min_thresh, max_thresh + step / 2, step)
    
    for t_up in thresholds:
        for t_down in thresholds:
            # Predict based on thresholds
            mask_up = preds_up >= t_up
            mask_down = preds_down >= t_down
            
            active_mask = mask_up | mask_down
            coverage = np.mean(active_mask) if len(active_mask) > 0 else 0.0
            
            if coverage < min_coverage:
                continue
                
            y_active = y_val[active_mask]
            
            # Reconstruct predictions in active mask
            preds_active = np.zeros_like(y_active)
            
            # Simple conflict resolution: pick highest prob if both trigger
            # or just assume mutually exclusive in most well-behaved classifiers
            conflict_mask = mask_up & mask_down
            if np.any(conflict_mask):
                safe_up = mask_up.copy()
                safe_down = mask_down.copy()
                safe_up[conflict_mask] = preds_up[conflict_mask] > preds_down[conflict_mask]
                safe_down[conflict_mask] = preds_down[conflict_mask] > preds_up[conflict_mask]
                preds_active[safe_up[active_mask]] = label_up
                preds_active[safe_down[active_mask]] = label_down
            else:
                preds_active[mask_up[active_mask]] = label_up
                preds_active[mask_down[active_mask]] = label_down
                
            dir_acc = np.mean(preds_active == y_active) if len(y_active) > 0 else 0.0
            
            if dir_acc < min_dir_acc:
                continue
                
            # Class-specific recall among ALL positive instances (not just active)
            actual_up = np.sum(y_val == label_up)
            actual_down = np.sum(y_val == label_down)
            
            up_recall = np.sum((preds_active == label_up) & (y_active == label_up)) / actual_up if actual_up > 0 else 0.0
            down_recall = np.sum((preds_active == label_down) & (y_active == label_down)) / actual_down if actual_down > 0 else 0.0
            
            imbalance = abs(up_recall - down_recall)
            
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_thresh_up = float(t_up)
                best_thresh_down = float(t_down)
                best_stats = {
                    "coverage": float(coverage),
                    "dir_acc": float(dir_acc),
                    "up_recall": float(up_recall),
                    "down_recall": float(down_recall),
                    "imbalance": float(imbalance)
                }

    # Fallback if no configuration meets constraints
    if best_imbalance == float('inf'):
        best_thresh_up = 0.50
        best_thresh_down = 0.50
        
    return best_thresh_up, best_thresh_down, best_stats
