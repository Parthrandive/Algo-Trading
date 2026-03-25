from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.agents.strategic.schemas import EnsembleEvaluationResult


def evaluate_equal_weight_ensemble(
    policy_actions: Mapping[str, float],
    confidences: Mapping[str, float],
) -> EnsembleEvaluationResult:
    action_values = np.asarray(list(policy_actions.values()), dtype=float)
    confidence_values = np.asarray(list(confidences.values()), dtype=float)
    mean_confidence = float(np.clip(np.mean(confidence_values), 0.0, 1.0)) if confidence_values.size else 0.0
    return EnsembleEvaluationResult(
        policy_ids=tuple(policy_actions.keys()),
        equal_weight_action=float(np.mean(action_values)) if action_values.size else 0.0,
        action_dispersion=float(np.std(action_values, ddof=0)) if action_values.size else 0.0,
        mean_confidence=mean_confidence,
        metadata={"policy_count": int(action_values.size)},
    )


def baseline_summary(returns: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(returns), dtype=float)
    if values.size == 0:
        return {"buy_and_hold": 0.0, "random_action": 0.0}
    rng = np.random.default_rng(42)
    random_actions = rng.choice([-1.0, 0.0, 1.0], size=values.size)
    return {
        "buy_and_hold": float(np.sum(values)),
        "random_action": float(np.sum(values * random_actions)),
    }


def ablation_drop(reference_metric: float, ablated_metric: float) -> float:
    return float(reference_metric - ablated_metric)
