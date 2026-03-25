from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Iterable

from src.agents.strategic.config import EnsembleConfig
from src.agents.strategic.schemas import (
    ActionType,
    EnsembleDecision,
    PolicyAction,
    PolicyWeight,
    RiskMode,
    StrategicObservation,
    ThresholdCandidate,
)


_ACTION_SIGN = {
    ActionType.BUY: 1.0,
    ActionType.SELL: -1.0,
    ActionType.HOLD: 0.0,
    ActionType.CLOSE: -0.5,
    ActionType.REDUCE: -0.25,
}


@dataclass(frozen=True)
class ThresholdOptimizerResult:
    best_candidate: ThresholdCandidate
    ranked_candidates: tuple[ThresholdCandidate, ...]


class MaxEntropyEnsemble:
    """
    Week 2 ensemble engine.

    Teacher-policy only: consumes offline/slow-loop teacher actions and emits one
    combined decision that can later be distilled into a student artifact.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

    def decide(
        self,
        observation: StrategicObservation,
        teacher_actions: Iterable[PolicyAction],
        *,
        threshold: ThresholdCandidate | None = None,
    ) -> EnsembleDecision:
        actions = tuple(teacher_actions)
        if not actions:
            raise ValueError("teacher_actions must not be empty")
        if any(action.loop_type.value == "fast" for action in actions):
            raise ValueError("teacher actions are blocked from Fast Loop usage")

        weights = self._compute_weights(actions, observation)
        signed_action = sum(_ACTION_SIGN[action.action] * weight.weight for action, weight in zip(actions, weights))
        weighted_size = sum(action.action_size * weight.weight for action, weight in zip(actions, weights))
        combined_confidence = sum(action.confidence * weight.weight for action, weight in zip(actions, weights))
        selected_threshold = threshold or self.default_thresholds()
        action = self._map_signed_action_to_enum(signed_action, combined_confidence, selected_threshold)
        dominant = max(weights, key=lambda item: item.weight)
        rationale = self._build_rationale(observation, signed_action, combined_confidence, dominant.policy_id)
        risk_mode = self._resolve_risk_mode(observation)
        mode = "divergence_hold" if observation.agent_divergence else "crisis" if observation.crisis_mode else "default"
        final_size = self._final_action_size(action, weighted_size, combined_confidence, observation)

        return EnsembleDecision(
            timestamp=observation.timestamp,
            symbol=observation.symbol,
            observation_snapshot_id=observation.snapshot_id,
            action=action,
            action_size=final_size,
            confidence=max(0.0, min(1.0, combined_confidence)),
            mode=mode,
            dominant_policy_id=dominant.policy_id,
            policy_weights=weights,
            threshold_candidate_id=selected_threshold.candidate_id,
            rationale=rationale,
            risk_mode=risk_mode,
            metadata={
                "signed_action": signed_action,
                "weighted_action_size": weighted_size,
                "teacher_policy_count": len(actions),
            },
        )

    def default_thresholds(self) -> ThresholdCandidate:
        return ThresholdCandidate(
            candidate_id="default",
            buy_threshold=0.30,
            sell_threshold=-0.30,
            reduce_threshold=0.20,
            hold_threshold=0.10,
            fitness=0.0,
        )

    def optimize_thresholds(
        self,
        *,
        candidates: Iterable[ThresholdCandidate],
        evaluation_scores: dict[str, float],
    ) -> ThresholdOptimizerResult:
        ranked = sorted(
            (
                candidate.model_copy(update={"fitness": float(evaluation_scores.get(candidate.candidate_id, 0.0))})
                for candidate in candidates
            ),
            key=lambda item: item.fitness,
            reverse=True,
        )
        if not ranked:
            raise ValueError("at least one threshold candidate is required")
        return ThresholdOptimizerResult(best_candidate=ranked[0], ranked_candidates=tuple(ranked))

    def _compute_weights(
        self,
        teacher_actions: tuple[PolicyAction, ...],
        observation: StrategicObservation,
    ) -> tuple[PolicyWeight, ...]:
        confidences = [max(self.config.confidence_floor, action.confidence) for action in teacher_actions]
        diversity_scores = []
        for index, action in enumerate(teacher_actions):
            peers = [other for idx, other in enumerate(teacher_actions) if idx != index]
            if not peers:
                diversity_scores.append(1.0)
                continue
            avg_distance = mean(abs(_ACTION_SIGN[action.action] - _ACTION_SIGN[peer.action]) for peer in peers)
            diversity_scores.append(1.0 + avg_distance)

        logits = []
        for action, confidence, diversity in zip(teacher_actions, confidences, diversity_scores):
            score = confidence + self.config.diversity_penalty * diversity
            if observation.consensus_direction == "BUY" and action.action == ActionType.BUY:
                score += self.config.consensus_boost
            elif observation.consensus_direction == "SELL" and action.action == ActionType.SELL:
                score += self.config.consensus_boost
            if observation.crisis_mode:
                score *= 1.0 - self.config.crisis_confidence_haircut
            if observation.agent_divergence:
                score = min(score, self.config.divergence_hold_confidence_cap)
            logits.append(score / max(self.config.temperature, 1e-6))

        normalized = _softmax(logits)
        adjusted = _floor_weights(normalized, self.config.minimum_policy_weight)
        return tuple(
            PolicyWeight(
                policy_id=action.policy_id,
                weight=weight,
                confidence=action.confidence,
                diversity_score=diversity,
            )
            for action, weight, diversity in zip(teacher_actions, adjusted, diversity_scores)
        )

    def _map_signed_action_to_enum(
        self,
        signed_action: float,
        confidence: float,
        threshold: ThresholdCandidate,
    ) -> ActionType:
        if confidence < threshold.hold_threshold:
            return ActionType.HOLD
        if signed_action >= threshold.buy_threshold:
            return ActionType.BUY
        if signed_action <= threshold.sell_threshold:
            return ActionType.SELL
        if abs(signed_action) >= threshold.reduce_threshold:
            return ActionType.REDUCE
        return ActionType.HOLD

    def _resolve_risk_mode(self, observation: StrategicObservation) -> RiskMode:
        regime = observation.regime_state.strip().lower()
        if observation.crisis_mode or regime == "crisis":
            return RiskMode.CLOSE_ONLY
        if observation.agent_divergence or regime == "alien":
            return RiskMode.REDUCE_ONLY
        return RiskMode.NORMAL

    def _final_action_size(
        self,
        action: ActionType,
        weighted_size: float,
        combined_confidence: float,
        observation: StrategicObservation,
    ) -> float:
        if action == ActionType.HOLD:
            return 0.0
        size = max(0.05, min(1.0, weighted_size * max(combined_confidence, 0.10)))
        if observation.crisis_mode:
            return min(size, 0.25)
        if observation.agent_divergence:
            return min(size, 0.10)
        if action in {ActionType.REDUCE, ActionType.CLOSE}:
            return min(size, 0.50)
        return size

    def _build_rationale(
        self,
        observation: StrategicObservation,
        signed_action: float,
        confidence: float,
        dominant_policy_id: str,
    ) -> str:
        parts = [
            f"max-entropy teacher blend from {dominant_policy_id}",
            f"signed_action={signed_action:.3f}",
            f"confidence={confidence:.3f}",
        ]
        if observation.crisis_mode:
            parts.append("crisis safeguards active")
        if observation.agent_divergence:
            parts.append("divergence hold safeguards active")
        return "; ".join(parts)


def build_threshold_candidates(now: datetime | None = None) -> tuple[ThresholdCandidate, ...]:
    now = now or datetime.now(UTC)
    seed = int(now.timestamp()) % 7
    candidates = []
    for index in range(5):
        spread = 0.25 + 0.03 * ((seed + index) % 4)
        candidates.append(
            ThresholdCandidate(
                candidate_id=f"ga_candidate_{index + 1}",
                buy_threshold=spread,
                sell_threshold=-spread,
                reduce_threshold=max(0.10, spread - 0.12),
                hold_threshold=0.08 + 0.01 * index,
                metadata={"offline_only": True},
            )
        )
    return tuple(candidates)


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    anchor = max(logits)
    exps = [math.exp(value - anchor) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def _floor_weights(weights: list[float], minimum_weight: float) -> list[float]:
    if not weights:
        return []
    floored = [max(minimum_weight, item) for item in weights]
    total = sum(floored)
    return [item / total for item in floored]
