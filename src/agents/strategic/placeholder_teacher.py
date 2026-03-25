from __future__ import annotations

from src.agents.strategic.schemas import (
    ActionType,
    LoopType,
    PolicyType,
    StrategicObservation,
    StrategicToExecutiveContract,
)


def generate_placeholder_teacher_actions(
    observations: list[StrategicObservation],
    *,
    policy_id: str,
) -> list[StrategicToExecutiveContract]:
    """
    Deterministic placeholder policy for Week 1 dry-runs.
    This remains offline/slow-loop and is never Fast Loop eligible.
    """
    decisions: list[StrategicToExecutiveContract] = []
    for observation in observations:
        action, size, reason = _decide(observation)
        confidence = max(observation.consensus_confidence, observation.technical_confidence)
        decision = StrategicToExecutiveContract(
            timestamp=observation.timestamp,
            symbol=observation.symbol,
            policy_id=policy_id,
            policy_type=PolicyType.TEACHER,
            loop_type=LoopType.SLOW,
            action=action,
            action_size=size,
            confidence=min(1.0, max(0.0, confidence)),
            observation_id=None,
            snapshot_id=observation.snapshot_id,
            observation_schema_version=observation.observation_schema_version,
            risk_override="reduce_only" if action == ActionType.REDUCE else None,
            decision_reason=reason,
            metadata={"quality_status": observation.quality_status, "is_placeholder": True},
        )
        decisions.append(decision)
    return decisions


def _decide(observation: StrategicObservation) -> tuple[ActionType, float, str]:
    regime = observation.regime_state.strip().lower()
    if observation.crisis_mode or observation.agent_divergence or regime in {"alien", "crisis"}:
        return ActionType.REDUCE, 0.25, "protective reduce due to crisis/divergence/alien regime"

    direction = observation.consensus_direction.strip().upper()
    confidence = observation.consensus_confidence
    if direction == "BUY" and confidence >= 0.55 and observation.technical_direction != "down":
        return ActionType.BUY, min(1.0, max(0.1, confidence)), "consensus buy aligned with technical"
    if direction == "SELL" and confidence >= 0.55 and observation.technical_direction != "up":
        return ActionType.SELL, min(1.0, max(0.1, confidence)), "consensus sell aligned with technical"
    if direction == "NEUTRAL":
        return ActionType.HOLD, 0.0, "neutral consensus"
    return ActionType.HOLD, 0.0, "insufficient agreement"
