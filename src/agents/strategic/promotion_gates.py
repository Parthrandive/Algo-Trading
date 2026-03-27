from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from src.agents.strategic.config import PromotionGateConfig
from src.agents.strategic.schemas import RiskMode


class PolicyStage(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    CHAMPION = "champion"
    DEMOTED = "demoted"


@dataclass(frozen=True)
class PromotionEvidence:
    walk_forward_outperformance: bool
    non_regression: bool
    crisis_slice_agreement: float
    latency_gate_passed: bool
    rollback_ready: bool
    critical_compliance_violations: int = 0


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    from_stage: PolicyStage
    to_stage: PolicyStage
    reasons: tuple[str, ...]
    evaluated_at: datetime


@dataclass(frozen=True)
class RollbackDrillResult:
    executed: bool
    reverted_to: str | None
    mttr_seconds: float
    reasons: tuple[str, ...]
    timestamp: datetime


class SafeRLTrainingGuard:
    """
    Day 5 safety filter for exploration in training and dry-runs.
    """

    def clamp_action_size(self, raw_size: float, *, risk_mode: RiskMode) -> float:
        size = max(0.0, min(1.0, float(raw_size)))
        if risk_mode == RiskMode.KILL_SWITCH:
            return 0.0
        if risk_mode == RiskMode.CLOSE_ONLY:
            return min(size, 0.25)
        if risk_mode == RiskMode.REDUCE_ONLY:
            return min(size, 0.50)
        return size


class PromotionGatePipeline:
    """
    Candidate -> Shadow -> Champion promotion gate with strict evidence checks.
    """

    _ALLOWED_TRANSITIONS = {
        PolicyStage.CANDIDATE: PolicyStage.SHADOW,
        PolicyStage.SHADOW: PolicyStage.CHAMPION,
        PolicyStage.CHAMPION: PolicyStage.CHAMPION,
        PolicyStage.DEMOTED: PolicyStage.CANDIDATE,
    }

    def __init__(self, config: PromotionGateConfig | None = None) -> None:
        self.config = config or PromotionGateConfig()

    def transition(
        self,
        *,
        current_stage: PolicyStage,
        requested_stage: PolicyStage,
        evidence: PromotionEvidence,
    ) -> PromotionDecision:
        now = datetime.now(UTC)
        reasons: list[str] = []
        allowed = self._ALLOWED_TRANSITIONS[current_stage]
        if requested_stage != allowed:
            reasons.append(f"invalid_transition:{current_stage.value}->{requested_stage.value}")
            return PromotionDecision(
                approved=False,
                from_stage=current_stage,
                to_stage=current_stage,
                reasons=tuple(reasons),
                evaluated_at=now,
            )

        if requested_stage == PolicyStage.SHADOW:
            if not evidence.rollback_ready:
                reasons.append("rollback_not_ready")
            if evidence.critical_compliance_violations > 0:
                reasons.append("critical_compliance_violation_detected")
        elif requested_stage == PolicyStage.CHAMPION:
            if not evidence.walk_forward_outperformance:
                reasons.append("walk_forward_outperformance_missing")
            if self.config.require_non_regression and not evidence.non_regression:
                reasons.append("non_regression_failed")
            if evidence.crisis_slice_agreement < float(self.config.min_crisis_agreement):
                reasons.append("crisis_slice_agreement_below_threshold")
            if self.config.require_latency_gate and not evidence.latency_gate_passed:
                reasons.append("latency_gate_failed")
            if self.config.require_rollback_readiness and not evidence.rollback_ready:
                reasons.append("rollback_not_ready")
            if evidence.critical_compliance_violations > 0:
                reasons.append("critical_compliance_violation_detected")

        approved = len(reasons) == 0
        return PromotionDecision(
            approved=approved,
            from_stage=current_stage,
            to_stage=requested_stage if approved else current_stage,
            reasons=tuple(reasons),
            evaluated_at=now,
        )


class RollbackDrillManager:
    """
    Tracks champion lineage and executes deterministic rollback drills.
    """

    def __init__(self) -> None:
        self._history: list[str] = []

    def set_champion(self, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if self._history and self._history[-1] == model_id:
            return
        self._history.append(model_id)

    def rollback(self, *, failed_model_id: str, started_at: datetime, ended_at: datetime) -> RollbackDrillResult:
        now = datetime.now(UTC)
        reasons: list[str] = []
        if ended_at < started_at:
            reasons.append("invalid_mttr_window")
        if not self._history or self._history[-1] != failed_model_id:
            reasons.append("failed_model_is_not_active_champion")
        if len(self._history) < 2:
            reasons.append("no_previous_champion_available")

        mttr_seconds = max(0.0, (ended_at - started_at).total_seconds())
        if reasons:
            return RollbackDrillResult(
                executed=False,
                reverted_to=None,
                mttr_seconds=mttr_seconds,
                reasons=tuple(reasons),
                timestamp=now,
            )

        reverted_to = self._history[-2]
        self._history.pop()  # Remove failed champion.
        return RollbackDrillResult(
            executed=True,
            reverted_to=reverted_to,
            mttr_seconds=mttr_seconds,
            reasons=(),
            timestamp=now,
        )
