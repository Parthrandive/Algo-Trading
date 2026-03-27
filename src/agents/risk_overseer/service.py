from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from src.agents.risk_overseer.config import RiskOverseerConfig
from src.agents.risk_overseer.schemas import (
    ActionType,
    RiskAssessment,
    RiskEvaluationInput,
    RiskTriggerCode,
    RiskTriggerEvent,
    RiskTriggerLayer,
)
from src.agents.strategic.schemas import RiskMode


@dataclass(frozen=True)
class _ManualLatch:
    operator_id: str
    reason: str
    activated_at: datetime


class RiskOverseerService:
    """
    Independent veto layer for execution.

    The service keeps its own operating mode and only permits recovery when all
    active triggers are clear and an explicit recovery request is supplied.
    """

    def __init__(self, config: RiskOverseerConfig | None = None) -> None:
        self.config = config or RiskOverseerConfig()
        self._mode = RiskMode.NORMAL
        self._manual_latch: _ManualLatch | None = None

    @property
    def current_mode(self) -> RiskMode:
        return self._mode

    def trigger_manual_kill_switch(
        self,
        *,
        operator_id: str,
        reason: str,
        timestamp: datetime | None = None,
    ) -> RiskTriggerEvent:
        timestamp = _utc_now() if timestamp is None else _normalize_time(timestamp)
        self._manual_latch = _ManualLatch(operator_id=operator_id, reason=reason, activated_at=timestamp)
        self._mode = RiskMode.KILL_SWITCH
        return RiskTriggerEvent(
            event_id=_event_id(),
            timestamp=timestamp,
            layer=RiskTriggerLayer.L4_MANUAL,
            trigger_code=RiskTriggerCode.MANUAL_KILL_SWITCH,
            mode=RiskMode.KILL_SWITCH,
            reason=reason,
            operator_id=operator_id,
            metadata={"service_name": self.config.service_name},
        )

    def evaluate(self, inputs: RiskEvaluationInput) -> RiskAssessment:
        previous_mode = self._mode
        active_events = list(self._collect_events(inputs))
        desired_mode = _highest_mode(event.mode for event in active_events) if active_events else RiskMode.NORMAL
        recovery_active = False
        recovery_reason: str | None = None

        if desired_mode != RiskMode.NORMAL and _severity(desired_mode) > _severity(previous_mode):
            self._mode = desired_mode
        elif not active_events and _severity(previous_mode) > _severity(RiskMode.NORMAL):
            if self._can_recover(inputs):
                if self._manual_latch is not None and inputs.recovery.clear_manual_override:
                    self._manual_latch = None
                self._mode = _recover_one_step(previous_mode)
                recovery_active = True
                recovery_reason = f"explicit recovery from {previous_mode.value}"
                active_events.append(
                    RiskTriggerEvent(
                        event_id=_event_id(),
                        timestamp=inputs.timestamp,
                        layer=RiskTriggerLayer.L4_MANUAL,
                        trigger_code=RiskTriggerCode.RECOVERY_STEP,
                        mode=self._mode,
                        reason=recovery_reason,
                        operator_id=inputs.recovery.requested_by,
                        metadata={"note": inputs.recovery.note or ""},
                    )
                )
            else:
                self._mode = previous_mode
        else:
            self._mode = _max_mode(previous_mode, desired_mode)

        permitted_actions = _permitted_actions(self._mode)
        veto_reason = None
        if self._mode == RiskMode.KILL_SWITCH:
            veto_reason = "kill_switch_active"
        elif self._mode == RiskMode.CLOSE_ONLY:
            veto_reason = "close_only_guardrail_active"
        elif self._mode == RiskMode.REDUCE_ONLY:
            veto_reason = "reduce_only_guardrail_active"

        return RiskAssessment(
            timestamp=inputs.timestamp,
            mode=self._mode,
            previous_mode=previous_mode,
            approved=self._mode != RiskMode.KILL_SWITCH,
            veto_reason=veto_reason,
            trigger_events=tuple(active_events),
            permitted_actions=permitted_actions,
            recovery_active=recovery_active,
            recovery_reason=recovery_reason,
            source_service=self.config.service_name,
            metadata={
                "manual_kill_switch_active": self._manual_latch is not None,
                "schema_version": self.config.schema_version,
            },
        )

    def _collect_events(self, inputs: RiskEvaluationInput) -> tuple[RiskTriggerEvent, ...]:
        events: list[RiskTriggerEvent] = []

        if inputs.model.student_drift >= self.config.student_drift_alert_threshold:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.STUDENT_DRIFT,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="student drift breached threshold",
                    value=inputs.model.student_drift,
                    threshold=self.config.student_drift_alert_threshold,
                    metadata={
                        "active_policy_id": inputs.model.active_policy_id,
                        "fallback_policy_id": inputs.model.fallback_policy_id,
                    },
                )
            )
        if inputs.model.teacher_student_divergence >= self.config.teacher_divergence_threshold:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.TEACHER_DIVERGENCE,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="teacher-student divergence breached threshold",
                    value=inputs.model.teacher_student_divergence,
                    threshold=self.config.teacher_divergence_threshold,
                )
            )

        if inputs.portfolio.max_drawdown >= self.config.max_drawdown_limit:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.MAX_DRAWDOWN_BREACH,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="max drawdown breached hard limit",
                    value=inputs.portfolio.max_drawdown,
                    threshold=self.config.max_drawdown_limit,
                )
            )
        if inputs.portfolio.daily_loss >= self.config.daily_loss_limit:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.DAILY_LOSS_BREACH,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="daily loss breached hard limit",
                    value=inputs.portfolio.daily_loss,
                    threshold=self.config.daily_loss_limit,
                )
            )
        if inputs.portfolio.concentration_breach:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.CONCENTRATION_BREACH,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="portfolio concentration breached",
                )
            )

        if not inputs.broker.api_healthy:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L3_BROKER,
                    trigger_code=RiskTriggerCode.BROKER_API_ERROR,
                    mode=RiskMode.CLOSE_ONLY,
                    reason=inputs.broker.latest_error or "broker API unhealthy",
                )
            )
        if inputs.broker.margin_call_active:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L3_BROKER,
                    trigger_code=RiskTriggerCode.MARGIN_CALL,
                    mode=RiskMode.CLOSE_ONLY,
                    reason="margin call active",
                )
            )
        if inputs.broker.rejection_rate >= self.config.broker_rejection_rate_limit:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L3_BROKER,
                    trigger_code=RiskTriggerCode.REJECTION_RATE_BREACH,
                    mode=RiskMode.CLOSE_ONLY,
                    reason="broker rejection rate breached threshold",
                    value=inputs.broker.rejection_rate,
                    threshold=self.config.broker_rejection_rate_limit,
                )
            )

        if self._manual_latch is not None:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L4_MANUAL,
                    trigger_code=RiskTriggerCode.MANUAL_KILL_SWITCH,
                    mode=RiskMode.KILL_SWITCH,
                    reason=self._manual_latch.reason,
                    operator_id=self._manual_latch.operator_id,
                    metadata={"activated_at": self._manual_latch.activated_at.isoformat()},
                )
            )

        return tuple(events)

    def _can_recover(self, inputs: RiskEvaluationInput) -> bool:
        if not inputs.recovery.requested or not inputs.recovery.all_conditions_resolved:
            return False
        if self._manual_latch is not None and self.config.manual_operator_ack_required:
            return inputs.recovery.operator_acknowledged and inputs.recovery.clear_manual_override
        if self.current_mode == RiskMode.KILL_SWITCH and self.config.manual_operator_ack_required:
            return inputs.recovery.operator_acknowledged
        return True


def _event_id() -> str:
    return f"risk-{uuid4().hex[:12]}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _severity(mode: RiskMode) -> int:
    return {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }[mode]


def _max_mode(left: RiskMode, right: RiskMode) -> RiskMode:
    return left if _severity(left) >= _severity(right) else right


def _highest_mode(modes) -> RiskMode:
    highest = RiskMode.NORMAL
    for mode in modes:
        highest = _max_mode(highest, mode)
    return highest


def _recover_one_step(mode: RiskMode) -> RiskMode:
    if mode == RiskMode.KILL_SWITCH:
        return RiskMode.CLOSE_ONLY
    if mode == RiskMode.CLOSE_ONLY:
        return RiskMode.REDUCE_ONLY
    return RiskMode.NORMAL


def _permitted_actions(mode: RiskMode) -> tuple[ActionType, ...]:
    if mode == RiskMode.NORMAL:
        return (
            ActionType.BUY,
            ActionType.SELL,
            ActionType.HOLD,
            ActionType.CLOSE,
            ActionType.REDUCE,
        )
    if mode == RiskMode.REDUCE_ONLY:
        return (ActionType.HOLD, ActionType.CLOSE, ActionType.REDUCE)
    if mode == RiskMode.CLOSE_ONLY:
        return (ActionType.HOLD, ActionType.CLOSE)
    return (ActionType.HOLD,)
