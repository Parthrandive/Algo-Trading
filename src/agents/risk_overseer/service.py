from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from src.agents.risk_overseer.config import RiskOverseerConfig
from src.agents.risk_overseer.schemas import (
    ActionType,
    CrisisState,
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
    Independent veto layer for execution and crisis taxonomy.

    Week 2 adds stateful crisis confirmation, divergence handling, staged
    re-risking, and protective overlays for sentiment/OOD conditions.
    """

    def __init__(self, config: RiskOverseerConfig | None = None) -> None:
        self.config = config or RiskOverseerConfig()
        self._mode = RiskMode.NORMAL
        self._manual_latch: _ManualLatch | None = None
        self._crisis_confirmations = 0
        self._crisis_active_since: datetime | None = None
        self._divergence_active = False
        self._divergence_since: datetime | None = None
        self._alignment_confirmations = 0
        self._rerisk_step_index = len(self.config.rerisk_step_fractions) - 1

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
        crisis_state = self._resolve_crisis_state(inputs, active_events)
        active_events.extend(self._build_week2_events(inputs, crisis_state, active_events))

        desired_mode = _highest_mode(event.mode for event in active_events) if active_events else RiskMode.NORMAL
        hard_limit_active = any(event.layer in {RiskTriggerLayer.L2_PORTFOLIO, RiskTriggerLayer.L3_BROKER} for event in active_events)
        if inputs.ood.alien and hard_limit_active:
            desired_mode = RiskMode.KILL_SWITCH
            crisis_state = CrisisState.OOD_ALIEN
            active_events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.OOD_HARD_LIMIT_BREACH,
                    mode=RiskMode.KILL_SWITCH,
                    reason="alien OOD coincided with hard limit breach",
                    metadata={"provenance_reliability": inputs.ood.provenance_reliability},
                )
            )

        recovery_active = False
        recovery_reason: str | None = None

        if desired_mode != RiskMode.NORMAL and _severity(desired_mode) > _severity(previous_mode):
            self._mode = desired_mode
        elif (
            not active_events
            and previous_mode == RiskMode.REDUCE_ONLY
            and inputs.recovery.confirmed_alignment
            and self._manual_latch is None
            and self._crisis_active_since is None
        ):
            self._mode = RiskMode.NORMAL
            recovery_active = True
            recovery_reason = "staged re-risking advanced"
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

        permitted_actions = self._resolve_permitted_actions(crisis_state)
        exposure_cap = self._resolve_exposure_cap(crisis_state, inputs)
        hedge_bias = self._resolve_hedge_bias(crisis_state, inputs)
        rerisk_fraction = self._current_rerisk_fraction()
        if not self._divergence_active and crisis_state != CrisisState.AGENT_DIVERGENCE:
            rerisk_fraction = min(rerisk_fraction, exposure_cap)
        elif self._divergence_active:
            exposure_cap = min(exposure_cap, rerisk_fraction)

        veto_reason = None
        if self._mode == RiskMode.KILL_SWITCH:
            veto_reason = "kill_switch_active"
        elif self._mode == RiskMode.CLOSE_ONLY:
            veto_reason = "close_only_guardrail_active"
        elif self._mode == RiskMode.REDUCE_ONLY and crisis_state == CrisisState.AGENT_DIVERGENCE:
            veto_reason = "neutral_hold_active"
        elif self._mode == RiskMode.REDUCE_ONLY:
            veto_reason = "reduce_only_guardrail_active"

        return RiskAssessment(
            timestamp=inputs.timestamp,
            mode=self._mode,
            previous_mode=previous_mode,
            approved=self._mode != RiskMode.KILL_SWITCH,
            crisis_state=crisis_state,
            veto_reason=veto_reason,
            trigger_events=tuple(active_events),
            permitted_actions=permitted_actions,
            exposure_cap=exposure_cap,
            crisis_weight_cap=self.config.crisis_weight_cap,
            rerisk_budget_fraction=rerisk_fraction,
            neutral_hold_active=self._divergence_active,
            hedge_bias=hedge_bias,
            recovery_active=recovery_active,
            recovery_reason=recovery_reason,
            source_service=self.config.service_name,
            metadata={
                "manual_kill_switch_active": self._manual_latch is not None,
                "schema_version": self.config.schema_version,
                "crisis_confirmations": self._crisis_confirmations,
                "divergence_alignment_confirmations": self._alignment_confirmations,
                "divergence_hold_active": self._divergence_active,
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

    def _resolve_crisis_state(
        self,
        inputs: RiskEvaluationInput,
        base_events: list[RiskTriggerEvent],
    ) -> CrisisState:
        if self._is_full_crisis_condition(inputs):
            self._crisis_confirmations += 1
            if self._crisis_confirmations >= self.config.crisis_hysteresis_ticks:
                if self._crisis_active_since is None:
                    self._crisis_active_since = inputs.timestamp
                return CrisisState.FULL_CRISIS
        else:
            self._crisis_confirmations = 0

        if self._crisis_active_since is not None:
            elapsed = inputs.timestamp - self._crisis_active_since
            if elapsed < self.config.crisis_max_duration:
                return CrisisState.FULL_CRISIS
            self._crisis_active_since = None

        if inputs.crisis.major_agent_disagreement_count >= self.config.divergence_major_agent_threshold:
            if not self._divergence_active:
                self._divergence_active = True
                self._divergence_since = inputs.timestamp
                self._alignment_confirmations = 0
                self._rerisk_step_index = 0
            return CrisisState.AGENT_DIVERGENCE

        if self._divergence_active:
            self._advance_rerisk(inputs)
            if self._rerisk_step_index < len(self.config.rerisk_step_fractions) - 1:
                return CrisisState.AGENT_DIVERGENCE
            self._divergence_active = False
            self._divergence_since = None
            self._alignment_confirmations = 0

        if inputs.crisis.feed_staleness_seconds >= self.config.feed_freeze_staleness_seconds:
            return CrisisState.FEED_FREEZE
        if inputs.crisis.drawdown_velocity >= self.config.slow_crash_drawdown_velocity_threshold:
            return CrisisState.SLOW_CRASH
        if inputs.ood.alien:
            return CrisisState.OOD_ALIEN
        if inputs.ood.warning:
            return CrisisState.OOD_WARNING
        if self._is_negative_sentiment(inputs):
            return CrisisState.NEGATIVE_SENTIMENT
        return CrisisState.NORMAL

    def _build_week2_events(
        self,
        inputs: RiskEvaluationInput,
        crisis_state: CrisisState,
        base_events: list[RiskTriggerEvent],
    ) -> tuple[RiskTriggerEvent, ...]:
        events: list[RiskTriggerEvent] = []

        if crisis_state == CrisisState.FULL_CRISIS:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.CRISIS_ENTRY,
                    mode=RiskMode.CLOSE_ONLY,
                    reason="full crisis confirmed after hysteresis",
                    metadata={
                        "realized_vol": inputs.crisis.realized_vol,
                        "baseline_vol": inputs.crisis.baseline_vol,
                        "liquidity_score": inputs.crisis.liquidity_score,
                        "agent_confidence": inputs.crisis.agent_confidence,
                    },
                )
            )
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.RISK_CAP_CHANGE,
                    mode=RiskMode.CLOSE_ONLY,
                    reason="crisis-weighted routing cap applied",
                    value=self.config.full_crisis_exposure_cap,
                    threshold=self.config.crisis_weight_cap,
                    metadata={"crisis_weight_cap": self.config.crisis_weight_cap},
                )
            )

        if crisis_state == CrisisState.AGENT_DIVERGENCE:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.AGENT_DIVERGENCE,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="major agents diverged; neutral-hold active",
                    value=float(inputs.crisis.major_agent_disagreement_count),
                    threshold=float(self.config.divergence_major_agent_threshold),
                    metadata={"rerisk_budget_fraction": self._current_rerisk_fraction()},
                )
            )
        elif self._divergence_active is False and self._rerisk_step_index < len(self.config.rerisk_step_fractions) - 1:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.RE_RISK_STEP,
                    mode=RiskMode.NORMAL,
                    reason="staged re-risking advanced",
                    value=self._current_rerisk_fraction(),
                    metadata={"alignment_confirmations": self._alignment_confirmations},
                )
            )

        if crisis_state == CrisisState.SLOW_CRASH:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L2_PORTFOLIO,
                    trigger_code=RiskTriggerCode.SLOW_CRASH,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="slow-crash de-risking activated",
                    value=inputs.crisis.drawdown_velocity,
                    threshold=self.config.slow_crash_drawdown_velocity_threshold,
                )
            )
        if crisis_state == CrisisState.FEED_FREEZE:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L3_BROKER,
                    trigger_code=RiskTriggerCode.FEED_FREEZE,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="feed freeze / staleness threshold breached",
                    value=inputs.crisis.feed_staleness_seconds,
                    threshold=float(self.config.feed_freeze_staleness_seconds),
                )
            )
        if self._is_negative_sentiment(inputs):
            trigger_code = (
                RiskTriggerCode.SENTIMENT_PRICE_MISMATCH if self._is_sentiment_price_mismatch(inputs) else RiskTriggerCode.NEGATIVE_SENTIMENT_SPIKE
            )
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=trigger_code,
                    mode=RiskMode.REDUCE_ONLY if trigger_code == RiskTriggerCode.SENTIMENT_PRICE_MISMATCH else RiskMode.NORMAL,
                    reason="protective sentiment overlay applied",
                    value=inputs.sentiment.z_t,
                    threshold=self.config.negative_sentiment_z_threshold,
                    metadata={"price_return": inputs.sentiment.price_return},
                )
            )
        if inputs.ood.warning:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.OOD_WARNING,
                    mode=RiskMode.NORMAL,
                    reason="OOD warning triggered staged de-risking",
                    metadata={"provenance_reliability": inputs.ood.provenance_reliability},
                )
            )
        if inputs.ood.alien:
            events.append(
                RiskTriggerEvent(
                    event_id=_event_id(),
                    timestamp=inputs.timestamp,
                    layer=RiskTriggerLayer.L1_MODEL,
                    trigger_code=RiskTriggerCode.OOD_ALIEN,
                    mode=RiskMode.REDUCE_ONLY,
                    reason="alien OOD triggered staged de-risking",
                    metadata={"provenance_reliability": inputs.ood.provenance_reliability},
                )
            )
        return tuple(events)

    def _resolve_exposure_cap(self, crisis_state: CrisisState, inputs: RiskEvaluationInput) -> float:
        cap = self.config.normal_exposure_cap
        if crisis_state == CrisisState.FULL_CRISIS:
            cap = min(cap, self.config.full_crisis_exposure_cap)
        elif crisis_state == CrisisState.AGENT_DIVERGENCE:
            cap = min(cap, self._current_rerisk_fraction())
        elif crisis_state == CrisisState.SLOW_CRASH:
            cap = min(cap, self.config.slow_crash_exposure_cap)
        elif crisis_state == CrisisState.FEED_FREEZE:
            cap = min(cap, self.config.feed_freeze_exposure_cap)
        elif crisis_state == CrisisState.NEGATIVE_SENTIMENT:
            cap = min(cap, self.config.protective_exposure_cap)
        elif crisis_state == CrisisState.OOD_WARNING:
            cap = min(cap, self.config.ood_warning_exposure_cap)
        elif crisis_state == CrisisState.OOD_ALIEN:
            cap = min(cap, self.config.ood_alien_exposure_cap)

        if self._is_negative_sentiment(inputs):
            cap = min(cap, self.config.protective_exposure_cap)
        if inputs.ood.warning:
            cap = min(cap, self.config.ood_warning_exposure_cap)
        if inputs.ood.alien:
            cap = min(cap, self.config.ood_alien_exposure_cap)
        return max(0.0, min(1.0, cap))

    def _resolve_hedge_bias(self, crisis_state: CrisisState, inputs: RiskEvaluationInput) -> float:
        if crisis_state in {CrisisState.FULL_CRISIS, CrisisState.FEED_FREEZE}:
            return 1.0
        if crisis_state in {CrisisState.SLOW_CRASH, CrisisState.OOD_ALIEN}:
            return 0.75
        if self._is_negative_sentiment(inputs):
            return 0.50
        if inputs.ood.warning:
            return 0.35
        return 0.0

    def _resolve_permitted_actions(self, crisis_state: CrisisState) -> tuple[ActionType, ...]:
        if self._mode == RiskMode.KILL_SWITCH:
            return (ActionType.HOLD,)
        if crisis_state == CrisisState.FULL_CRISIS:
            return (ActionType.HOLD, ActionType.CLOSE)
        if crisis_state == CrisisState.AGENT_DIVERGENCE and self._divergence_active:
            return (ActionType.HOLD, ActionType.CLOSE, ActionType.REDUCE)
        if self._mode == RiskMode.CLOSE_ONLY:
            return (ActionType.HOLD, ActionType.CLOSE)
        if self._mode == RiskMode.REDUCE_ONLY:
            return (ActionType.HOLD, ActionType.CLOSE, ActionType.REDUCE)
        return (
            ActionType.BUY,
            ActionType.SELL,
            ActionType.HOLD,
            ActionType.CLOSE,
            ActionType.REDUCE,
        )

    def _can_recover(self, inputs: RiskEvaluationInput) -> bool:
        if not inputs.recovery.requested or not inputs.recovery.all_conditions_resolved:
            return False
        if self._manual_latch is not None and self.config.manual_operator_ack_required:
            return inputs.recovery.operator_acknowledged and inputs.recovery.clear_manual_override
        if self.current_mode == RiskMode.KILL_SWITCH and self.config.manual_operator_ack_required:
            return inputs.recovery.operator_acknowledged
        return True

    def _advance_rerisk(self, inputs: RiskEvaluationInput) -> None:
        if not inputs.recovery.confirmed_alignment:
            self._alignment_confirmations = 0
            return
        self._alignment_confirmations += 1
        if self._alignment_confirmations < self.config.divergence_alignment_signals_required:
            return
        if self._rerisk_step_index < len(self.config.rerisk_step_fractions) - 1:
            self._rerisk_step_index += 1
        self._alignment_confirmations = 0

    def _current_rerisk_fraction(self) -> float:
        return float(self.config.rerisk_step_fractions[self._rerisk_step_index])

    def _is_full_crisis_condition(self, inputs: RiskEvaluationInput) -> bool:
        return (
            inputs.crisis.realized_vol >= inputs.crisis.baseline_vol * self.config.full_crisis_vol_multiplier
            and inputs.crisis.liquidity_score <= self.config.full_crisis_liquidity_floor
            and inputs.crisis.agent_confidence <= self.config.full_crisis_confidence_floor
        )

    def _is_negative_sentiment(self, inputs: RiskEvaluationInput) -> bool:
        z_t = inputs.sentiment.z_t
        if z_t is None:
            return False
        return z_t <= self.config.negative_sentiment_z_threshold or self._is_sentiment_price_mismatch(inputs)

    def _is_sentiment_price_mismatch(self, inputs: RiskEvaluationInput) -> bool:
        sentiment_score = inputs.sentiment.sentiment_score
        if sentiment_score is None:
            return False
        return sentiment_score > 0.0 and inputs.sentiment.price_return <= self.config.sentiment_price_mismatch_drop_threshold


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
