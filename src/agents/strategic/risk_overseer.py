from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from src.agents.strategic.schemas import RiskMode


class KillSwitchLayer(str, Enum):
    L1_MODEL = "L1_MODEL"
    L2_PORTFOLIO = "L2_PORTFOLIO"
    L3_BROKER = "L3_BROKER"
    L4_MANUAL = "L4_MANUAL"
    FAIL_CLOSED = "FAIL_CLOSED"
    FULL_CRISIS = "FULL_CRISIS"
    AGENT_DIVERGENCE = "AGENT_DIVERGENCE"
    SLOW_CRASH = "SLOW_CRASH"
    SENTIMENT = "SENTIMENT"
    OOD_STAGED = "OOD_STAGED"
    OOD_HARD_LIMIT = "OOD_HARD_LIMIT"


_MODE_ORDER: dict[RiskMode, int] = {
    RiskMode.NORMAL: 0,
    RiskMode.REDUCE_ONLY: 1,
    RiskMode.CLOSE_ONLY: 2,
    RiskMode.KILL_SWITCH: 3,
}

_LAYER_TO_MIN_MODE: dict[KillSwitchLayer, RiskMode] = {
    KillSwitchLayer.L1_MODEL: RiskMode.REDUCE_ONLY,
    KillSwitchLayer.L2_PORTFOLIO: RiskMode.REDUCE_ONLY,
    KillSwitchLayer.L3_BROKER: RiskMode.CLOSE_ONLY,
    KillSwitchLayer.L4_MANUAL: RiskMode.KILL_SWITCH,
    KillSwitchLayer.FAIL_CLOSED: RiskMode.CLOSE_ONLY,
    KillSwitchLayer.FULL_CRISIS: RiskMode.CLOSE_ONLY,
    KillSwitchLayer.AGENT_DIVERGENCE: RiskMode.REDUCE_ONLY,
    # Week 2 staged controls are budget caps, not immediate neutralization.
    KillSwitchLayer.SLOW_CRASH: RiskMode.NORMAL,
    KillSwitchLayer.SENTIMENT: RiskMode.NORMAL,
    KillSwitchLayer.OOD_STAGED: RiskMode.NORMAL,
    KillSwitchLayer.OOD_HARD_LIMIT: RiskMode.KILL_SWITCH,
}


@dataclass(frozen=True)
class RiskOverseerThresholds:
    l1_student_drift_threshold: float = 0.12
    l1_teacher_student_divergence_threshold: float = 0.20
    l2_max_drawdown_limit: float = 0.08
    l2_daily_loss_limit: float = 0.02
    l2_concentration_limit: float = 0.30
    l3_rejection_rate_limit: float = 0.10
    heartbeat_timeout: timedelta = timedelta(seconds=5)
    crisis_hysteresis_ticks: int = 2
    crisis_max_duration_ticks: int = 5
    crisis_weight_cap: float = 0.70
    divergence_re_risk_steps: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    negative_sentiment_z_threshold: float = -2.5
    sentiment_price_mismatch_return_threshold: float = -0.01
    slow_crash_risk_budget_fraction: float = 0.50
    sentiment_protective_risk_budget_fraction: float = 0.50
    ood_stage_1_persistence: int = 1
    ood_stage_2_persistence: int = 2
    ood_stage_3_persistence: int = 3
    ood_stage_1_risk_budget_fraction: float = 0.50
    ood_stage_2_risk_budget_fraction: float = 0.25
    ood_stage_3_risk_budget_fraction: float = 0.00


@dataclass(frozen=True)
class RiskSignalSnapshot:
    timestamp: datetime
    student_drift: float = 0.0
    teacher_student_divergence: float = 0.0
    model_anomaly: bool = False
    max_drawdown: float = 0.0
    daily_loss_pct: float = 0.0
    concentration_pct: float = 0.0
    broker_api_error: bool = False
    margin_call: bool = False
    broker_rejection_rate: float = 0.0
    manual_kill_switch: bool = False
    realized_vol_break: bool = False
    liquidity_deterioration: bool = False
    confidence_floor_breach: bool = False
    agent_divergence: bool = False
    alignment_recovered: bool = False
    slow_crash: bool = False
    sentiment_z_t: float | None = None
    price_return: float | None = None
    ood_flag: bool = False
    hard_limit_breach: bool = False


@dataclass(frozen=True)
class ModeTransitionEvent:
    event_id: str
    event_type: str
    timestamp_utc: datetime
    from_mode: RiskMode
    to_mode: RiskMode
    trigger_layer: str
    trigger_reason: str
    authorizer: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RiskOverseerDecision:
    mode: RiskMode
    trigger_layer: str
    trigger_reason: str
    should_cancel_orders: bool
    block_new_orders: bool
    fail_closed: bool
    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskOverseerStateMachine:
    """
    Phase 4 Week 1-2 risk overseer state machine.
    Enforces one-way-down transitions unless explicit recovery criteria are met.
    """

    def __init__(self, *, thresholds: RiskOverseerThresholds | None = None) -> None:
        self.thresholds = thresholds or RiskOverseerThresholds()
        self._mode = RiskMode.NORMAL
        self._last_heartbeat: datetime | None = None
        self._events: list[ModeTransitionEvent] = []
        self._crisis_persistence_ticks = 0
        self._crisis_stale_ticks = 0
        self._full_crisis_active = False
        self._ood_persistence_ticks = 0
        self._divergence_hold_active = False
        self._re_risk_index = 0

    @property
    def mode(self) -> RiskMode:
        return self._mode

    def register_heartbeat(self, *, timestamp: datetime) -> None:
        self._last_heartbeat = _as_utc(timestamp)

    def evaluate(
        self,
        snapshot: RiskSignalSnapshot,
        *,
        heartbeat_ok: bool = True,
        authorizer: str = "system",
    ) -> RiskOverseerDecision:
        now = _as_utc(snapshot.timestamp)
        self._last_heartbeat = now if heartbeat_ok else self._last_heartbeat

        if not heartbeat_ok:
            return self._trigger(
                target_mode=_LAYER_TO_MIN_MODE[KillSwitchLayer.FAIL_CLOSED],
                layer=KillSwitchLayer.FAIL_CLOSED,
                reason="risk_overseer_unreachable_fail_closed",
                now=now,
                authorizer=authorizer,
                fail_closed=True,
            )

        layer, reason, metadata = self._detect_trigger(snapshot)
        if layer is None:
            return self._decision(
                layer="NONE",
                reason="no_breach",
                fail_closed=False,
                metadata=self._state_metadata(),
            )
        if reason == "full_crisis_expired_without_revalidation":
            return self._auto_recover_one_step(
                now=now,
                authorizer=authorizer,
                reason=reason,
                metadata=metadata,
            )
        target_mode = _LAYER_TO_MIN_MODE[layer]
        return self._trigger(
            target_mode=target_mode,
            layer=layer,
            reason=reason,
            now=now,
            authorizer=authorizer,
            fail_closed=False,
            metadata=metadata,
        )

    def attempt_recovery(
        self,
        *,
        all_conditions_cleared: bool,
        operator_acknowledged: bool = False,
        timestamp: datetime | None = None,
        authorizer: str = "system",
    ) -> RiskOverseerDecision:
        now = _as_utc(timestamp or datetime.now(UTC))
        if not all_conditions_cleared:
            return self._decision(
                layer="RECOVERY_BLOCKED",
                reason="conditions_not_cleared",
                fail_closed=False,
                metadata=self._state_metadata(),
            )
        if self._mode == RiskMode.KILL_SWITCH and not operator_acknowledged:
            return self._decision(
                layer="RECOVERY_BLOCKED",
                reason="kill_switch_exit_requires_operator_ack",
                fail_closed=False,
                metadata=self._state_metadata(),
            )

        next_mode = self._mode
        if self._mode == RiskMode.KILL_SWITCH:
            next_mode = RiskMode.CLOSE_ONLY
        elif self._mode == RiskMode.CLOSE_ONLY:
            next_mode = RiskMode.REDUCE_ONLY
        elif self._mode == RiskMode.REDUCE_ONLY:
            next_mode = RiskMode.NORMAL

        if next_mode != self._mode:
            event = self._record_event(
                event_type="KILL_SWITCH_EXIT" if self._mode == RiskMode.KILL_SWITCH else "MODE_CHANGE",
                from_mode=self._mode,
                to_mode=next_mode,
                layer="RECOVERY",
                reason="explicit_recovery",
                now=now,
                authorizer=authorizer,
                metadata={"operator_acknowledged": bool(operator_acknowledged)},
            )
            self._mode = next_mode
            self._reset_full_crisis_state(reset_persistence=True)
            return self._decision(
                layer="RECOVERY",
                reason="explicit_recovery",
                fail_closed=False,
                event_id=event.event_id,
                metadata=self._state_metadata(),
            )

        return self._decision(
            layer="RECOVERY",
            reason="already_normal",
            fail_closed=False,
            metadata=self._state_metadata(),
        )

    def recent_events(self, *, limit: int = 100) -> tuple[ModeTransitionEvent, ...]:
        return tuple(self._events[-max(0, int(limit)) :])

    def _detect_trigger(self, snapshot: RiskSignalSnapshot) -> tuple[KillSwitchLayer | None, str, dict[str, Any]]:
        if snapshot.manual_kill_switch:
            self._reset_full_crisis_state(reset_persistence=True)
            return KillSwitchLayer.L4_MANUAL, "manual_kill_switch_triggered", self._state_metadata()

        if (
            snapshot.broker_api_error
            or snapshot.margin_call
            or snapshot.broker_rejection_rate >= self.thresholds.l3_rejection_rate_limit
        ):
            self._reset_full_crisis_state(reset_persistence=True)
            return KillSwitchLayer.L3_BROKER, "broker_layer_breach", self._state_metadata()

        if snapshot.hard_limit_breach and snapshot.ood_flag:
            self._reset_full_crisis_state(reset_persistence=True)
            return KillSwitchLayer.OOD_HARD_LIMIT, "ood_with_hard_limit_breach", self._state_metadata()

        full_crisis = self._is_full_crisis(snapshot)
        if full_crisis:
            self._crisis_persistence_ticks += 1
        else:
            self._crisis_persistence_ticks = 0

        if self._full_crisis_active:
            if full_crisis:
                self._crisis_stale_ticks = 0
                return (
                    KillSwitchLayer.FULL_CRISIS,
                    "full_crisis_revalidated",
                    {
                        **self._state_metadata(),
                        "crisis_weight_cap": float(self.thresholds.crisis_weight_cap),
                        "crisis_persistence_ticks": int(self._crisis_persistence_ticks),
                    },
                )
            self._crisis_stale_ticks += 1
            if self._crisis_stale_ticks >= max(1, self.thresholds.crisis_max_duration_ticks):
                self._full_crisis_active = False
                self._crisis_stale_ticks = 0
                return (
                    KillSwitchLayer.SLOW_CRASH,
                    "full_crisis_expired_without_revalidation",
                    {
                        **self._state_metadata(),
                        "risk_budget_fraction": float(self.thresholds.slow_crash_risk_budget_fraction),
                        "crisis_max_duration_ticks": int(self.thresholds.crisis_max_duration_ticks),
                    },
                )
            return (
                KillSwitchLayer.FULL_CRISIS,
                "full_crisis_holding_for_revalidation",
                {
                    **self._state_metadata(),
                    "crisis_weight_cap": float(self.thresholds.crisis_weight_cap),
                    "crisis_stale_ticks": int(self._crisis_stale_ticks),
                    "crisis_max_duration_ticks": int(self.thresholds.crisis_max_duration_ticks),
                },
            )

        if self._crisis_persistence_ticks >= max(1, self.thresholds.crisis_hysteresis_ticks):
            self._full_crisis_active = True
            self._crisis_stale_ticks = 0
            return (
                KillSwitchLayer.FULL_CRISIS,
                "full_crisis_confirmed",
                {
                    **self._state_metadata(),
                    "crisis_weight_cap": float(self.thresholds.crisis_weight_cap),
                    "crisis_persistence_ticks": int(self._crisis_persistence_ticks),
                },
            )

        if (
            snapshot.max_drawdown >= self.thresholds.l2_max_drawdown_limit
            or snapshot.daily_loss_pct >= self.thresholds.l2_daily_loss_limit
            or snapshot.concentration_pct >= self.thresholds.l2_concentration_limit
        ):
            self._reset_full_crisis_state(reset_persistence=True)
            return KillSwitchLayer.L2_PORTFOLIO, "portfolio_layer_breach", self._state_metadata()

        if snapshot.agent_divergence:
            self._divergence_hold_active = True
            self._re_risk_index = 0
            return (
                KillSwitchLayer.AGENT_DIVERGENCE,
                "agent_divergence_hold",
                {
                    **self._state_metadata(),
                    "risk_budget_fraction": float(self.thresholds.divergence_re_risk_steps[self._re_risk_index]),
                },
            )

        if self._divergence_hold_active:
            if snapshot.alignment_recovered:
                self._re_risk_index = min(
                    self._re_risk_index + 1,
                    len(self.thresholds.divergence_re_risk_steps) - 1,
                )
            if self._re_risk_index < len(self.thresholds.divergence_re_risk_steps) - 1:
                return (
                    KillSwitchLayer.AGENT_DIVERGENCE,
                    "staged_re_risking",
                    {
                        **self._state_metadata(),
                        "risk_budget_fraction": float(self.thresholds.divergence_re_risk_steps[self._re_risk_index]),
                    },
                )
            final_fraction = float(self.thresholds.divergence_re_risk_steps[self._re_risk_index])
            self._divergence_hold_active = False
            return (
                KillSwitchLayer.AGENT_DIVERGENCE,
                "staged_re_risking_complete",
                {
                    **self._state_metadata(),
                    "risk_budget_fraction": final_fraction,
                },
            )

        if snapshot.slow_crash:
            return (
                KillSwitchLayer.SLOW_CRASH,
                "slow_crash_detected",
                {
                    **self._state_metadata(),
                    "risk_budget_fraction": float(self.thresholds.slow_crash_risk_budget_fraction),
                },
            )

        sentiment_z = snapshot.sentiment_z_t
        if sentiment_z is not None and sentiment_z <= self.thresholds.negative_sentiment_z_threshold:
            return (
                KillSwitchLayer.SENTIMENT,
                "extreme_negative_sentiment",
                {
                    **self._state_metadata(),
                    "risk_budget_fraction": float(self.thresholds.sentiment_protective_risk_budget_fraction),
                    "sentiment_z_t": float(sentiment_z),
                },
            )
        if sentiment_z is not None and sentiment_z > 0 and snapshot.price_return is not None:
            if snapshot.price_return <= self.thresholds.sentiment_price_mismatch_return_threshold:
                return (
                    KillSwitchLayer.SENTIMENT,
                    "sentiment_price_mismatch",
                    {
                        **self._state_metadata(),
                        "risk_budget_fraction": float(self.thresholds.sentiment_protective_risk_budget_fraction),
                        "sentiment_z_t": float(sentiment_z),
                        "price_return": float(snapshot.price_return),
                    },
                )

        if snapshot.ood_flag:
            self._ood_persistence_ticks += 1
            stage = self._ood_stage()
            return (
                KillSwitchLayer.OOD_STAGED,
                f"ood_stage_{stage}",
                {
                    **self._state_metadata(),
                    "ood_stage": int(stage),
                    "risk_budget_fraction": float(self._ood_risk_budget_fraction(stage)),
                },
            )
        self._ood_persistence_ticks = 0

        if (
            snapshot.model_anomaly
            or snapshot.student_drift >= self.thresholds.l1_student_drift_threshold
            or snapshot.teacher_student_divergence >= self.thresholds.l1_teacher_student_divergence_threshold
        ):
            self._reset_full_crisis_state(reset_persistence=True)
            return KillSwitchLayer.L1_MODEL, "model_layer_breach", self._state_metadata()

        return None, "no_breach", self._state_metadata()

    def _trigger(
        self,
        *,
        target_mode: RiskMode,
        layer: KillSwitchLayer,
        reason: str,
        now: datetime,
        authorizer: str,
        fail_closed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> RiskOverseerDecision:
        if _MODE_ORDER[target_mode] <= _MODE_ORDER[self._mode]:
            return self._decision(
                layer=layer.value,
                reason=reason,
                fail_closed=fail_closed,
                metadata={**self._state_metadata(), **(metadata or {})},
            )
        event = self._record_event(
            event_type="KILL_SWITCH_TRIGGER" if target_mode == RiskMode.KILL_SWITCH else "MODE_CHANGE",
            from_mode=self._mode,
            to_mode=target_mode,
            layer=layer.value,
            reason=reason,
            now=now,
            authorizer=authorizer,
            metadata={**self._state_metadata(), **(metadata or {})},
        )
        self._mode = target_mode
        return self._decision(
            layer=layer.value,
            reason=reason,
            fail_closed=fail_closed,
            event_id=event.event_id,
            metadata={**self._state_metadata(), **(metadata or {})},
        )

    def _record_event(
        self,
        *,
        event_type: str,
        from_mode: RiskMode,
        to_mode: RiskMode,
        layer: str,
        reason: str,
        now: datetime,
        authorizer: str,
        metadata: dict[str, Any],
    ) -> ModeTransitionEvent:
        event = ModeTransitionEvent(
            event_id=f"risk-{uuid4().hex[:16]}",
            event_type=event_type,
            timestamp_utc=now,
            from_mode=from_mode,
            to_mode=to_mode,
            trigger_layer=layer,
            trigger_reason=reason,
            authorizer=authorizer,
            metadata=metadata,
        )
        self._events.append(event)
        return event

    def _decision(
        self,
        *,
        layer: str,
        reason: str,
        fail_closed: bool,
        event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RiskOverseerDecision:
        mode = self._mode
        return RiskOverseerDecision(
            mode=mode,
            trigger_layer=layer,
            trigger_reason=reason,
            should_cancel_orders=mode in {RiskMode.CLOSE_ONLY, RiskMode.KILL_SWITCH},
            block_new_orders=mode != RiskMode.NORMAL,
            fail_closed=bool(fail_closed),
            event_id=event_id,
            metadata=metadata or {},
        )

    def _is_full_crisis(self, snapshot: RiskSignalSnapshot) -> bool:
        return (
            bool(snapshot.realized_vol_break)
            and bool(snapshot.liquidity_deterioration)
            and bool(snapshot.confidence_floor_breach)
        )

    def _auto_recover_one_step(
        self,
        *,
        now: datetime,
        authorizer: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> RiskOverseerDecision:
        if self._mode not in {RiskMode.CLOSE_ONLY, RiskMode.REDUCE_ONLY}:
            return self._decision(
                layer="RECOVERY",
                reason=reason,
                fail_closed=False,
                metadata={**self._state_metadata(), **metadata},
            )
        next_mode = RiskMode.REDUCE_ONLY if self._mode == RiskMode.CLOSE_ONLY else RiskMode.NORMAL
        event = self._record_event(
            event_type="MODE_CHANGE",
            from_mode=self._mode,
            to_mode=next_mode,
            layer="RECOVERY",
            reason=reason,
            now=now,
            authorizer=authorizer,
            metadata={**self._state_metadata(), **metadata},
        )
        self._mode = next_mode
        return self._decision(
            layer="RECOVERY",
            reason=reason,
            fail_closed=False,
            event_id=event.event_id,
            metadata={**self._state_metadata(), **metadata},
        )

    def _ood_stage(self) -> int:
        if self._ood_persistence_ticks >= self.thresholds.ood_stage_3_persistence:
            return 3
        if self._ood_persistence_ticks >= self.thresholds.ood_stage_2_persistence:
            return 2
        return 1

    def _ood_risk_budget_fraction(self, stage: int) -> float:
        if stage >= 3:
            return self.thresholds.ood_stage_3_risk_budget_fraction
        if stage == 2:
            return self.thresholds.ood_stage_2_risk_budget_fraction
        return self.thresholds.ood_stage_1_risk_budget_fraction

    def _state_metadata(self) -> dict[str, Any]:
        return {
            "divergence_hold_active": bool(self._divergence_hold_active),
            "re_risk_step_index": int(self._re_risk_index),
            "crisis_persistence_ticks": int(self._crisis_persistence_ticks),
            "crisis_stale_ticks": int(self._crisis_stale_ticks),
            "full_crisis_active": bool(self._full_crisis_active),
            "ood_persistence_ticks": int(self._ood_persistence_ticks),
        }

    def _reset_full_crisis_state(self, *, reset_persistence: bool) -> None:
        self._full_crisis_active = False
        self._crisis_stale_ticks = 0
        if reset_persistence:
            self._crisis_persistence_ticks = 0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
