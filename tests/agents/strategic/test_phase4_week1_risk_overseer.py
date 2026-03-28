from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.risk_overseer import (
    RiskOverseerStateMachine,
    RiskSignalSnapshot,
)
from src.agents.strategic.schemas import RiskMode


def _snapshot(**kwargs) -> RiskSignalSnapshot:
    payload = {
        "timestamp": datetime(2026, 4, 27, 9, 15, tzinfo=UTC),
    }
    payload.update(kwargs)
    return RiskSignalSnapshot(**payload)


def test_one_way_down_transition_is_enforced():
    overseer = RiskOverseerStateMachine()

    first = overseer.evaluate(_snapshot(student_drift=0.20))
    assert first.mode == RiskMode.REDUCE_ONLY

    second = overseer.evaluate(_snapshot(broker_api_error=True))
    assert second.mode == RiskMode.CLOSE_ONLY

    third = overseer.evaluate(_snapshot(student_drift=0.25))
    assert third.mode == RiskMode.CLOSE_ONLY
    assert third.trigger_layer == "L1_MODEL"


def test_kill_switch_exit_requires_operator_ack():
    overseer = RiskOverseerStateMachine()

    triggered = overseer.evaluate(_snapshot(manual_kill_switch=True), authorizer="operator_a")
    assert triggered.mode == RiskMode.KILL_SWITCH

    blocked = overseer.attempt_recovery(
        all_conditions_cleared=True,
        operator_acknowledged=False,
        authorizer="operator_a",
    )
    assert blocked.mode == RiskMode.KILL_SWITCH
    assert blocked.trigger_reason == "kill_switch_exit_requires_operator_ack"

    recovered = overseer.attempt_recovery(
        all_conditions_cleared=True,
        operator_acknowledged=True,
        authorizer="operator_a",
    )
    assert recovered.mode == RiskMode.CLOSE_ONLY


def test_layer_mapping_l2_and_l3():
    overseer = RiskOverseerStateMachine()

    l2 = overseer.evaluate(_snapshot(max_drawdown=0.10))
    assert l2.mode == RiskMode.REDUCE_ONLY
    assert l2.trigger_layer == "L2_PORTFOLIO"

    l3 = overseer.evaluate(_snapshot(broker_rejection_rate=0.25))
    assert l3.mode == RiskMode.CLOSE_ONLY
    assert l3.trigger_layer == "L3_BROKER"


def test_fail_closed_on_unreachable_overseer():
    overseer = RiskOverseerStateMachine()
    decision = overseer.evaluate(
        _snapshot(timestamp=datetime.now(UTC) + timedelta(seconds=1)),
        heartbeat_ok=False,
    )
    assert decision.mode == RiskMode.CLOSE_ONLY
    assert decision.fail_closed is True
    assert decision.block_new_orders is True


def test_mode_change_event_contains_week1_audit_fields():
    overseer = RiskOverseerStateMachine()
    result = overseer.evaluate(_snapshot(student_drift=0.2), authorizer="risk_bot")
    assert result.event_id is not None

    events = overseer.recent_events(limit=1)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "MODE_CHANGE"
    assert event.from_mode == RiskMode.NORMAL
    assert event.to_mode == RiskMode.REDUCE_ONLY
    assert event.trigger_reason == "model_layer_breach"
    assert event.authorizer == "risk_bot"
