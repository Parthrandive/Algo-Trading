from __future__ import annotations

from datetime import UTC, datetime

from src.agents.strategic.execution import ExecutionContext, ExecutionEngine, OrderRequest, OrderType
from src.agents.strategic.risk_overseer import RiskOverseerStateMachine, RiskSignalSnapshot
from src.agents.strategic.schemas import RiskMode


def _context() -> ExecutionContext:
    return ExecutionContext(
        timestamp=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
        symbol="RELIANCE.NS",
        current_price=2_500.0,
        orderbook_imbalance=0.1,
        queue_pressure=0.05,
        avg_volume_1h=500_000.0,
    )


def _buy_request() -> OrderRequest:
    return OrderRequest(
        symbol="RELIANCE.NS",
        direction="BUY",
        target_quantity=100,
        target_notional=250_000.0,
        confidence=0.90,
        order_type=OrderType.LIMIT,
        risk_mode=RiskMode.NORMAL,
    )


def test_execution_engine_applies_risk_overseer_reduce_only_veto():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)
    plan = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
            student_drift=0.30,
        ),
    )

    assert plan.compliance.passed is False
    assert plan.compliance.risk_mode == RiskMode.REDUCE_ONLY
    assert "risk_mode_reduce_only_blocks_buy" in plan.compliance.reasons
    assert plan.instruction.quantity == 0
    assert any(event.get("event") == "risk_overseer_decision" for event in plan.audit_events)


def test_execution_engine_fail_closed_when_overseer_unreachable():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)
    plan = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(timestamp=datetime(2026, 4, 28, 10, 0, tzinfo=UTC)),
        heartbeat_ok=False,
    )

    assert plan.compliance.passed is False
    assert plan.compliance.risk_mode == RiskMode.CLOSE_ONLY
    assert plan.instruction.quantity == 0
    risk_event = next(event for event in plan.audit_events if event.get("event") == "risk_overseer_decision")
    assert risk_event["fail_closed"] is True
    assert risk_event["trigger_reason"] == "risk_overseer_unreachable_fail_closed"


def test_execution_engine_blocks_all_orders_on_manual_kill_switch():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)
    sell_request = OrderRequest(
        symbol="RELIANCE.NS",
        direction="SELL",
        target_quantity=120,
        target_notional=300_000.0,
        confidence=0.9,
        order_type=OrderType.LIMIT,
    )
    plan = engine.plan_execution(
        sell_request,
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
            manual_kill_switch=True,
        ),
    )

    assert plan.compliance.passed is False
    assert plan.compliance.risk_mode == RiskMode.KILL_SWITCH
    assert "risk_mode_kill_switch_blocks_all_orders" in plan.compliance.reasons
    assert plan.instruction.quantity == 0
    assert any(event.get("event") == "risk_cancel_orders" for event in plan.audit_events)
