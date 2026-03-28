from __future__ import annotations

from datetime import UTC, datetime

from src.agents.strategic.execution import ExecutionContext, ExecutionEngine, OrderRequest, OrderType
from src.agents.strategic.risk_overseer import RiskOverseerStateMachine, RiskSignalSnapshot
from src.agents.strategic.schemas import RiskMode


def _context() -> ExecutionContext:
    return ExecutionContext(
        timestamp=datetime(2026, 5, 5, 10, 0, tzinfo=UTC),
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
        confidence=0.9,
        order_type=OrderType.LIMIT,
        risk_mode=RiskMode.NORMAL,
    )


def test_execution_engine_applies_staged_ood_risk_budget_scaling_on_buys():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)

    stage_1 = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 5, 5, 10, 0, tzinfo=UTC),
            ood_flag=True,
        ),
    )
    assert stage_1.compliance.passed is True
    assert stage_1.compliance.risk_mode == RiskMode.NORMAL
    assert stage_1.instruction.quantity == 50

    stage_2 = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 5, 5, 10, 1, tzinfo=UTC),
            ood_flag=True,
        ),
    )
    assert stage_2.instruction.quantity == 25

    stage_3 = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 5, 5, 10, 2, tzinfo=UTC),
            ood_flag=True,
        ),
    )
    assert stage_3.instruction.quantity == 0
    risk_event = next(event for event in stage_3.audit_events if event.get("event") == "risk_overseer_decision")
    assert risk_event["risk_budget_fraction"] == 0.0


def test_execution_engine_applies_sentiment_protective_scaling():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)

    plan = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 5, 5, 10, 0, tzinfo=UTC),
            sentiment_z_t=-3.2,
        ),
    )
    assert plan.compliance.passed is True
    assert plan.compliance.risk_mode == RiskMode.NORMAL
    assert plan.instruction.quantity == 50


def test_execution_engine_still_blocks_buys_during_divergence_hold():
    overseer = RiskOverseerStateMachine()
    engine = ExecutionEngine(risk_overseer=overseer)

    plan = engine.plan_execution(
        _buy_request(),
        _context(),
        risk_snapshot=RiskSignalSnapshot(
            timestamp=datetime(2026, 5, 5, 10, 0, tzinfo=UTC),
            agent_divergence=True,
        ),
    )

    assert plan.compliance.passed is False
    assert plan.compliance.risk_mode == RiskMode.REDUCE_ONLY
    assert "risk_mode_reduce_only_blocks_buy" in plan.compliance.reasons
    assert plan.instruction.quantity == 0
