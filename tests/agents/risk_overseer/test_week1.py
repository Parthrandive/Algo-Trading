from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.risk_overseer import (
    ModelRiskSnapshot,
    RecoveryRequest,
    RiskEvaluationInput,
    RiskOverseerService,
)
from src.agents.strategic.execution import ExecutionPlanner
from src.agents.strategic.schemas import (
    ActionType,
    EnsembleDecision,
    PolicyWeight,
    PortfolioCheckResult,
    PortfolioIntent,
    RiskMode,
)
from src.db.models import RiskEventDB, RiskStateSnapshotDB
from src.db.phase3_recorder import Phase3Recorder


def _timestamp() -> datetime:
    return datetime(2026, 4, 27, 9, 15, tzinfo=UTC)


def _decision(action: ActionType) -> EnsembleDecision:
    return EnsembleDecision(
        timestamp=_timestamp(),
        symbol="RELIANCE.NS",
        observation_snapshot_id="obs-1",
        action=action,
        action_size=0.4,
        confidence=0.8,
        mode="default",
        dominant_policy_id="student_v1",
        policy_weights=(
            PolicyWeight(policy_id="student_v1", weight=1.0, confidence=0.8, diversity_score=1.0),
        ),
        rationale="test decision",
        risk_mode=RiskMode.NORMAL,
    )


def _intent(action: ActionType) -> PortfolioIntent:
    return PortfolioIntent(
        symbol="RELIANCE.NS",
        decision=_decision(action),
        target_notional=50_000.0,
        target_quantity=100.0,
    )


def _portfolio_check() -> PortfolioCheckResult:
    return PortfolioCheckResult(
        approved=True,
        risk_mode=RiskMode.NORMAL,
        adjusted_quantity=100.0,
        adjusted_notional=50_000.0,
        metadata={"participation_limit": 0.05},
    )


def test_risk_overseer_one_way_down_and_recovery():
    service = RiskOverseerService()

    reduce = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            model=ModelRiskSnapshot(student_drift=0.20),
        )
    )
    assert reduce.mode == RiskMode.REDUCE_ONLY
    assert not reduce.can_submit_order(ActionType.BUY)

    sticky = service.evaluate(RiskEvaluationInput(timestamp=_timestamp()))
    assert sticky.mode == RiskMode.REDUCE_ONLY

    recovered = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            recovery=RecoveryRequest(requested=True, all_conditions_resolved=True, requested_by="ops"),
        )
    )
    assert recovered.mode == RiskMode.NORMAL
    assert recovered.recovery_active is True


def test_manual_kill_switch_requires_explicit_ack_before_recovery():
    service = RiskOverseerService()
    service.trigger_manual_kill_switch(operator_id="alice", reason="drill", timestamp=_timestamp())

    still_killed = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            recovery=RecoveryRequest(
                requested=True,
                all_conditions_resolved=True,
                operator_acknowledged=False,
                clear_manual_override=True,
                requested_by="alice",
            ),
        )
    )
    assert still_killed.mode == RiskMode.KILL_SWITCH

    stepped = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            recovery=RecoveryRequest(
                requested=True,
                all_conditions_resolved=True,
                operator_acknowledged=True,
                clear_manual_override=True,
                requested_by="alice",
            ),
        )
    )
    assert stepped.mode == RiskMode.CLOSE_ONLY
    assert stepped.previous_mode == RiskMode.KILL_SWITCH


def test_execution_planner_fails_closed_when_risk_overseer_is_unreachable():
    planner = ExecutionPlanner()
    plan = planner.plan_order(
        intent=_intent(ActionType.BUY),
        portfolio_check=_portfolio_check(),
        market_price=500.0,
        available_margin=100_000.0,
        required_margin=20_000.0,
        risk_assessment=None,
    )

    assert plan.compliance.passed is False
    assert "risk_overseer_unreachable" in plan.compliance.reasons
    assert plan.compliance.risk_mode == RiskMode.KILL_SWITCH
    assert plan.audit_events[-1]["event_type"] == "REJECTION"


def test_execution_planner_blocks_new_open_under_reduce_only_mode_but_allows_close():
    planner = ExecutionPlanner()
    service = RiskOverseerService()
    assessment = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            model=ModelRiskSnapshot(student_drift=0.20),
        )
    )

    blocked_plan = planner.plan_order(
        intent=_intent(ActionType.BUY),
        portfolio_check=_portfolio_check(),
        market_price=500.0,
        available_margin=100_000.0,
        required_margin=20_000.0,
        risk_assessment=assessment,
    )
    assert blocked_plan.compliance.passed is False
    assert "risk_mode_blocked:reduce_only" in blocked_plan.compliance.reasons

    close_plan = planner.plan_order(
        intent=_intent(ActionType.CLOSE),
        portfolio_check=_portfolio_check(),
        market_price=500.0,
        available_margin=100_000.0,
        required_margin=20_000.0,
        risk_assessment=assessment,
    )
    assert close_plan.compliance.passed is True
    assert close_plan.compliance.risk_mode == RiskMode.REDUCE_ONLY


def test_phase3_recorder_bootstraps_and_persists_risk_tables():
    engine = create_engine("sqlite:///:memory:")
    recorder = Phase3Recorder(engine=engine, session_factory=sessionmaker(bind=engine))
    service = RiskOverseerService()
    assessment = service.evaluate(
        RiskEvaluationInput(
            timestamp=_timestamp(),
            model=ModelRiskSnapshot(student_drift=0.20),
        )
    )

    recorder.save_risk_state(assessment, symbol="RELIANCE.NS")
    recorder.save_risk_event(assessment.trigger_events[0], symbol="RELIANCE.NS")

    with recorder.Session() as session:
        assert session.execute(select(RiskStateSnapshotDB)).scalar_one() is not None
        assert session.execute(select(RiskEventDB)).scalar_one() is not None
