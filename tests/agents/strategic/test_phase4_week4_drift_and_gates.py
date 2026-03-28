from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.risk_budgets import VolatilityScaledRiskBudgetEngine
from src.agents.strategic.week4 import (
    ADDMDriftEngine,
    FalseTriggerRateGovernor,
    L4KillSwitchDrillResult,
    Week4Controller,
    Week4RiskGateReviewer,
    build_week4_bundle,
    DriftReading,
)
from src.agents.strategic.xai_attribution import PnLAttributionEngine, XAILogger


def test_week4_addm_drift_engine_escalates_and_demotes() -> None:
    engine = ADDMDriftEngine(
        drift_alert_threshold=0.12,
        sustained_window=2,
        demotion_window=3,
        min_size_multiplier=0.10,
    )
    t0 = datetime(2026, 5, 25, 9, 15, tzinfo=UTC)

    stable = engine.evaluate(
        DriftReading(
            timestamp=t0,
            phase2_input_drift=0.04,
            phase3_output_drift=0.05,
            provenance_reliability=0.95,
        )
    )
    assert stable.drift_alert is False
    assert stable.sustained_drift is False
    assert stable.demotion_triggered is False

    alert_1 = engine.evaluate(
        DriftReading(
            timestamp=t0 + timedelta(minutes=1),
            phase2_input_drift=0.16,
            phase3_output_drift=0.13,
            provenance_reliability=0.55,
        )
    )
    assert alert_1.drift_alert is True
    assert alert_1.sustained_drift is False

    alert_2 = engine.evaluate(
        DriftReading(
            timestamp=t0 + timedelta(minutes=2),
            phase2_input_drift=0.18,
            phase3_output_drift=0.14,
            provenance_reliability=0.50,
        )
    )
    assert alert_2.drift_alert is True
    assert alert_2.sustained_drift is True
    assert alert_2.demotion_triggered is False

    alert_3 = engine.evaluate(
        DriftReading(
            timestamp=t0 + timedelta(minutes=3),
            phase2_input_drift=0.20,
            phase3_output_drift=0.19,
            provenance_reliability=0.45,
        )
    )
    assert alert_3.drift_alert is True
    assert alert_3.sustained_drift is True
    assert alert_3.demotion_triggered is True
    assert alert_3.reason == "sustained_drift_demotion_triggered"
    assert alert_3.size_multiplier < stable.size_multiplier


def test_week4_false_trigger_governor_escalates_operator_review() -> None:
    budget_engine = VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
    governor = FalseTriggerRateGovernor(acceptance_limit=0.20, rolling_days=30)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)

    for i in range(5):
        budget_engine.register_trigger_outcome(timestamp=now + timedelta(minutes=i), was_false_trigger=True)
    budget_engine.register_trigger_outcome(timestamp=now + timedelta(minutes=6), was_false_trigger=False)

    review = governor.evaluate_budget_engine(budget_engine, now=now + timedelta(minutes=7))
    assert review.false_trigger_rate > 0.20
    assert review.auto_adjustment_paused is True
    assert review.escalation_required is True
    assert review.reason == "operator_review_required"


def test_week4_risk_gate_requires_shap_attribution_overrides_and_l4_drill() -> None:
    reviewer = Week4RiskGateReviewer(min_shap_coverage=0.80)
    xai = XAILogger()
    pnl = PnLAttributionEngine()
    now = datetime(2026, 5, 25, 11, 0, tzinfo=UTC)

    for trade_id in ("t1", "t2", "t3", "t4", "t5"):
        xai.mark_trade_seen(trade_id)
    for trade_id in ("t1", "t2", "t3", "t4"):
        xai.log_trade(
            trade_id=trade_id,
            symbol="RELIANCE.NS",
            feature_contributions={"orderbook_imbalance": 0.5, "sentiment_z_t": -0.3},
            agent_contributions={"technical": 0.6, "regime": 0.4},
            signal_family_contributions={"microstructure": 0.7, "risk": 0.3},
            timestamp=now,
        )

    pnl.add_event(
        trade_id="t4",
        symbol="RELIANCE.NS",
        sector="energy",
        agent="technical",
        signal_family="microstructure",
        realized_pnl=1200.0,
        timestamp=now,
    )
    risk_events = (
        {"event": "risk_overseer_decision", "mode": "reduce_only"},
        {"event": "risk_cancel_orders", "mode": "close_only"},
    )

    report = reviewer.evaluate(
        xai_logger=xai,
        pnl_attribution=pnl,
        risk_audit_events=risk_events,
    )
    assert report.passed is True
    assert report.shap_coverage == 0.8
    assert report.risk_override_events == 2
    assert report.failure_reasons == ()


def test_week4_risk_gate_fails_when_controls_missing() -> None:
    reviewer = Week4RiskGateReviewer(min_shap_coverage=0.80)
    xai = XAILogger()
    pnl = PnLAttributionEngine()
    xai.mark_trade_seen("t1")
    xai.mark_trade_seen("t2")
    xai.log_trade(
        trade_id="t1",
        symbol="TCS.NS",
        feature_contributions={"queue_pressure": 0.2},
        agent_contributions={"technical": 1.0},
        signal_family_contributions={"risk": 1.0},
    )
    failing_drill = L4KillSwitchDrillResult(
        timestamp=datetime(2026, 5, 25, 12, 0, tzinfo=UTC),
        passed=False,
        mode=RiskMode.NORMAL,
        should_cancel_orders=False,
        event_recorded=False,
        reason="forced_fail",
    )

    report = reviewer.evaluate(
        xai_logger=xai,
        pnl_attribution=pnl,
        risk_audit_events=(),
        l4_drill_result=failing_drill,
    )
    assert report.passed is False
    assert "shap_coverage" in report.failure_reasons
    assert "pnl_dashboard_active" in report.failure_reasons
    assert "risk_overrides_visible" in report.failure_reasons
    assert "l4_drill_passed" in report.failure_reasons


def test_week4_controller_wires_new_day1_to_day5_flows() -> None:
    bundle = build_week4_bundle()
    controller = Week4Controller(bundle)
    t0 = datetime(2026, 5, 25, 13, 0, tzinfo=UTC)

    decisions = controller.run_day1_and_day2_drift(
        (
            DriftReading(timestamp=t0, phase2_input_drift=0.05, phase3_output_drift=0.03, provenance_reliability=0.95),
            DriftReading(timestamp=t0 + timedelta(minutes=1), phase2_input_drift=0.19, phase3_output_drift=0.18, provenance_reliability=0.5),
        )
    )
    assert len(decisions) == 2
    assert decisions[1].drift_alert is True

    budget_engine = VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
    for i in range(5):
        budget_engine.register_trigger_outcome(timestamp=t0 + timedelta(minutes=10 + i), was_false_trigger=True)
    budget_engine.register_trigger_outcome(timestamp=t0 + timedelta(minutes=16), was_false_trigger=False)
    review = controller.run_day3_false_trigger_review(budget_engine, now=t0 + timedelta(minutes=17))
    assert review.escalation_required is True

    xai = XAILogger()
    for trade_id in ("a", "b", "c", "d", "e"):
        xai.mark_trade_seen(trade_id)
    for trade_id in ("a", "b", "c", "d"):
        xai.log_trade(
            trade_id=trade_id,
            symbol="RELIANCE.NS",
            feature_contributions={"f1": 0.3},
            agent_contributions={"technical": 1.0},
            signal_family_contributions={"risk": 1.0},
            timestamp=t0,
        )
    pnl = PnLAttributionEngine()
    pnl.add_event(
        trade_id="a",
        symbol="RELIANCE.NS",
        sector="energy",
        agent="technical",
        signal_family="risk",
        realized_pnl=500.0,
        timestamp=t0,
    )
    gate = controller.run_day4_and_day5_risk_gate(
        xai_logger=xai,
        pnl_attribution=pnl,
        risk_audit_events=({"event": "risk_overseer_decision", "mode": "close_only"},),
        drill_timestamp=t0 + timedelta(minutes=30),
    )
    assert gate.passed is True
