from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.risk_overseer import (
    DriftObservation,
    DriftSurveillanceMonitor,
    Phase4RiskGateAuditor,
    RiskFalseTriggerTracker,
    RiskOverseerConfig,
)
from src.agents.strategic.risk_budgets import VolatilityScaledRiskBudgetEngine
from src.agents.strategic.xai_attribution import PnLAttributionEngine, XAILogger
from src.db.models import DriftAlertEventDB, Phase4RiskGateAuditDB
from src.db.phase3_recorder import Phase3Recorder


def _ts(offset: int = 0) -> datetime:
    return datetime(2026, 5, 18, 9, 15, tzinfo=UTC) + timedelta(minutes=offset)


def test_week4_drift_monitor_escalates_to_sustained_demotion() -> None:
    monitor = DriftSurveillanceMonitor(
        RiskOverseerConfig(
            drift_alert_threshold=0.12,
            drift_sustained_breach_count=2,
            provenance_exposure_floor=0.20,
        )
    )

    first = monitor.evaluate(
        DriftObservation(
            timestamp=_ts(0),
            policy_id="student_v1",
            input_drift_score=0.14,
            policy_drift_score=0.10,
            provenance_reliability=0.90,
        )
    )
    second = monitor.evaluate(
        DriftObservation(
            timestamp=_ts(1),
            policy_id="student_v1",
            input_drift_score=0.16,
            policy_drift_score=0.18,
            provenance_reliability=0.35,
        )
    )

    assert first.sustained_drift is False
    assert first.demotion_triggered is False
    assert first.recommended_risk_mode.value == "reduce_only"
    assert second.sustained_drift is True
    assert second.demotion_triggered is True
    assert second.recommended_risk_mode.value == "reduce_only"
    assert second.exposure_cap_multiplier <= 0.25
    assert second.exposure_cap_multiplier >= 0.20


def test_week4_false_trigger_review_requires_manual_escalation_when_rate_is_high() -> None:
    engine = VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
    tracker = RiskFalseTriggerTracker(engine)
    base = _ts(0)

    for idx in range(4):
        engine.register_trigger_outcome(timestamp=base + timedelta(minutes=idx), was_false_trigger=True)
    engine.register_trigger_outcome(timestamp=base + timedelta(minutes=5), was_false_trigger=False)

    review = tracker.review(timestamp=base + timedelta(minutes=6))

    assert review.false_trigger_rate == 0.8
    assert review.auto_adjustment_paused is True
    assert review.escalate_to_manual_review is True


def test_week4_gate_audit_checks_xai_pnl_and_false_trigger_resolution() -> None:
    monitor = DriftSurveillanceMonitor(RiskOverseerConfig(drift_sustained_breach_count=1))
    drift_alert = monitor.evaluate(
        DriftObservation(
            timestamp=_ts(0),
            policy_id="student_v1",
            input_drift_score=0.15,
            policy_drift_score=0.11,
            provenance_reliability=0.8,
        )
    )
    assert drift_alert.demotion_triggered is True

    xai = XAILogger()
    xai.mark_trade_seen("t1")
    xai.mark_trade_seen("t2")
    xai.mark_trade_seen("t3")
    xai.mark_trade_seen("t4")
    for trade_id in ("t1", "t2", "t3", "t4"):
        xai.log_trade(
            trade_id=trade_id,
            symbol="RELIANCE.NS",
            feature_contributions={"orderbook_imbalance": 0.5, "sentiment_z_t": 0.2},
            agent_contributions={"technical": 0.6, "regime": 0.4},
            signal_family_contributions={"microstructure": 0.7, "consensus": 0.3},
            timestamp=_ts(1),
        )

    pnl = PnLAttributionEngine()
    pnl.add_event(
        trade_id="t1",
        symbol="RELIANCE.NS",
        sector="energy",
        agent="technical",
        signal_family="momentum",
        realized_pnl=1200.0,
        timestamp=_ts(2),
    )

    review = RiskFalseTriggerTracker(VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})).review(timestamp=_ts(3))
    audit = Phase4RiskGateAuditor().audit(
        drift_monitor=monitor,
        false_trigger_review=review,
        xai_logger=xai,
        pnl_attribution=pnl,
        surprise_kill_switch_passed=True,
        drift_demotion_tested=True,
    )

    assert audit.passed is True
    assert audit.checks["xai_coverage"] is True
    assert audit.checks["pnl_dashboard_live"] is True
    assert audit.checks["drift_monitoring_active"] is True
    assert audit.blocker_reasons == ()


def test_week4_recorder_persists_drift_alert_and_gate_audit() -> None:
    engine = create_engine("sqlite:///:memory:")
    recorder = Phase3Recorder(engine=engine, session_factory=sessionmaker(bind=engine))

    monitor = DriftSurveillanceMonitor(RiskOverseerConfig(drift_sustained_breach_count=1))
    alert = monitor.evaluate(
        DriftObservation(
            timestamp=_ts(0),
            policy_id="student_v1",
            input_drift_score=0.20,
            policy_drift_score=0.18,
            provenance_reliability=0.5,
        )
    )
    audit = Phase4RiskGateAuditor().audit(
        drift_monitor=monitor,
        false_trigger_review=RiskFalseTriggerTracker(
            VolatilityScaledRiskBudgetEngine({"nse_large_cap": 0.02})
        ).review(timestamp=_ts(1)),
        xai_logger=XAILogger(),
        pnl_attribution=PnLAttributionEngine(),
        surprise_kill_switch_passed=False,
        drift_demotion_tested=True,
    )

    recorder.save_drift_alert_event(alert)
    recorder.save_phase4_risk_gate_audit(audit)

    with recorder.Session() as session:
        assert session.execute(select(DriftAlertEventDB)).scalar_one() is not None
        assert session.execute(select(Phase4RiskGateAuditDB)).scalar_one() is not None
