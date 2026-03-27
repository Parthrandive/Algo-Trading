from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.schemas import RiskMode
from src.agents.strategic.week4 import (
    FullStackBacktestEngine,
    GoNoGoAssessor,
    MLOpsGovernanceAuditor,
    PaperTradeDecision,
    PaperTradingConfig,
    PaperTradingHarness,
    Phase3GateEvidenceCollector,
    StressScenarioResult,
    StressTestEngine,
)


def test_week4_backtest_metrics_can_pass_go_live_targets() -> None:
    engine = FullStackBacktestEngine(periods_per_year=252)
    returns = [0.015, 0.012, 0.011, -0.002, 0.013, 0.012, 0.014, 0.010, 0.011, -0.001, 0.013, 0.012]
    report = engine.run_walk_forward([returns[:6], returns[6:]])

    assert len(report.fold_metrics) == 2
    assert report.aggregate_metrics.periods == len(returns)
    assert report.target_pass is True
    assert all(report.target_checks.values())


def test_week4_leakage_audit_detects_future_and_lag_breaches() -> None:
    engine = FullStackBacktestEngine()
    base = datetime(2026, 4, 21, 9, 15, tzinfo=UTC)
    rows = [
        {
            "observation_timestamp": base,
            "feature_timestamp": base - timedelta(seconds=5),
            "decision_timestamp": base + timedelta(seconds=2),
        },
        {
            "observation_timestamp": base + timedelta(minutes=1),
            "feature_timestamp": base + timedelta(minutes=1),
            "decision_timestamp": base + timedelta(seconds=30),
        },
        {
            "observation_timestamp": base + timedelta(minutes=2),
            "feature_timestamp": base,
            "decision_timestamp": base + timedelta(minutes=2, seconds=1),
        },
    ]

    result = engine.audit_leakage(rows, max_feature_lag_seconds=60.0)

    assert result.passed is False
    assert result.leakage_rows == 1
    assert result.lag_breach_rows == 1
    assert "future_data_leakage" in result.reasons
    assert "feature_lag_breach" in result.reasons


def test_week4_survivorship_audit_requires_point_in_time_and_delisted_coverage() -> None:
    engine = FullStackBacktestEngine()
    base = datetime(2026, 4, 21, tzinfo=UTC)

    good_rows = [
        {
            "symbol": "RELIANCE.NS",
            "as_of": base,
            "point_in_time_version": "u_v1",
            "is_active": True,
            "delisted_at": None,
        },
        {
            "symbol": "OLDCO.NS",
            "as_of": base,
            "point_in_time_version": "u_v1",
            "is_active": False,
            "delisted_at": base - timedelta(days=400),
        },
    ]
    bad_rows = [
        {
            "symbol": "ONLYLIVE.NS",
            "as_of": base,
            "point_in_time_version": None,
            "is_active": True,
            "delisted_at": None,
        }
    ]

    assert engine.audit_survivorship(good_rows).passed is True
    bad = engine.audit_survivorship(bad_rows)
    assert bad.passed is False
    assert "missing_delisted_symbol_coverage" in bad.reasons


def test_week4_stress_engine_flags_capacity_and_mode_failures() -> None:
    engine = StressTestEngine(capacity_impact_cap_bps=25.0)
    scenarios = [
        StressScenarioResult(
            scenario_id="covid_crash",
            protective_mode=RiskMode.CLOSE_ONLY,
            expected_min_mode=RiskMode.REDUCE_ONLY,
            snapback_ticks=10,
            max_snapback_ticks=30,
            capacity_multiplier=1.0,
            impact_bps=14.0,
        ),
        StressScenarioResult(
            scenario_id="liquidity_3x",
            protective_mode=RiskMode.NORMAL,
            expected_min_mode=RiskMode.REDUCE_ONLY,
            capacity_multiplier=3.0,
            impact_bps=32.0,
        ),
    ]

    report = engine.evaluate(scenarios)

    assert report.passed is False
    assert report.failure_count >= 2
    assert any("impact_cap_breach" in item for item in report.failure_reasons)
    assert any("insufficient_protective_mode" in item for item in report.failure_reasons)


def test_week4_paper_harness_produces_complete_audit_trail() -> None:
    harness = PaperTradingHarness(
        config=PaperTradingConfig(
            initial_cash=1_000_000.0,
            rejection_probability=0.0,
            partial_fill_probability=0.0,
            partial_fill_ratio=0.5,
            slippage_realism_tolerance_bps=20.0,
            session_minutes=375,
            uptime_target=0.80,
        )
    )
    base = datetime(2026, 4, 23, 9, 15, tzinfo=UTC)
    decisions = [
        PaperTradeDecision(
            timestamp=base,
            symbol="RELIANCE.NS",
            direction="BUY",
            quantity=100,
            confidence=0.8,
            price=2500.0,
            signal_source="strategic_ensemble",
        ),
        PaperTradeDecision(
            timestamp=base + timedelta(minutes=10),
            symbol="TCS.NS",
            direction="BUY",
            quantity=80,
            confidence=0.75,
            price=3900.0,
            signal_source="strategic_ensemble",
        ),
        PaperTradeDecision(
            timestamp=base + timedelta(minutes=20),
            symbol="RELIANCE.NS",
            direction="SELL",
            quantity=60,
            confidence=0.82,
            price=2515.0,
            signal_source="strategic_ensemble",
        ),
    ]

    report = harness.run_session(decisions, outage_minutes=15.0, seed=11)

    assert report.crashed is False
    assert report.total_orders == 3
    assert report.rejected_orders == 0
    assert report.audit_trail_complete is True
    assert report.all_agents_emitting is True
    assert report.slippage_realism_passed is True
    assert report.uptime_ratio >= 0.80


def test_week4_governance_evidence_and_gonogo_assessment() -> None:
    auditor = MLOpsGovernanceAuditor()
    models = [
        {
            "model_id": f"model_{idx}",
            "version": "1.0.0",
            "training_data_snapshot_hash": f"datahash_{idx}",
            "code_hash": f"codehash_{idx}",
            "feature_schema_version": "1.0",
            "hyperparameters": {"lr": 1e-3},
            "validation_metrics": {"sharpe": 2.0},
            "baseline_comparison": {"delta": 0.1},
            "plan_version": "v1.3.7",
            "created_by": "owner",
            "reviewed_by": "partner",
        }
        for idx in range(6)
    ]

    t0 = datetime(2026, 4, 24, 10, 0, tzinfo=UTC)
    rollback_result = auditor.run_rollback_drill(
        champions=("phase3_student_v1", "phase3_student_v2"),
        failed_model_id="phase3_student_v2",
        started_at=t0,
        ended_at=t0 + timedelta(seconds=18),
    )
    governance = auditor.audit_registry(models, rollback_result=rollback_result)

    assert governance.registry_complete is True
    assert governance.reproducibility_ready is True
    assert governance.promotion_gate_passed is True
    assert governance.rollback_passed is True

    backtest_metrics = FullStackBacktestEngine().compute_metrics(
        [0.014, 0.010, 0.012, -0.001, 0.011, 0.013, 0.012, -0.001]
    )
    evidence = Phase3GateEvidenceCollector().collect(
        latency_p99_ms=7.8,
        latency_p999_ms=9.9,
        degrade_path_passed=True,
        crisis_slice_agreement=0.86,
        rollback_result=rollback_result,
        backtest_metrics=backtest_metrics,
        observation_schema_version="1.0",
        tier1_status={
            "impact_monitor_functional": True,
            "risk_budgets_functional": True,
            "orderbook_imbalance_integrated": True,
            "latency_ci_gate_ready": True,
        },
        compliance_violations=0,
        blocking_defects=0,
    )

    assessment = GoNoGoAssessor().assess(evidence)
    assert assessment.go is True
    assert assessment.at_risk_items == ()
