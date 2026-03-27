from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    FastLoopLatencyEventDB,
    ImpactEventDB,
    LatencyBenchmarkArtifactDB,
    OperationalMetricsSnapshotDB,
    OrderBookFeatureEventDB,
    PnLAttributionEventDB,
    PromotionGateEventDB,
    RiskCapEventDB,
    RollbackDrillEventDB,
    XAITradeExplanationDB,
)
from src.db.phase3_recorder import Phase3Recorder


def test_phase3_recorder_persists_week3_telemetry_rows():
    engine = create_engine("sqlite:///:memory:")
    recorder = Phase3Recorder(engine=engine, session_factory=sessionmaker(bind=engine))
    ts = datetime(2026, 4, 20, 9, 30, tzinfo=UTC)

    recorder.save_impact_event(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "bucket": "liquid_large_cap",
            "event_type": "IMPACT_BREACH",
            "breach": True,
            "participation_rate": 0.06,
            "slippage_delta_bps": 24.0,
            "impact_score": 1.4,
            "size_multiplier": 0.75,
            "cooldown_until": ts,
            "risk_override": "reduce_only",
            "reasons": ["slippage_delta_breach"],
        }
    )
    recorder.save_risk_cap_event(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "asset_cluster": "nse_large_cap",
            "regime": "elevated",
            "cap_fraction": 0.70,
            "changed": True,
            "event_type": "RISK_CAP_CHANGE",
            "false_trigger_rate": 0.1,
            "auto_adjustment_paused": False,
        }
    )
    recorder.save_orderbook_feature_event(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "quality_flag": "pass",
            "degraded": False,
            "degradation_reason": None,
            "imbalance": 0.22,
            "queue_pressure": 0.18,
        }
    )
    recorder.save_fastloop_latency_event(
        {
            "timestamp": ts,
            "stage": "decision_path",
            "mode": "normal",
            "event_type": "FASTLOOP_OK",
            "reason": "within_target",
            "sample_count": 10,
            "p50_ms": 3.0,
            "p95_ms": 6.0,
            "p99_ms": 7.5,
            "p999_ms": 8.2,
            "jitter_ms": 5.2,
        }
    )
    recorder.save_latency_benchmark_artifact(
        {
            "run_id": "week3-run-1",
            "timestamp": ts,
            "passed": True,
            "reasons": [],
            "artifact": {"max_p99_ms": 7.5},
        }
    )
    recorder.save_promotion_gate_event(
        {
            "timestamp": ts,
            "policy_id": "phase3_student_v1",
            "from_stage": "candidate",
            "to_stage": "shadow",
            "approved": True,
            "reasons": [],
            "evidence": {"latency_gate_passed": True},
        }
    )
    recorder.save_rollback_drill_event(
        {
            "timestamp": ts,
            "failed_model_id": "phase3_student_v2",
            "reverted_to": "phase3_student_v1",
            "executed": True,
            "mttr_seconds": 42.0,
            "reasons": [],
        }
    )
    recorder.save_xai_trade_explanation(
        {
            "trade_id": "trade-1",
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "top_feature_contributions": [("orderbook_imbalance", 0.5)],
            "agent_contributions": [("technical", 0.4)],
            "signal_family_contributions": [("microstructure", 0.6)],
            "metadata": {"week3": True},
        }
    )
    recorder.save_pnl_attribution_event(
        {
            "trade_id": "trade-1",
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "sector": "equity",
            "agent": "technical",
            "signal_family": "momentum",
            "realized_pnl": 1200.0,
        }
    )
    recorder.save_operational_metrics_snapshot(
        {
            "timestamp": ts,
            "decision_staleness_avg_s": 0.5,
            "feature_lag_avg_s": 0.2,
            "mode_switch_frequency": 1,
            "ood_trigger_rate": 0,
            "kill_switch_false_positives": 0,
            "mttr_avg_s": 42.0,
        }
    )

    with recorder.Session() as session:
        assert session.execute(select(ImpactEventDB)).scalar_one() is not None
        assert session.execute(select(RiskCapEventDB)).scalar_one() is not None
        assert session.execute(select(OrderBookFeatureEventDB)).scalar_one() is not None
        assert session.execute(select(FastLoopLatencyEventDB)).scalar_one() is not None
        assert session.execute(select(LatencyBenchmarkArtifactDB)).scalar_one() is not None
        assert session.execute(select(PromotionGateEventDB)).scalar_one() is not None
        assert session.execute(select(RollbackDrillEventDB)).scalar_one() is not None
        assert session.execute(select(XAITradeExplanationDB)).scalar_one() is not None
        assert session.execute(select(PnLAttributionEventDB)).scalar_one() is not None
        assert session.execute(select(OperationalMetricsSnapshotDB)).scalar_one() is not None
