from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    DeliberationLogDB,
    DistillationRunDB,
    OrderDB,
    OrderFillDB,
    PolicySnapshotDB,
    PortfolioSnapshotDB,
    RewardLogDB,
    StudentPolicyDB,
    TradeDecisionDB,
)
from src.db.phase3_recorder import Phase3Recorder


def test_phase3_recorder_persists_core_database_plan_rows():
    engine = create_engine("sqlite:///:memory:")
    recorder = Phase3Recorder(engine=engine, session_factory=sessionmaker(bind=engine))
    ts = datetime(2026, 4, 21, 9, 30, tzinfo=UTC)

    recorder.save_policy_snapshot(
        {
            "snapshot_id": "snap-001",
            "policy_id": "phase3_sac_teacher_v1",
            "policy_type": "teacher",
            "generated_at": ts,
            "expires_at": ts + timedelta(minutes=10),
            "is_active": True,
            "artifact_path": "/tmp/snap-001.bin",
            "quality_status": "pass",
            "source_type": "scheduled_refresh",
        }
    )
    recorder.save_trade_decision(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "observation_id": 1,
            "policy_snapshot_id": "snap-001",
            "policy_id": "phase3_sac_teacher_v1",
            "action": "buy",
            "action_size": 100.0,
            "confidence": 0.82,
            "decision_latency_ms": 6.8,
            "loop_type": "fast",
            "genetic_threshold": 0.75,
            "deliberation_used": True,
        }
    )
    recorder.save_order(
        {
            "order_id": "order-001",
            "decision_id": 1,
            "symbol": "RELIANCE.NS",
            "exchange": "NSE",
            "product_type": "equity",
            "order_type": "limit",
            "side": "buy",
            "quantity": 25,
            "price": 2500.0,
            "status": "submitted",
            "model_version": "phase3_sac_teacher_v1",
            "compliance_check_passed": True,
            "created_at": ts,
            "updated_at": ts,
        }
    )
    recorder.save_order_fill(
        {
            "order_id": "order-001",
            "fill_timestamp": ts + timedelta(seconds=1),
            "fill_price": 2500.25,
            "fill_quantity": 25,
            "fees": 1.2,
            "impact_cost_bps": 3.4,
        }
    )
    recorder.save_portfolio_snapshot(
        {
            "timestamp": ts,
            "symbol": "RELIANCE.NS",
            "position_qty": 25,
            "avg_entry_price": 2500.25,
            "market_price": 2501.0,
            "unrealized_pnl": 18.75,
            "realized_pnl_session": 0.0,
            "realized_pnl_cumulative": 0.0,
            "notional_exposure": 62531.25,
            "net_exposure": 62531.25,
            "gross_exposure": 62531.25,
            "operating_mode": "normal",
            "decision_id": 1,
        }
    )
    recorder.save_reward_log(
        {
            "decision_id": 1,
            "timestamp": ts + timedelta(minutes=1),
            "symbol": "RELIANCE.NS",
            "policy_id": "phase3_sac_teacher_v1",
            "reward_function": "ra_drl_composite",
            "total_reward": 0.18,
            "return_component": 0.24,
            "risk_penalty": -0.03,
            "transaction_cost": -0.03,
            "regime_state": "bull",
        }
    )
    recorder.save_student_policy(
        {
            "student_id": "student_sac_v1_q8",
            "teacher_policy_id": "phase3_sac_teacher_v1",
            "version": "1.0",
            "status": "candidate",
            "compression_method": "distillation",
            "teacher_agreement_pct": 94.2,
            "crisis_agreement_pct": 90.1,
            "p99_inference_ms": 2.3,
            "p999_inference_ms": 3.1,
            "artifact_path": "/tmp/student_sac_v1_q8.bin",
            "drift_threshold": 0.15,
            "created_at": ts,
        }
    )
    recorder.save_distillation_run(
        {
            "student_id": "student_sac_v1_q8",
            "teacher_policy_id": "phase3_sac_teacher_v1",
            "run_timestamp": ts,
            "epochs": 15,
            "avg_day_agreement": 94.2,
            "crisis_slice_agreement": 90.1,
            "kl_divergence": 0.08,
            "inference_latency_p99": 2.3,
            "dataset_snapshot_id": "snap-ds-001",
            "code_hash": "abc123",
        }
    )
    recorder.save_deliberation_log(
        {
            "timestamp": ts + timedelta(minutes=2),
            "symbol": "RELIANCE.NS",
            "deliberation_type": "policy_refresh",
            "input_snapshot_id": "snap-001",
            "output_snapshot_id": "snap-002",
            "duration_ms": 182.0,
            "result": {"refreshed": True},
            "triggered_refresh": True,
        }
    )

    with recorder.Session() as session:
        decision = session.execute(select(TradeDecisionDB)).scalar_one()
        reward = session.execute(select(RewardLogDB)).scalar_one()
        assert session.execute(select(PolicySnapshotDB)).scalar_one() is not None
        assert session.execute(select(OrderDB)).scalar_one() is not None
        assert session.execute(select(OrderFillDB)).scalar_one() is not None
        assert session.execute(select(PortfolioSnapshotDB)).scalar_one() is not None
        assert session.execute(select(StudentPolicyDB)).scalar_one() is not None
        assert session.execute(select(DistillationRunDB)).scalar_one() is not None
        assert session.execute(select(DeliberationLogDB)).scalar_one() is not None
        assert decision.policy_snapshot_id == "snap-001"
        assert decision.deliberation_used is True
        assert reward.reward_function == "ra_drl_composite"
        assert reward.total_reward == 0.18


def test_phase3_recorder_repairs_legacy_trade_and_reward_columns():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE trade_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    observation_id INTEGER,
                    policy_id VARCHAR(128) NOT NULL,
                    action VARCHAR(16) NOT NULL,
                    action_size FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    entropy FLOAT,
                    sac_weight FLOAT,
                    ppo_weight FLOAT,
                    td3_weight FLOAT,
                    risk_override VARCHAR(16),
                    risk_override_reason TEXT,
                    decision_latency_ms FLOAT NOT NULL DEFAULT 0.0,
                    loop_type VARCHAR(8) NOT NULL DEFAULT 'slow',
                    policy_type VARCHAR(16) NOT NULL DEFAULT 'teacher',
                    is_placeholder BOOLEAN NOT NULL DEFAULT 1,
                    contract_version VARCHAR(16) NOT NULL DEFAULT 'strat_exec_v1',
                    schema_version VARCHAR(8) NOT NULL DEFAULT '1.0'
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE reward_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    policy_id VARCHAR(128) NOT NULL,
                    reward_name VARCHAR(32) NOT NULL,
                    reward_value FLOAT NOT NULL,
                    components_json TEXT,
                    schema_version VARCHAR(8) NOT NULL DEFAULT '1.0'
                );
                """
            )
        )

    recorder = Phase3Recorder(engine=engine, session_factory=sessionmaker(bind=engine))
    ts = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
    recorder.save_trade_decision(
        {
            "timestamp": ts,
            "symbol": "TCS.NS",
            "policy_id": "phase3_sac_teacher_v1",
            "action": "hold",
            "action_size": 0.0,
            "confidence": 0.5,
            "policy_snapshot_id": "legacy-snap-1",
            "genetic_threshold": 0.2,
            "deliberation_used": True,
            "decision_latency_ms": 4.0,
            "loop_type": "fast",
        }
    )
    recorder.save_reward_log(
        {
            "timestamp": ts,
            "symbol": "TCS.NS",
            "policy_id": "phase3_sac_teacher_v1",
            "reward_function": "ra_drl_composite",
            "total_reward": 0.11,
            "return_component": 0.14,
            "risk_penalty": -0.03,
        }
    )

    with recorder.Session() as session:
        decision = session.execute(select(TradeDecisionDB)).scalar_one()
        reward = session.execute(select(RewardLogDB)).scalar_one()
        assert decision.policy_snapshot_id == "legacy-snap-1"
        assert decision.genetic_threshold == 0.2
        assert decision.deliberation_used is True
        assert reward.reward_function == "ra_drl_composite"
        assert reward.total_reward == 0.11
