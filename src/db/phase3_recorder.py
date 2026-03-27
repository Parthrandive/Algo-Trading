from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.connection import get_engine, get_session
from src.db.models import (
    Base,
    DeliberationLogDB,
    DistillationRunDB,
    FastLoopLatencyEventDB,
    ImpactEventDB,
    LatencyBenchmarkArtifactDB,
    ObservationDB,
    OrderDB,
    OrderFillDB,
    OperationalMetricsSnapshotDB,
    OrderBookFeatureEventDB,
    PnLAttributionEventDB,
    PolicySnapshotDB,
    PortfolioSnapshotDB,
    PromotionGateEventDB,
    RLPolicyDB,
    RLTrainingRunDB,
    RewardLogDB,
    RiskEventDB,
    RiskCapEventDB,
    RiskStateSnapshotDB,
    RollbackDrillEventDB,
    StudentPolicyDB,
    TradeDecisionDB,
    XAITradeExplanationDB,
)

logger = logging.getLogger(__name__)

PHASE3_TABLES = (
    ObservationDB.__table__,
    RLPolicyDB.__table__,
    RLTrainingRunDB.__table__,
    TradeDecisionDB.__table__,
    RewardLogDB.__table__,
    PolicySnapshotDB.__table__,
    OrderDB.__table__,
    OrderFillDB.__table__,
    PortfolioSnapshotDB.__table__,
    StudentPolicyDB.__table__,
    DistillationRunDB.__table__,
    DeliberationLogDB.__table__,
    ImpactEventDB.__table__,
    RiskCapEventDB.__table__,
    OrderBookFeatureEventDB.__table__,
    FastLoopLatencyEventDB.__table__,
    LatencyBenchmarkArtifactDB.__table__,
    PromotionGateEventDB.__table__,
    RollbackDrillEventDB.__table__,
    XAITradeExplanationDB.__table__,
    PnLAttributionEventDB.__table__,
    OperationalMetricsSnapshotDB.__table__,
    RiskStateSnapshotDB.__table__,
    RiskEventDB.__table__,
)

PHASE3_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_observations_symbol_ts ON observations (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_policies_status ON rl_policies (status, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_training_runs_policy_ts ON rl_training_runs (policy_id, run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_trade_decisions_symbol_ts ON trade_decisions (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_trade_decisions_latency ON trade_decisions (decision_latency_ms DESC);",
    "CREATE INDEX IF NOT EXISTS idx_reward_logs_symbol_ts ON reward_logs (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_policy_snapshots_active_policy ON policy_snapshots (is_active, policy_type);",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders (symbol, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);",
    "CREATE INDEX IF NOT EXISTS idx_orders_decision_id ON orders (decision_id);",
    "CREATE INDEX IF NOT EXISTS idx_order_fills_order_id_ts ON order_fills (order_id, fill_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_symbol_ts ON portfolio_snapshots (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_mode ON portfolio_snapshots (operating_mode);",
    "CREATE INDEX IF NOT EXISTS idx_student_policies_status ON student_policies (status, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_distillation_runs_student_ts ON distillation_runs (student_id, run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_deliberation_logs_symbol_ts ON deliberation_logs (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_impact_events_symbol_ts ON impact_events (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_risk_cap_events_symbol_ts ON risk_cap_events (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_orderbook_feature_events_symbol_ts ON orderbook_feature_events (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fastloop_latency_events_stage_ts ON fastloop_latency_events (stage, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_latency_benchmark_artifacts_run_ts ON latency_benchmark_artifacts (run_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_promotion_gate_events_policy_ts ON promotion_gate_events (policy_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rollback_drill_events_failed_ts ON rollback_drill_events (failed_model_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_xai_trade_explanations_trade_ts ON xai_trade_explanations (trade_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_pnl_attribution_events_symbol_ts ON pnl_attribution_events (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_operational_metrics_snapshots_ts ON operational_metrics_snapshots (timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_risk_state_snapshots_symbol_ts ON risk_state_snapshots (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_risk_events_symbol_ts ON risk_events (symbol, timestamp DESC);",
)

PHASE3_COLUMN_REPAIR_DDL: dict[str, dict[str, str]] = {
    "observations": {
        "event_id": "ALTER TABLE observations ADD COLUMN event_id VARCHAR(64);",
    },
    "trade_decisions": {
        "event_id": "ALTER TABLE trade_decisions ADD COLUMN event_id VARCHAR(64);",
        "policy_snapshot_id": "ALTER TABLE trade_decisions ADD COLUMN policy_snapshot_id VARCHAR(128);",
        "observation_event_id": "ALTER TABLE trade_decisions ADD COLUMN observation_event_id VARCHAR(64);",
        "observation_timestamp": "ALTER TABLE trade_decisions ADD COLUMN observation_timestamp TIMESTAMPTZ;",
        "genetic_threshold": "ALTER TABLE trade_decisions ADD COLUMN genetic_threshold FLOAT;",
        "deliberation_used": (
            "ALTER TABLE trade_decisions ADD COLUMN deliberation_used BOOLEAN NOT NULL DEFAULT FALSE;"
        ),
    },
    "orders": {
        "event_id": "ALTER TABLE orders ADD COLUMN event_id VARCHAR(64);",
        "decision_event_id": "ALTER TABLE orders ADD COLUMN decision_event_id VARCHAR(64);",
        "decision_timestamp": "ALTER TABLE orders ADD COLUMN decision_timestamp TIMESTAMPTZ;",
    },
    "order_fills": {
        "event_id": "ALTER TABLE order_fills ADD COLUMN event_id VARCHAR(64);",
        "order_event_id": "ALTER TABLE order_fills ADD COLUMN order_event_id VARCHAR(64);",
        "order_created_at": "ALTER TABLE order_fills ADD COLUMN order_created_at TIMESTAMPTZ;",
    },
    "reward_logs": {
        "event_id": "ALTER TABLE reward_logs ADD COLUMN event_id VARCHAR(64);",
        "reward_function": (
            "ALTER TABLE reward_logs ADD COLUMN reward_function VARCHAR(32) NOT NULL DEFAULT 'ra_drl_composite';"
        ),
        "total_reward": "ALTER TABLE reward_logs ADD COLUMN total_reward FLOAT NOT NULL DEFAULT 0.0;",
        "return_component": "ALTER TABLE reward_logs ADD COLUMN return_component FLOAT;",
        "risk_penalty": "ALTER TABLE reward_logs ADD COLUMN risk_penalty FLOAT;",
        "regime_weight": "ALTER TABLE reward_logs ADD COLUMN regime_weight FLOAT;",
        "sentiment_weight": "ALTER TABLE reward_logs ADD COLUMN sentiment_weight FLOAT;",
        "transaction_cost": "ALTER TABLE reward_logs ADD COLUMN transaction_cost FLOAT;",
        "regime_state": "ALTER TABLE reward_logs ADD COLUMN regime_state VARCHAR(32);",
    },
    "portfolio_snapshots": {
        "event_id": "ALTER TABLE portfolio_snapshots ADD COLUMN event_id VARCHAR(64);",
    },
    "deliberation_logs": {
        "event_id": "ALTER TABLE deliberation_logs ADD COLUMN event_id VARCHAR(64);",
    },
}


class Phase3Recorder:
    """Persist Phase 3 strategic artifacts and operational records."""

    def __init__(
        self,
        database_url: str | None = None,
        engine=None,
        session_factory=None,
        *,
        bootstrap_schema: bool = True,
    ):
        self.engine = engine or get_engine(database_url)
        self._bootstrap_schema = bool(bootstrap_schema)
        if self._bootstrap_schema:
            self._ensure_phase3_schema()
        self.Session = session_factory or get_session(self.engine)

    def _ensure_phase3_schema(self) -> None:
        """
        Idempotently creates the Phase 3 strategic tables and indexes.
        Mirrors Phase2Recorder._ensure_phase2_schema() for consistency.
        """
        try:
            Base.metadata.create_all(self.engine, tables=list(PHASE3_TABLES))
            with self.engine.begin() as conn:
                self._repair_phase3_columns(conn)
                for ddl in PHASE3_INDEX_DDL:
                    conn.execute(text(ddl))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to bootstrap Phase 3 recorder schema: {type(exc).__name__}: {exc}"
            ) from exc

    def _repair_phase3_columns(self, conn) -> None:
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())
        for table_name, column_ddls in PHASE3_COLUMN_REPAIR_DDL.items():
            if table_name not in table_names:
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, ddl in column_ddls.items():
                if column_name in existing_columns:
                    continue
                conn.execute(text(ddl))

    def save_observation(self, observation: Any) -> None:
        observation = self._ensure_dict(observation)
        row = self._normalize_observation(observation)
        with self.Session() as session:
            self._upsert(session, ObservationDB, row, key_fields=("symbol", "timestamp", "snapshot_id"))
            session.commit()

    def save_observation_batch(self, observations: list[Any]) -> None:
        if not observations:
            return
        with self.Session() as session:
            for observation in observations:
                observation = self._ensure_dict(observation)
                row = self._normalize_observation(observation)
                self._upsert(session, ObservationDB, row, key_fields=("symbol", "timestamp", "snapshot_id"))
            session.commit()

    def save_rl_policy(self, policy: Any) -> None:
        policy = self._ensure_dict(policy)
        now = datetime.now(timezone.utc)
        row = {
            "policy_id": str(policy["policy_id"]),
            "algorithm": str(policy["algorithm"]),
            "version": str(policy.get("version", "1.0")),
            "status": str(policy.get("status", "candidate")),
            "created_at": self._coerce_datetime(policy.get("created_at", now)),
            "promoted_at": self._optional_datetime(policy.get("promoted_at")),
            "retired_at": self._optional_datetime(policy.get("retired_at")),
            "artifact_path": str(policy.get("artifact_path", "")),
            "observation_schema_version": str(policy.get("observation_schema_version", "1.0")),
            "reward_function": str(policy.get("reward_function", "ra_drl_composite")),
            "hyperparams_json": self._to_json(policy.get("hyperparams", {})),
            "training_metrics_json": self._to_json(policy.get("training_metrics"))
            if policy.get("training_metrics") is not None
            else None,
            "compression_method": policy.get("compression_method"),
            "p99_inference_ms": self._optional_float(policy.get("p99_inference_ms")),
            "p999_inference_ms": self._optional_float(policy.get("p999_inference_ms")),
            "schema_version": str(policy.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            self._upsert(session, RLPolicyDB, row, key_fields=("policy_id",))
            session.commit()

    def save_rl_training_run(self, run: Any) -> None:
        run = self._ensure_dict(run)
        row = {
            "policy_id": str(run["policy_id"]),
            "run_timestamp": self._coerce_datetime(run.get("started_at", datetime.now(timezone.utc))),
            "training_start": self._coerce_datetime(run.get("train_start", run.get("started_at"))),
            "training_end": self._coerce_datetime(run.get("train_end", run.get("completed_at"))),
            "episodes": int(run.get("episodes", 0)),
            "total_steps": int(run.get("total_steps", 0)),
            "final_reward": self._optional_float(run.get("final_reward")),
            "sharpe": self._optional_float(run.get("sharpe")),
            "sortino": self._optional_float(run.get("sortino")),
            "max_drawdown": self._optional_float(run.get("max_drawdown")),
            "win_rate": self._optional_float(run.get("win_rate")),
            "dataset_snapshot_id": run.get("dataset_snapshot_id"),
            "code_hash": run.get("code_hash"),
            "duration_seconds": self._optional_float(run.get("duration_seconds")),
            "notes": run.get("notes"),
        }
        with self.Session() as session:
            session.add(RLTrainingRunDB(**row))
            session.commit()

    def save_training_run(self, run: Any) -> None:
        """Compatibility alias for legacy recorder call sites."""
        self.save_rl_training_run(run)

    def save_trade_decision(self, decision: dict[str, Any]) -> None:
        timestamp = self._coerce_datetime(decision["timestamp"])
        event_id = str(
            decision.get("event_id")
            or self._stable_event_id(
                "trade_decision",
                decision.get("symbol"),
                timestamp.isoformat(),
                decision.get("policy_snapshot_id"),
                decision.get("policy_id"),
                decision.get("action"),
            )
        )
        row = {
            "event_id": event_id,
            "id": self._coerce_or_stable_id(
                decision.get("id"),
                "trade_decision",
                decision.get("symbol"),
                timestamp.isoformat(),
                decision.get("policy_id"),
                decision.get("action"),
            ),
            "timestamp": timestamp,
            "symbol": str(decision["symbol"]),
            "observation_id": decision.get("observation_id"),
            "observation_event_id": decision.get("observation_event_id"),
            "observation_timestamp": self._optional_datetime(decision.get("observation_timestamp")),
            "policy_snapshot_id": decision.get("policy_snapshot_id"),
            "policy_id": str(decision["policy_id"]),
            "action": str(decision["action"]),
            "action_size": float(decision["action_size"]),
            "confidence": float(decision.get("confidence", 0.0)),
            "entropy": self._optional_float(decision.get("entropy")),
            "sac_weight": self._optional_float(decision.get("sac_weight")),
            "ppo_weight": self._optional_float(decision.get("ppo_weight")),
            "td3_weight": self._optional_float(decision.get("td3_weight")),
            "genetic_threshold": self._optional_float(decision.get("genetic_threshold")),
            "deliberation_used": bool(decision.get("deliberation_used", False)),
            "risk_override": decision.get("risk_override"),
            "risk_override_reason": decision.get("risk_override_reason"),
            "decision_latency_ms": float(decision.get("decision_latency_ms", 0.0)),
            "loop_type": str(decision.get("loop_type", "slow")),
            "policy_type": str(decision.get("policy_type", "teacher")),
            "is_placeholder": bool(decision.get("is_placeholder", True)),
            "contract_version": str(decision.get("contract_version", "strat_exec_v1")),
            "schema_version": str(decision.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(TradeDecisionDB(**row))
            session.commit()

    def save_trade_decision_batch(self, decisions: list[Any]) -> None:
        if not decisions:
            return
        with self.Session() as session:
            for decision in decisions:
                decision = self._ensure_dict(decision)
                timestamp = self._coerce_datetime(decision["timestamp"])
                event_id = str(
                    decision.get("event_id")
                    or self._stable_event_id(
                        "trade_decision",
                        decision.get("symbol"),
                        timestamp.isoformat(),
                        decision.get("policy_snapshot_id"),
                        decision.get("policy_id"),
                        decision.get("action"),
                    )
                )
                row = {
                    "event_id": event_id,
                    "id": self._coerce_or_stable_id(
                        decision.get("id"),
                        "trade_decision",
                        decision.get("symbol"),
                        timestamp.isoformat(),
                        decision.get("policy_id"),
                        decision.get("action"),
                    ),
                    "timestamp": timestamp,
                    "symbol": str(decision["symbol"]),
                    "observation_id": decision.get("observation_id"),
                    "observation_event_id": decision.get("observation_event_id"),
                    "observation_timestamp": self._optional_datetime(decision.get("observation_timestamp")),
                    "policy_snapshot_id": decision.get("policy_snapshot_id"),
                    "policy_id": str(decision["policy_id"]),
                    "action": str(decision["action"]),
                    "action_size": float(decision["action_size"]),
                    "confidence": float(decision.get("confidence", 0.0)),
                    "entropy": self._optional_float(decision.get("entropy")),
                    "sac_weight": self._optional_float(decision.get("sac_weight")),
                    "ppo_weight": self._optional_float(decision.get("ppo_weight")),
                    "td3_weight": self._optional_float(decision.get("td3_weight")),
                    "genetic_threshold": self._optional_float(decision.get("genetic_threshold")),
                    "deliberation_used": bool(decision.get("deliberation_used", False)),
                    "risk_override": decision.get("risk_override"),
                    "risk_override_reason": decision.get("risk_override_reason"),
                    "decision_latency_ms": float(decision.get("decision_latency_ms", 0.0)),
                    "loop_type": str(decision.get("loop_type", "slow")),
                    "policy_type": str(decision.get("policy_type", "teacher")),
                    "is_placeholder": bool(decision.get("is_placeholder", True)),
                    "contract_version": str(decision.get("contract_version", "strat_exec_v1")),
                    "schema_version": str(decision.get("schema_version", "1.0")),
                }
                session.add(TradeDecisionDB(**row))
            session.commit()

    def save_reward_log(self, reward: Any) -> None:
        reward = self._ensure_dict(reward)
        reward_name = str(reward.get("reward_name", reward.get("reward_function", "ra_drl_composite")))
        reward_value = float(reward.get("reward_value", reward.get("total_reward", 0.0)))
        reward_timestamp = self._coerce_datetime(reward["timestamp"])
        row = {
            "id": self._coerce_or_stable_id(
                reward.get("id"),
                "reward_log",
                reward.get("symbol"),
                reward_timestamp.isoformat(),
                reward.get("policy_id"),
                reward_name,
            ),
            "decision_id": reward.get("decision_id"),
            "timestamp": reward_timestamp,
            "symbol": str(reward["symbol"]),
            "policy_id": str(reward.get("policy_id", "foundation")),
            "reward_name": reward_name,
            "reward_value": reward_value,
            "reward_function": str(reward.get("reward_function", reward_name)),
            "total_reward": float(reward.get("total_reward", reward_value)),
            "return_component": self._optional_float(reward.get("return_component")),
            "risk_penalty": self._optional_float(reward.get("risk_penalty")),
            "regime_weight": self._optional_float(reward.get("regime_weight")),
            "sentiment_weight": self._optional_float(reward.get("sentiment_weight")),
            "transaction_cost": self._optional_float(reward.get("transaction_cost")),
            "regime_state": reward.get("regime_state"),
            "components_json": self._to_json(reward.get("components")) if reward.get("components") is not None else None,
            "schema_version": str(reward.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RewardLogDB(**row))
            session.commit()

    def save_reward_log_batch(self, rewards: list[Any]) -> None:
        if not rewards:
            return
        with self.Session() as session:
            for reward in rewards:
                reward = self._ensure_dict(reward)
                reward_name = str(reward.get("reward_name", reward.get("reward_function", "ra_drl_composite")))
                reward_value = float(reward.get("reward_value", reward.get("total_reward", 0.0)))
                reward_timestamp = self._coerce_datetime(reward["timestamp"])
                row = {
                    "id": self._coerce_or_stable_id(
                        reward.get("id"),
                        "reward_log",
                        reward.get("symbol"),
                        reward_timestamp.isoformat(),
                        reward.get("policy_id"),
                        reward_name,
                    ),
                    "decision_id": reward.get("decision_id"),
                    "timestamp": reward_timestamp,
                    "symbol": str(reward["symbol"]),
                    "policy_id": str(reward.get("policy_id", "foundation")),
                    "reward_name": reward_name,
                    "reward_value": reward_value,
                    "reward_function": str(reward.get("reward_function", reward_name)),
                    "total_reward": float(reward.get("total_reward", reward_value)),
                    "return_component": self._optional_float(reward.get("return_component")),
                    "risk_penalty": self._optional_float(reward.get("risk_penalty")),
                    "regime_weight": self._optional_float(reward.get("regime_weight")),
                    "sentiment_weight": self._optional_float(reward.get("sentiment_weight")),
                    "transaction_cost": self._optional_float(reward.get("transaction_cost")),
                    "regime_state": reward.get("regime_state"),
                    "components_json": self._to_json(reward.get("components"))
                    if reward.get("components") is not None
                    else None,
                    "schema_version": str(reward.get("schema_version", "1.0")),
                }
                session.add(RewardLogDB(**row))
            session.commit()

    def save_policy_snapshot(self, snapshot: Any) -> None:
        snapshot = self._ensure_dict(snapshot)
        row = {
            "snapshot_id": str(snapshot["snapshot_id"]),
            "policy_id": str(snapshot["policy_id"]),
            "policy_type": str(snapshot.get("policy_type", "student")),
            "generated_at": self._coerce_datetime(snapshot["generated_at"]),
            "expires_at": self._coerce_datetime(snapshot["expires_at"]),
            "is_active": bool(snapshot.get("is_active", False)),
            "artifact_path": str(snapshot.get("artifact_path", "")),
            "quality_status": str(snapshot.get("quality_status", "pass")),
            "source_type": str(snapshot.get("source_type", "scheduled_refresh")),
            "schema_version": str(snapshot.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            self._upsert(session, PolicySnapshotDB, row, key_fields=("snapshot_id",))
            session.commit()

    def save_policy_snapshot_batch(self, snapshots: list[Any]) -> None:
        if not snapshots:
            return
        with self.Session() as session:
            for snapshot in snapshots:
                snapshot = self._ensure_dict(snapshot)
                row = {
                    "snapshot_id": str(snapshot["snapshot_id"]),
                    "policy_id": str(snapshot["policy_id"]),
                    "policy_type": str(snapshot.get("policy_type", "student")),
                    "generated_at": self._coerce_datetime(snapshot["generated_at"]),
                    "expires_at": self._coerce_datetime(snapshot["expires_at"]),
                    "is_active": bool(snapshot.get("is_active", False)),
                    "artifact_path": str(snapshot.get("artifact_path", "")),
                    "quality_status": str(snapshot.get("quality_status", "pass")),
                    "source_type": str(snapshot.get("source_type", "scheduled_refresh")),
                    "schema_version": str(snapshot.get("schema_version", "1.0")),
                }
                self._upsert(session, PolicySnapshotDB, row, key_fields=("snapshot_id",))
            session.commit()

    def save_order(self, order: Any) -> None:
        order = self._ensure_dict(order)
        now = datetime.now(timezone.utc)
        created_at = self._coerce_datetime(order.get("created_at", now))
        event_id = str(
            order.get("event_id")
            or self._stable_event_id(
                "order",
                order.get("order_id"),
                created_at.isoformat(),
            )
        )
        row = {
            "event_id": event_id,
            "id": self._coerce_or_stable_id(
                order.get("id"),
                "order",
                order.get("order_id"),
                created_at.isoformat(),
            ),
            "created_at": created_at,
            "order_id": str(order["order_id"]),
            "decision_id": order.get("decision_id"),
            "decision_event_id": order.get("decision_event_id"),
            "decision_timestamp": self._optional_datetime(order.get("decision_timestamp")),
            "symbol": str(order["symbol"]),
            "exchange": str(order.get("exchange", "NSE")),
            "product_type": str(order.get("product_type", "equity")),
            "order_type": str(order["order_type"]),
            "side": str(order["side"]),
            "quantity": int(order["quantity"]),
            "price": self._optional_float(order.get("price")),
            "trigger_price": self._optional_float(order.get("trigger_price")),
            "status": str(order.get("status", "pending")),
            "submitted_at": self._optional_datetime(order.get("submitted_at")),
            "filled_at": self._optional_datetime(order.get("filled_at")),
            "cancelled_at": self._optional_datetime(order.get("cancelled_at")),
            "broker_order_id": order.get("broker_order_id"),
            "avg_fill_price": self._optional_float(order.get("avg_fill_price")),
            "filled_quantity": int(order.get("filled_quantity", 0)),
            "slippage_bps": self._optional_float(order.get("slippage_bps")),
            "model_version": str(order.get("model_version", "unknown")),
            "compliance_check_passed": bool(order.get("compliance_check_passed", True)),
            "rejection_reason": order.get("rejection_reason"),
            "updated_at": self._coerce_datetime(order.get("updated_at", now)),
            "schema_version": str(order.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            self._upsert(session, OrderDB, row, key_fields=("order_id", "created_at"))
            session.commit()

    def save_order_batch(self, orders: list[Any]) -> None:
        if not orders:
            return
        with self.Session() as session:
            now = datetime.now(timezone.utc)
            for order in orders:
                order = self._ensure_dict(order)
                created_at = self._coerce_datetime(order.get("created_at", now))
                event_id = str(
                    order.get("event_id")
                    or self._stable_event_id(
                        "order",
                        order.get("order_id"),
                        created_at.isoformat(),
                    )
                )
                row = {
                    "event_id": event_id,
                    "id": self._coerce_or_stable_id(
                        order.get("id"),
                        "order",
                        order.get("order_id"),
                        created_at.isoformat(),
                    ),
                    "created_at": created_at,
                    "order_id": str(order["order_id"]),
                    "decision_id": order.get("decision_id"),
                    "decision_event_id": order.get("decision_event_id"),
                    "decision_timestamp": self._optional_datetime(order.get("decision_timestamp")),
                    "symbol": str(order["symbol"]),
                    "exchange": str(order.get("exchange", "NSE")),
                    "product_type": str(order.get("product_type", "equity")),
                    "order_type": str(order["order_type"]),
                    "side": str(order["side"]),
                    "quantity": int(order["quantity"]),
                    "price": self._optional_float(order.get("price")),
                    "trigger_price": self._optional_float(order.get("trigger_price")),
                    "status": str(order.get("status", "pending")),
                    "submitted_at": self._optional_datetime(order.get("submitted_at")),
                    "filled_at": self._optional_datetime(order.get("filled_at")),
                    "cancelled_at": self._optional_datetime(order.get("cancelled_at")),
                    "broker_order_id": order.get("broker_order_id"),
                    "avg_fill_price": self._optional_float(order.get("avg_fill_price")),
                    "filled_quantity": int(order.get("filled_quantity", 0)),
                    "slippage_bps": self._optional_float(order.get("slippage_bps")),
                    "model_version": str(order.get("model_version", "unknown")),
                    "compliance_check_passed": bool(order.get("compliance_check_passed", True)),
                    "rejection_reason": order.get("rejection_reason"),
                    "updated_at": self._coerce_datetime(order.get("updated_at", now)),
                    "schema_version": str(order.get("schema_version", "1.0")),
                }
                self._upsert(session, OrderDB, row, key_fields=("order_id", "created_at"))
            session.commit()

    def save_order_fill(self, fill: Any) -> None:
        fill = self._ensure_dict(fill)
        fill_timestamp = self._coerce_datetime(fill["fill_timestamp"])
        event_id = str(
            fill.get("event_id")
            or self._stable_event_id(
                "order_fill",
                fill.get("order_id"),
                fill_timestamp.isoformat(),
                fill.get("exchange_trade_id"),
                fill.get("fill_quantity"),
            )
        )
        row = {
            "event_id": event_id,
            "id": self._coerce_or_stable_id(
                fill.get("id"),
                "order_fill",
                fill.get("order_id"),
                fill_timestamp.isoformat(),
                fill.get("exchange_trade_id"),
            ),
            "order_id": str(fill["order_id"]),
            "order_event_id": fill.get("order_event_id"),
            "order_created_at": self._optional_datetime(fill.get("order_created_at")),
            "fill_timestamp": fill_timestamp,
            "fill_price": float(fill["fill_price"]),
            "fill_quantity": int(fill["fill_quantity"]),
            "exchange_trade_id": fill.get("exchange_trade_id"),
            "fees": self._optional_float(fill.get("fees")),
            "impact_cost_bps": self._optional_float(fill.get("impact_cost_bps")),
            "schema_version": str(fill.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            self._upsert(session, OrderFillDB, row, key_fields=("event_id", "fill_timestamp"))
            session.commit()

    def save_order_fill_batch(self, fills: list[Any]) -> None:
        if not fills:
            return
        with self.Session() as session:
            for fill in fills:
                fill = self._ensure_dict(fill)
                fill_timestamp = self._coerce_datetime(fill["fill_timestamp"])
                event_id = str(
                    fill.get("event_id")
                    or self._stable_event_id(
                        "order_fill",
                        fill.get("order_id"),
                        fill_timestamp.isoformat(),
                        fill.get("exchange_trade_id"),
                        fill.get("fill_quantity"),
                    )
                )
                row = {
                    "event_id": event_id,
                    "id": self._coerce_or_stable_id(
                        fill.get("id"),
                        "order_fill",
                        fill.get("order_id"),
                        fill_timestamp.isoformat(),
                        fill.get("exchange_trade_id"),
                    ),
                    "order_id": str(fill["order_id"]),
                    "order_event_id": fill.get("order_event_id"),
                    "order_created_at": self._optional_datetime(fill.get("order_created_at")),
                    "fill_timestamp": fill_timestamp,
                    "fill_price": float(fill["fill_price"]),
                    "fill_quantity": int(fill["fill_quantity"]),
                    "exchange_trade_id": fill.get("exchange_trade_id"),
                    "fees": self._optional_float(fill.get("fees")),
                    "impact_cost_bps": self._optional_float(fill.get("impact_cost_bps")),
                    "schema_version": str(fill.get("schema_version", "1.0")),
                }
                self._upsert(session, OrderFillDB, row, key_fields=("event_id", "fill_timestamp"))
            session.commit()

    def save_portfolio_snapshot(self, snapshot: Any) -> None:
        snapshot = self._ensure_dict(snapshot)
        row = {
            "timestamp": self._coerce_datetime(snapshot["timestamp"]),
            "symbol": str(snapshot["symbol"]),
            "position_qty": int(snapshot.get("position_qty", 0)),
            "avg_entry_price": float(snapshot.get("avg_entry_price", 0.0)),
            "market_price": float(snapshot.get("market_price", 0.0)),
            "unrealized_pnl": float(snapshot.get("unrealized_pnl", 0.0)),
            "realized_pnl_session": float(snapshot.get("realized_pnl_session", 0.0)),
            "realized_pnl_cumulative": float(snapshot.get("realized_pnl_cumulative", 0.0)),
            "notional_exposure": float(snapshot.get("notional_exposure", 0.0)),
            "net_exposure": float(snapshot.get("net_exposure", 0.0)),
            "gross_exposure": float(snapshot.get("gross_exposure", 0.0)),
            "sector": snapshot.get("sector"),
            "risk_budget_used_pct": self._optional_float(snapshot.get("risk_budget_used_pct")),
            "operating_mode": str(snapshot.get("operating_mode", "normal")),
            "decision_id": snapshot.get("decision_id"),
            "schema_version": str(snapshot.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(PortfolioSnapshotDB(**row))
            session.commit()

    def save_portfolio_snapshot_batch(self, snapshots: list[Any]) -> None:
        if not snapshots:
            return
        with self.Session() as session:
            for snapshot in snapshots:
                snapshot = self._ensure_dict(snapshot)
                row = {
                    "timestamp": self._coerce_datetime(snapshot["timestamp"]),
                    "symbol": str(snapshot["symbol"]),
                    "position_qty": int(snapshot.get("position_qty", 0)),
                    "avg_entry_price": float(snapshot.get("avg_entry_price", 0.0)),
                    "market_price": float(snapshot.get("market_price", 0.0)),
                    "unrealized_pnl": float(snapshot.get("unrealized_pnl", 0.0)),
                    "realized_pnl_session": float(snapshot.get("realized_pnl_session", 0.0)),
                    "realized_pnl_cumulative": float(snapshot.get("realized_pnl_cumulative", 0.0)),
                    "notional_exposure": float(snapshot.get("notional_exposure", 0.0)),
                    "net_exposure": float(snapshot.get("net_exposure", 0.0)),
                    "gross_exposure": float(snapshot.get("gross_exposure", 0.0)),
                    "sector": snapshot.get("sector"),
                    "risk_budget_used_pct": self._optional_float(snapshot.get("risk_budget_used_pct")),
                    "operating_mode": str(snapshot.get("operating_mode", "normal")),
                    "decision_id": snapshot.get("decision_id"),
                    "schema_version": str(snapshot.get("schema_version", "1.0")),
                }
                session.add(PortfolioSnapshotDB(**row))
            session.commit()

    def save_student_policy(self, student: Any) -> None:
        student = self._ensure_dict(student)
        now = datetime.now(timezone.utc)
        row = {
            "student_id": str(student["student_id"]),
            "teacher_policy_id": str(student["teacher_policy_id"]),
            "version": str(student.get("version", "1.0")),
            "status": str(student.get("status", "candidate")),
            "compression_method": str(student.get("compression_method", "distillation")),
            "compression_ratio": self._optional_float(student.get("compression_ratio")),
            "teacher_agreement_pct": float(student.get("teacher_agreement_pct", 0.0)),
            "crisis_agreement_pct": float(student.get("crisis_agreement_pct", 0.0)),
            "p99_inference_ms": float(student.get("p99_inference_ms", 0.0)),
            "p999_inference_ms": float(student.get("p999_inference_ms", 0.0)),
            "artifact_path": str(student.get("artifact_path", "")),
            "created_at": self._coerce_datetime(student.get("created_at", now)),
            "promoted_at": self._optional_datetime(student.get("promoted_at")),
            "demoted_at": self._optional_datetime(student.get("demoted_at")),
            "drift_threshold": float(student.get("drift_threshold", 0.0)),
            "schema_version": str(student.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            self._upsert(session, StudentPolicyDB, row, key_fields=("student_id",))
            session.commit()

    def save_distillation_run(self, run: Any) -> None:
        run = self._ensure_dict(run)
        row = {
            "student_id": str(run["student_id"]),
            "teacher_policy_id": str(run["teacher_policy_id"]),
            "run_timestamp": self._coerce_datetime(run.get("run_timestamp", datetime.now(timezone.utc))),
            "epochs": int(run.get("epochs", 0)),
            "avg_day_agreement": float(run.get("avg_day_agreement", 0.0)),
            "crisis_slice_agreement": float(run.get("crisis_slice_agreement", 0.0)),
            "kl_divergence": self._optional_float(run.get("kl_divergence")),
            "inference_latency_p99": float(run.get("inference_latency_p99", 0.0)),
            "dataset_snapshot_id": run.get("dataset_snapshot_id"),
            "code_hash": run.get("code_hash"),
            "notes": run.get("notes"),
            "schema_version": str(run.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(DistillationRunDB(**row))
            session.commit()

    def save_deliberation_log(self, log: Any) -> None:
        log = self._ensure_dict(log)
        row = {
            "timestamp": self._coerce_datetime(log["timestamp"]),
            "symbol": str(log["symbol"]),
            "deliberation_type": str(log.get("deliberation_type", "policy_refresh")),
            "input_snapshot_id": log.get("input_snapshot_id"),
            "output_snapshot_id": log.get("output_snapshot_id"),
            "duration_ms": float(log.get("duration_ms", 0.0)),
            "result_json": self._to_json(log.get("result", log.get("result_json", {}))),
            "triggered_refresh": bool(log.get("triggered_refresh", False)),
            "schema_version": str(log.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(DeliberationLogDB(**row))
            session.commit()

    def save_deliberation_log_batch(self, logs: list[Any]) -> None:
        if not logs:
            return
        with self.Session() as session:
            for log in logs:
                log = self._ensure_dict(log)
                row = {
                    "timestamp": self._coerce_datetime(log["timestamp"]),
                    "symbol": str(log["symbol"]),
                    "deliberation_type": str(log.get("deliberation_type", "policy_refresh")),
                    "input_snapshot_id": log.get("input_snapshot_id"),
                    "output_snapshot_id": log.get("output_snapshot_id"),
                    "duration_ms": float(log.get("duration_ms", 0.0)),
                    "result_json": self._to_json(log.get("result", log.get("result_json", {}))),
                    "triggered_refresh": bool(log.get("triggered_refresh", False)),
                    "schema_version": str(log.get("schema_version", "1.0")),
                }
                session.add(DeliberationLogDB(**row))
            session.commit()

    def save_impact_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "symbol": str(event["symbol"]),
            "bucket": str(event["bucket"]),
            "event_type": str(event.get("event_type", "IMPACT_OK")),
            "breach": bool(event.get("breach", False)),
            "participation_rate": float(event.get("participation_rate", 0.0)),
            "slippage_delta_bps": float(event.get("slippage_delta_bps", 0.0)),
            "impact_score": float(event.get("impact_score", 0.0)),
            "size_multiplier": float(event.get("size_multiplier", 1.0)),
            "cooldown_until": self._optional_datetime(event.get("cooldown_until")),
            "risk_override": event.get("risk_override"),
            "reasons_json": self._to_json(event.get("reasons", [])),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(ImpactEventDB(**row))
            session.commit()

    def save_impact_event_batch(self, events: list[Any]) -> None:
        if not events:
            return
        with self.Session() as session:
            for event in events:
                event = self._ensure_dict(event)
                row = {
                    "timestamp": self._coerce_datetime(event["timestamp"]),
                    "symbol": str(event["symbol"]),
                    "bucket": str(event["bucket"]),
                    "event_type": str(event.get("event_type", "IMPACT_OK")),
                    "breach": bool(event.get("breach", False)),
                    "participation_rate": float(event.get("participation_rate", 0.0)),
                    "slippage_delta_bps": float(event.get("slippage_delta_bps", 0.0)),
                    "impact_score": float(event.get("impact_score", 0.0)),
                    "size_multiplier": float(event.get("size_multiplier", 1.0)),
                    "cooldown_until": self._optional_datetime(event.get("cooldown_until")),
                    "risk_override": event.get("risk_override"),
                    "reasons_json": self._to_json(event.get("reasons", [])),
                    "schema_version": str(event.get("schema_version", "1.0")),
                }
                session.add(ImpactEventDB(**row))
            session.commit()

    def save_risk_cap_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "symbol": str(event["symbol"]),
            "asset_cluster": str(event["asset_cluster"]),
            "regime": str(event["regime"]),
            "cap_fraction": float(event["cap_fraction"]),
            "changed": bool(event.get("changed", False)),
            "event_type": str(event.get("event_type", "RISK_CAP_STEADY")),
            "false_trigger_rate": float(event.get("false_trigger_rate", 0.0)),
            "auto_adjustment_paused": bool(event.get("auto_adjustment_paused", False)),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RiskCapEventDB(**row))
            session.commit()

    def save_risk_cap_event_batch(self, events: list[Any]) -> None:
        if not events:
            return
        with self.Session() as session:
            for event in events:
                event = self._ensure_dict(event)
                row = {
                    "timestamp": self._coerce_datetime(event["timestamp"]),
                    "symbol": str(event["symbol"]),
                    "asset_cluster": str(event["asset_cluster"]),
                    "regime": str(event["regime"]),
                    "cap_fraction": float(event["cap_fraction"]),
                    "changed": bool(event.get("changed", False)),
                    "event_type": str(event.get("event_type", "RISK_CAP_STEADY")),
                    "false_trigger_rate": float(event.get("false_trigger_rate", 0.0)),
                    "auto_adjustment_paused": bool(event.get("auto_adjustment_paused", False)),
                    "schema_version": str(event.get("schema_version", "1.0")),
                }
                session.add(RiskCapEventDB(**row))
            session.commit()

    def save_orderbook_feature_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "symbol": str(event["symbol"]),
            "quality_flag": str(event.get("quality_flag", "pass")),
            "degraded": bool(event.get("degraded", False)),
            "degradation_reason": event.get("degradation_reason"),
            "imbalance": float(event.get("imbalance", 0.0)),
            "queue_pressure": float(event.get("queue_pressure", 0.0)),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(OrderBookFeatureEventDB(**row))
            session.commit()

    def save_orderbook_feature_event_batch(self, events: list[Any]) -> None:
        if not events:
            return
        with self.Session() as session:
            for event in events:
                event = self._ensure_dict(event)
                row = {
                    "timestamp": self._coerce_datetime(event["timestamp"]),
                    "symbol": str(event["symbol"]),
                    "quality_flag": str(event.get("quality_flag", "pass")),
                    "degraded": bool(event.get("degraded", False)),
                    "degradation_reason": event.get("degradation_reason"),
                    "imbalance": float(event.get("imbalance", 0.0)),
                    "queue_pressure": float(event.get("queue_pressure", 0.0)),
                    "schema_version": str(event.get("schema_version", "1.0")),
                }
                session.add(OrderBookFeatureEventDB(**row))
            session.commit()

    def save_fastloop_latency_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "stage": str(event.get("stage", "decision_path")),
            "mode": str(event.get("mode", "normal")),
            "event_type": str(event.get("event_type", "FASTLOOP_OK")),
            "reason": str(event.get("reason", "")),
            "sample_count": int(event.get("sample_count", 0)),
            "p50_ms": float(event.get("p50_ms", 0.0)),
            "p95_ms": float(event.get("p95_ms", 0.0)),
            "p99_ms": float(event.get("p99_ms", 0.0)),
            "p999_ms": float(event.get("p999_ms", 0.0)),
            "jitter_ms": float(event.get("jitter_ms", 0.0)),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(FastLoopLatencyEventDB(**row))
            session.commit()

    def save_latency_benchmark_artifact(self, artifact: Any) -> None:
        artifact = self._ensure_dict(artifact)
        row = {
            "run_id": str(artifact["run_id"]),
            "timestamp": self._coerce_datetime(artifact["timestamp"]),
            "passed": bool(artifact.get("passed", False)),
            "reasons_json": self._to_json(artifact.get("reasons", [])),
            "artifact_json": self._to_json(artifact.get("artifact", {})),
            "schema_version": str(artifact.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(LatencyBenchmarkArtifactDB(**row))
            session.commit()

    def save_promotion_gate_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "policy_id": str(event["policy_id"]),
            "from_stage": str(event.get("from_stage", "candidate")),
            "to_stage": str(event.get("to_stage", "candidate")),
            "approved": bool(event.get("approved", False)),
            "reasons_json": self._to_json(event.get("reasons", [])),
            "evidence_json": self._to_json(event.get("evidence", {})),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(PromotionGateEventDB(**row))
            session.commit()

    def save_rollback_drill_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "failed_model_id": str(event["failed_model_id"]),
            "reverted_to": event.get("reverted_to"),
            "executed": bool(event.get("executed", False)),
            "mttr_seconds": float(event.get("mttr_seconds", 0.0)),
            "reasons_json": self._to_json(event.get("reasons", [])),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RollbackDrillEventDB(**row))
            session.commit()

    def save_xai_trade_explanation(self, explanation: Any) -> None:
        explanation = self._ensure_dict(explanation)
        row = {
            "trade_id": str(explanation["trade_id"]),
            "timestamp": self._coerce_datetime(explanation["timestamp"]),
            "symbol": str(explanation["symbol"]),
            "top_feature_contributions_json": self._to_json(explanation.get("top_feature_contributions", [])),
            "agent_contributions_json": self._to_json(explanation.get("agent_contributions", [])),
            "signal_family_contributions_json": self._to_json(explanation.get("signal_family_contributions", [])),
            "metadata_json": self._to_json(explanation.get("metadata", {})),
            "schema_version": str(explanation.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(XAITradeExplanationDB(**row))
            session.commit()

    def save_xai_trade_explanation_batch(self, explanations: list[Any]) -> None:
        if not explanations:
            return
        with self.Session() as session:
            for explanation in explanations:
                explanation = self._ensure_dict(explanation)
                row = {
                    "trade_id": str(explanation["trade_id"]),
                    "timestamp": self._coerce_datetime(explanation["timestamp"]),
                    "symbol": str(explanation["symbol"]),
                    "top_feature_contributions_json": self._to_json(explanation.get("top_feature_contributions", [])),
                    "agent_contributions_json": self._to_json(explanation.get("agent_contributions", [])),
                    "signal_family_contributions_json": self._to_json(explanation.get("signal_family_contributions", [])),
                    "metadata_json": self._to_json(explanation.get("metadata", {})),
                    "schema_version": str(explanation.get("schema_version", "1.0")),
                }
                session.add(XAITradeExplanationDB(**row))
            session.commit()

    def save_pnl_attribution_event(self, event: Any) -> None:
        event = self._ensure_dict(event)
        row = {
            "trade_id": str(event["trade_id"]),
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "symbol": str(event["symbol"]),
            "sector": str(event.get("sector", "unknown")),
            "agent": str(event.get("agent", "unknown")),
            "signal_family": str(event.get("signal_family", "unknown")),
            "realized_pnl": float(event.get("realized_pnl", 0.0)),
            "schema_version": str(event.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(PnLAttributionEventDB(**row))
            session.commit()

    def save_pnl_attribution_event_batch(self, events: list[Any]) -> None:
        if not events:
            return
        with self.Session() as session:
            for event in events:
                event = self._ensure_dict(event)
                row = {
                    "trade_id": str(event["trade_id"]),
                    "timestamp": self._coerce_datetime(event["timestamp"]),
                    "symbol": str(event["symbol"]),
                    "sector": str(event.get("sector", "unknown")),
                    "agent": str(event.get("agent", "unknown")),
                    "signal_family": str(event.get("signal_family", "unknown")),
                    "realized_pnl": float(event.get("realized_pnl", 0.0)),
                    "schema_version": str(event.get("schema_version", "1.0")),
                }
                session.add(PnLAttributionEventDB(**row))
            session.commit()

    def save_operational_metrics_snapshot(self, snapshot: Any) -> None:
        snapshot = self._ensure_dict(snapshot)
        row = {
            "timestamp": self._coerce_datetime(snapshot["timestamp"]),
            "decision_staleness_avg_s": float(snapshot.get("decision_staleness_avg_s", 0.0)),
            "feature_lag_avg_s": float(snapshot.get("feature_lag_avg_s", 0.0)),
            "mode_switch_frequency": int(snapshot.get("mode_switch_frequency", 0)),
            "ood_trigger_rate": int(snapshot.get("ood_trigger_rate", 0)),
            "kill_switch_false_positives": int(snapshot.get("kill_switch_false_positives", 0)),
            "mttr_avg_s": float(snapshot.get("mttr_avg_s", 0.0)),
            "schema_version": str(snapshot.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(OperationalMetricsSnapshotDB(**row))
            session.commit()

    def save_risk_state(self, assessment: Any, *, symbol: str | None = None) -> None:
        assessment = self._ensure_dict(assessment)
        row = {
            "timestamp": self._coerce_datetime(assessment["timestamp"]),
            "symbol": symbol or assessment.get("symbol"),
            "mode": self._enum_value(assessment["mode"]),
            "previous_mode": self._enum_value(assessment.get("previous_mode", assessment["mode"])),
            "approved": bool(assessment.get("approved", True)),
            "veto_reason": assessment.get("veto_reason"),
            "recovery_active": bool(assessment.get("recovery_active", False)),
            "source_service": str(assessment.get("source_service", "independent_risk_overseer")),
            "metadata_json": self._to_json(assessment.get("metadata", {})),
            "schema_version": str(assessment.get("schema_version", "phase4_risk_overseer_v2")),
        }
        with self.Session() as session:
            session.add(RiskStateSnapshotDB(**row))
            session.commit()

    def save_risk_event(self, event: Any, *, symbol: str | None = None) -> None:
        event = self._ensure_dict(event)
        row = {
            "event_id": str(event["event_id"]),
            "timestamp": self._coerce_datetime(event["timestamp"]),
            "symbol": symbol or event.get("symbol"),
            "layer": self._enum_value(event["layer"]),
            "trigger_code": self._enum_value(event["trigger_code"]),
            "mode": self._enum_value(event["mode"]),
            "reason": str(event["reason"]),
            "operator_id": event.get("operator_id"),
            "trigger_value": self._optional_float(event.get("value")),
            "threshold_value": self._optional_float(event.get("threshold")),
            "metadata_json": self._to_json(event.get("metadata", {})),
            "schema_version": str(event.get("schema_version", "phase4_risk_overseer_v2")),
        }
        with self.Session() as session:
            self._upsert(session, RiskEventDB, row, key_fields=("event_id",))
            session.commit()

    def _normalize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        timestamp = self._coerce_datetime(observation["timestamp"])
        symbol = str(observation["symbol"])
        snapshot_id = str(observation["snapshot_id"])
        event_id = str(observation.get("event_id") or self._stable_event_id("obs", symbol, timestamp.isoformat(), snapshot_id))
        return {
            "event_id": event_id,
            "id": self._coerce_or_stable_id(
                observation.get("id"),
                "observation",
                symbol,
                timestamp.isoformat(),
                snapshot_id,
            ),
            "timestamp": timestamp,
            "symbol": symbol,
            "snapshot_id": snapshot_id,
            "technical_direction": str(observation["technical_direction"]),
            "technical_confidence": float(observation.get("technical_confidence", 0.0)),
            "price_forecast": float(observation.get("price_forecast", 0.0)),
            "var_95": float(observation.get("var_95", 0.0)),
            "es_95": float(observation.get("es_95", 0.0)),
            "regime_state": str(observation.get("regime_state", "UNKNOWN")),
            "regime_transition_prob": float(observation.get("regime_transition_prob", 0.0)),
            "sentiment_score": self._optional_float(observation.get("sentiment_score")),
            "sentiment_z_t": self._optional_float(observation.get("sentiment_z_t")),
            "consensus_direction": str(observation.get("consensus_direction", "HOLD")),
            "consensus_confidence": float(observation.get("consensus_confidence", 0.0)),
            "crisis_mode": bool(observation.get("crisis_mode", False)),
            "agent_divergence": bool(observation.get("agent_divergence", False)),
            "orderbook_imbalance": self._optional_float(observation.get("orderbook_imbalance")),
            "queue_pressure": self._optional_float(observation.get("queue_pressure")),
            "current_position": float(observation.get("current_position", 0.0)),
            "unrealized_pnl": float(observation.get("unrealized_pnl", 0.0)),
            "notional_exposure": float(observation.get("notional_exposure", 0.0)),
            "portfolio_features_json": self._to_json(observation.get("portfolio_features"))
            if observation.get("portfolio_features") is not None
            else None,
            "observation_schema_version": str(observation.get("observation_schema_version", "1.0")),
            "quality_status": str(observation.get("quality_status", "pass")),
        }

    def _upsert(self, session, model_cls, values: dict[str, Any], *, key_fields: tuple[str, ...]) -> None:
        dialect_name = session.bind.dialect.name
        if dialect_name == "postgresql":
            stmt = pg_insert(model_cls).values(**values)
            update_map = {field: getattr(stmt.excluded, field) for field in values if field not in key_fields}
            session.execute(stmt.on_conflict_do_update(index_elements=list(key_fields), set_=update_map))
            return

        if dialect_name == "sqlite":
            stmt = sqlite_insert(model_cls).values(**values)
            update_map = {field: getattr(stmt.excluded, field) for field in values if field not in key_fields}
            session.execute(stmt.on_conflict_do_update(index_elements=list(key_fields), set_=update_map))
            return

        existing = (
            session.query(model_cls)
            .filter_by(**{field: values[field] for field in key_fields})
            .one_or_none()
        )
        if existing is None:
            session.add(model_cls(**values))
            return
        for key, value in values.items():
            if key in key_fields:
                continue
            setattr(existing, key, value)

    @staticmethod
    def _stable_event_id(*parts: Any) -> str:
        normalized = "|".join("" if part is None else str(part) for part in parts)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _stable_numeric_id(*parts: Any) -> int:
        normalized = "|".join("" if part is None else str(part) for part in parts)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        # Keep within signed 32-bit integer range for broad DB compatibility.
        return (int(digest[:8], 16) % 2_147_483_647) or 1

    def _coerce_or_stable_id(self, raw_id: Any, *parts: Any) -> int:
        if raw_id is None:
            return self._stable_numeric_id(*parts)
        try:
            return int(raw_id)
        except Exception:
            return self._stable_numeric_id(*parts)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        raise TypeError(f"Unsupported datetime value: {type(value)!r}")

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        return Phase3Recorder._coerce_datetime(value)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _to_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if isinstance(value, dict):
            return value
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
