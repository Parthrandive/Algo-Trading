from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.connection import get_engine, get_session
from src.db.models import ObservationDB, RLPolicyDB, RLTrainingRunDB, RewardLogDB, TradeDecisionDB


class Phase3Recorder:
    """Persist Phase 3 strategic artifacts and operational records."""

    def __init__(self, database_url: str | None = None, engine=None, session_factory=None):
        self.engine = engine or get_engine(database_url)
        self.Session = session_factory or get_session(self.engine)

    def save_observation(self, observation: dict[str, Any]) -> None:
        row = self._normalize_observation(observation)
        with self.Session() as session:
            self._upsert(session, ObservationDB, row, key_fields=("symbol", "timestamp", "snapshot_id"))
            session.commit()

    def save_observation_batch(self, observations: list[dict[str, Any]]) -> None:
        if not observations:
            return
        with self.Session() as session:
            for observation in observations:
                row = self._normalize_observation(observation)
                self._upsert(session, ObservationDB, row, key_fields=("symbol", "timestamp", "snapshot_id"))
            session.commit()

    def save_rl_policy(self, policy: dict[str, Any]) -> None:
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

    def save_rl_training_run(self, run: dict[str, Any]) -> None:
        row = {
            "policy_id": str(run["policy_id"]),
            "run_timestamp": self._coerce_datetime(run["run_timestamp"]),
            "training_start": self._coerce_datetime(run["training_start"]),
            "training_end": self._coerce_datetime(run["training_end"]),
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

    def save_trade_decision(self, decision: dict[str, Any]) -> None:
        row = {
            "timestamp": self._coerce_datetime(decision["timestamp"]),
            "symbol": str(decision["symbol"]),
            "observation_id": decision.get("observation_id"),
            "policy_id": str(decision["policy_id"]),
            "action": str(decision["action"]),
            "action_size": float(decision["action_size"]),
            "confidence": float(decision.get("confidence", 0.0)),
            "entropy": self._optional_float(decision.get("entropy")),
            "sac_weight": self._optional_float(decision.get("sac_weight")),
            "ppo_weight": self._optional_float(decision.get("ppo_weight")),
            "td3_weight": self._optional_float(decision.get("td3_weight")),
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

    def save_trade_decision_batch(self, decisions: list[dict[str, Any]]) -> None:
        if not decisions:
            return
        with self.Session() as session:
            for decision in decisions:
                row = {
                    "timestamp": self._coerce_datetime(decision["timestamp"]),
                    "symbol": str(decision["symbol"]),
                    "observation_id": decision.get("observation_id"),
                    "policy_id": str(decision["policy_id"]),
                    "action": str(decision["action"]),
                    "action_size": float(decision["action_size"]),
                    "confidence": float(decision.get("confidence", 0.0)),
                    "entropy": self._optional_float(decision.get("entropy")),
                    "sac_weight": self._optional_float(decision.get("sac_weight")),
                    "ppo_weight": self._optional_float(decision.get("ppo_weight")),
                    "td3_weight": self._optional_float(decision.get("td3_weight")),
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

    def save_reward_log(self, reward: dict[str, Any]) -> None:
        row = {
            "decision_id": reward.get("decision_id"),
            "timestamp": self._coerce_datetime(reward["timestamp"]),
            "symbol": str(reward["symbol"]),
            "policy_id": str(reward["policy_id"]),
            "reward_name": str(reward["reward_name"]),
            "reward_value": float(reward["reward_value"]),
            "components_json": self._to_json(reward.get("components")) if reward.get("components") is not None else None,
            "schema_version": str(reward.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RewardLogDB(**row))
            session.commit()

    def save_reward_log_batch(self, rewards: list[dict[str, Any]]) -> None:
        if not rewards:
            return
        with self.Session() as session:
            for reward in rewards:
                row = {
                    "decision_id": reward.get("decision_id"),
                    "timestamp": self._coerce_datetime(reward["timestamp"]),
                    "symbol": str(reward["symbol"]),
                    "policy_id": str(reward["policy_id"]),
                    "reward_name": str(reward["reward_name"]),
                    "reward_value": float(reward["reward_value"]),
                    "components_json": self._to_json(reward.get("components"))
                    if reward.get("components") is not None
                    else None,
                    "schema_version": str(reward.get("schema_version", "1.0")),
                }
                session.add(RewardLogDB(**row))
            session.commit()

    def _normalize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": self._coerce_datetime(observation["timestamp"]),
            "symbol": str(observation["symbol"]),
            "snapshot_id": str(observation["snapshot_id"]),
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
