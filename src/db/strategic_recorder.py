from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.connection import get_engine, get_session
from src.db.models import Base
from src.db.models import RLPolicyDB, RLTrainingRunDB, RewardLogDB, StrategicObservationDB

STRATEGIC_TABLES = (
    StrategicObservationDB.__table__,
    RewardLogDB.__table__,
    RLPolicyDB.__table__,
    RLTrainingRunDB.__table__,
)

STRATEGIC_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_observations_symbol_ts ON observations (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_reward_logs_symbol_ts ON reward_logs (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_policies_policy_id ON rl_policies (policy_id, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_training_runs_policy_id ON rl_training_runs (policy_id, started_at DESC);",
)


class StrategicRecorder:
    """Persists Week 1 RL foundation artifacts without executing training."""

    def __init__(
        self,
        database_url: str | None = None,
        engine=None,
        session_factory=None,
        *,
        bootstrap_schema: bool = True,
    ) -> None:
        self.engine = engine or get_engine(database_url)
        self._bootstrap_schema = bool(bootstrap_schema)
        if self._bootstrap_schema:
            self._ensure_schema()
        self.Session = session_factory or get_session(self.engine)

    def _ensure_schema(self) -> None:
        Base.metadata.create_all(self.engine, tables=list(STRATEGIC_TABLES))
        with self.engine.begin() as conn:
            for ddl in STRATEGIC_INDEX_DDL:
                conn.execute(text(ddl))

    def save_observation(self, observation: Any) -> None:
        payload = self._coerce_mapping(observation)
        row = {
            "symbol": str(payload["symbol"]),
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "schema_version": str(payload.get("schema_version", "1.0")),
            "observation_vector_json": self._to_json(payload["observation_vector"]),
            "mapping_version": str(payload["mapping_version"]),
            "feature_names_json": self._to_json(payload["feature_names"]),
            "technical_model_id": payload.get("technical_model_id"),
            "regime_model_id": payload.get("regime_model_id"),
            "sentiment_model_id": payload.get("sentiment_model_id"),
            "consensus_model_id": payload.get("consensus_model_id"),
            "alignment_tolerance_seconds": float(payload.get("alignment_tolerance_seconds", 300.0)),
            "source_timestamp_json": self._to_json(payload.get("source_timestamps", {})),
            "metadata_json": self._to_json(payload.get("metadata", {})),
        }
        with self.Session() as session:
            self._upsert(
                session,
                StrategicObservationDB,
                row,
                key_fields=("symbol", "timestamp", "schema_version"),
            )
            session.commit()

    def save_reward_log(self, reward_log: Any) -> None:
        payload = self._coerce_mapping(reward_log)
        row = {
            "symbol": str(payload["symbol"]),
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "episode_id": str(payload["episode_id"]),
            "reward_name": str(payload["reward_name"]),
            "reward_value": float(payload["reward_value"]),
            "portfolio_value": self._optional_float(payload.get("portfolio_value")),
            "gross_return": self._optional_float(payload.get("gross_return")),
            "net_return": self._optional_float(payload.get("net_return")),
            "transaction_cost": self._optional_float(payload.get("transaction_cost")),
            "slippage_cost": self._optional_float(payload.get("slippage_cost")),
            "action": self._optional_float(payload.get("action")),
            "position_before": self._optional_float(payload.get("position_before")),
            "position_after": self._optional_float(payload.get("position_after")),
            "metadata_json": self._to_json(payload.get("metadata", {})),
            "schema_version": str(payload.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RewardLogDB(**row))
            session.commit()

    def save_rl_policy(self, policy: Any) -> None:
        payload = self._coerce_mapping(policy)
        now = datetime.now(timezone.utc)
        row = {
            "policy_id": str(payload["policy_id"]),
            "algorithm": str(payload["algorithm"]),
            "version": str(payload.get("version", "1.0")),
            "stage": str(payload.get("stage", "foundation")),
            "training_status": str(payload.get("training_status", "not_started")),
            "observation_schema_version": str(payload.get("observation_schema_version", "1.0")),
            "action_space": str(payload.get("action_space", "continuous")),
            "checkpoint_path": payload.get("checkpoint_path"),
            "checkpoint_status": str(payload.get("checkpoint_status", "not_available")),
            "is_teacher_policy": bool(payload.get("is_teacher_policy", True)),
            "offline_only": bool(payload.get("offline_only", True)),
            "notes": payload.get("notes"),
            "metadata_json": self._to_json(payload.get("metadata", {})),
            "created_at": self._coerce_datetime(payload.get("created_at", now)),
            "updated_at": self._coerce_datetime(payload.get("updated_at", now)),
        }
        with self.Session() as session:
            self._upsert(session, RLPolicyDB, row, key_fields=("policy_id",))
            session.commit()

    def save_training_run(self, run: Any) -> None:
        payload = self._coerce_mapping(run)
        row = {
            "policy_id": str(payload["policy_id"]),
            "started_at": self._coerce_datetime(payload["started_at"]),
            "completed_at": self._optional_datetime(payload.get("completed_at")),
            "status": str(payload.get("status", "planned")),
            "split_label": str(payload.get("split_label", "walk_forward")),
            "train_start": self._optional_datetime(payload.get("train_start")),
            "train_end": self._optional_datetime(payload.get("train_end")),
            "validation_start": self._optional_datetime(payload.get("validation_start")),
            "validation_end": self._optional_datetime(payload.get("validation_end")),
            "test_start": self._optional_datetime(payload.get("test_start")),
            "test_end": self._optional_datetime(payload.get("test_end")),
            "reward_name": payload.get("reward_name"),
            "metrics_json": self._to_json(payload.get("metrics", {})),
            "params_json": self._to_json(payload.get("params", {})),
            "checkpoint_path": payload.get("checkpoint_path"),
            "notes": payload.get("notes"),
            "schema_version": str(payload.get("schema_version", "1.0")),
        }
        with self.Session() as session:
            session.add(RLTrainingRunDB(**row))
            session.commit()

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

        filters = [getattr(model_cls, field) == values[field] for field in key_fields]
        existing = session.execute(select(model_cls).where(*filters)).scalar_one_or_none()
        if existing is None:
            session.add(model_cls(**values))
            return
        for field, value in values.items():
            if field in key_fields:
                continue
            setattr(existing, field, value)

    @staticmethod
    def _coerce_mapping(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return dict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Expected dict-like or Pydantic object, got {type(obj)!r}")

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _optional_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        return self._coerce_datetime(value)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    def _to_json(self, payload: Any) -> str:
        return json.dumps(self._normalize_json(payload), separators=(",", ":"))

    def _normalize_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_json(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_json(item) for item in value]
        if isinstance(value, datetime):
            value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if isinstance(value, pd.Timestamp):
            return self._coerce_datetime(value).isoformat()
        return value
