from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.connection import get_engine, get_session
from src.db.models import Base
from src.db.models import (
    BacktestRunDB,
    ConsensusSignalDB,
    ModelCardDB,
    PredictionLogDB,
    RegimePredictionDB,
    SentimentScoreDB,
    TechnicalPredictionDB,
)

logger = logging.getLogger(__name__)

PHASE2_TABLES = (
    TechnicalPredictionDB.__table__,
    RegimePredictionDB.__table__,
    SentimentScoreDB.__table__,
    ConsensusSignalDB.__table__,
    ModelCardDB.__table__,
    BacktestRunDB.__table__,
    PredictionLogDB.__table__,
)

PHASE2_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_tech_pred_sym_ts ON technical_predictions (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_regime_pred_sym_ts ON regime_predictions (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_sym_ts ON sentiment_scores (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_consensus_sym_ts ON consensus_signals (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_pred_log_agent_ts ON prediction_log (agent, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_backtest_model ON backtest_runs (model_id, run_timestamp DESC);",
)

PHASE2_POSTGRES_REPAIR_DDL = (
    "ALTER TABLE IF EXISTS sentiment_scores ALTER COLUMN lane TYPE VARCHAR(16);",
)


class Phase2Recorder:
    """
    Persists Phase 2 agent outputs and queryable metadata.

    The public save methods accept either Pydantic models or plain dictionaries.
    """

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
            self._ensure_phase2_schema()
        self.Session = session_factory or get_session(self.engine)

    def _ensure_phase2_schema(self) -> None:
        """
        Idempotently creates the Phase 2 registry/prediction tables required by
        the analyst-board agents. This protects first-run bootstrap flows where
        the base DB exists but Phase 2 tables were never initialized.
        """
        try:
            Base.metadata.create_all(self.engine, tables=list(PHASE2_TABLES))
            with self.engine.begin() as conn:
                if self.engine.dialect.name == "postgresql":
                    for ddl in PHASE2_POSTGRES_REPAIR_DDL:
                        conn.execute(text(ddl))
                for ddl in PHASE2_INDEX_DDL:
                    conn.execute(text(ddl))
        except Exception as exc:
            raise RuntimeError("Failed to bootstrap Phase 2 recorder schema.") from exc

    def save_technical_prediction(
        self,
        pred: Any,
        *,
        latency_ms: float | None = None,
        data_snapshot_id: str | None = None,
    ) -> None:
        payload = self._coerce_mapping(pred)
        row = {
            "symbol": payload["symbol"],
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "price_forecast": float(payload["price_forecast"]),
            "direction": str(payload["direction"]),
            "volatility_estimate": float(payload["volatility_estimate"]),
            "var_95": float(payload["var_95"]),
            "var_99": float(payload["var_99"]),
            "es_95": float(payload["es_95"]),
            "es_99": float(payload["es_99"]),
            "confidence": float(payload["confidence"]),
            "model_id": str(payload["model_id"]),
            "schema_version": str(payload.get("schema_version", "1.0")),
        }

        with self.Session() as session:
            self._upsert(
                session,
                TechnicalPredictionDB,
                row,
                key_fields=("symbol", "timestamp", "model_id"),
            )
            self._insert_prediction_log(
                session,
                agent="technical",
                payload=payload,
                model_id=row["model_id"],
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                latency_ms=latency_ms,
                data_snapshot_id=data_snapshot_id,
            )
            session.commit()

    def save_regime_prediction(
        self,
        pred: Any,
        *,
        latency_ms: float | None = None,
        data_snapshot_id: str | None = None,
    ) -> None:
        payload = self._coerce_mapping(pred)
        details = payload.get("details")
        row = {
            "symbol": payload["symbol"],
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "regime_state": self._stringify(payload["regime_state"]),
            "transition_probability": float(payload["transition_probability"]),
            "confidence": float(payload["confidence"]),
            "risk_level": self._stringify(payload["risk_level"]),
            "model_id": str(payload["model_id"]),
            "details_json": self._to_json(details) if details is not None else None,
            "schema_version": str(payload.get("schema_version", "1.0")),
        }

        with self.Session() as session:
            self._upsert(
                session,
                RegimePredictionDB,
                row,
                key_fields=("symbol", "timestamp", "model_id"),
            )
            self._insert_prediction_log(
                session,
                agent="regime",
                payload=payload,
                model_id=row["model_id"],
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                latency_ms=latency_ms,
                data_snapshot_id=data_snapshot_id,
            )
            session.commit()

    def save_sentiment_score(
        self,
        score: Any,
        *,
        latency_ms: float | None = None,
        data_snapshot_id: str | None = None,
    ) -> None:
        payload = self._coerce_mapping(score)
        row = {
            "symbol": payload.get("symbol"),
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "lane": str(payload["lane"]),
            "source_id": payload.get("source_id"),
            "source_type": payload.get("source_type"),
            "sentiment_class": str(payload["sentiment_class"]),
            "sentiment_score": float(payload["sentiment_score"]),
            "z_t": self._optional_float(payload.get("z_t")),
            "confidence": float(payload["confidence"]),
            "source_count": int(payload.get("source_count", 0)),
            "ttl_seconds": self._optional_int(payload.get("ttl_seconds")),
            "freshness_flag": payload.get("freshness_flag"),
            "headline_timestamp": self._optional_datetime(payload.get("headline_timestamp")),
            "score_timestamp": self._optional_datetime(payload.get("score_timestamp")),
            "quality_status": payload.get("quality_status"),
            "metadata_json": self._to_json(payload.get("metadata")) if payload.get("metadata") is not None else None,
            "model_id": str(payload["model_id"]),
            "schema_version": str(payload.get("schema_version", "1.0")),
        }

        with self.Session() as session:
            self._upsert(
                session,
                SentimentScoreDB,
                row,
                key_fields=("symbol", "timestamp", "lane", "model_id"),
            )
            self._insert_prediction_log(
                session,
                agent="sentiment",
                payload=payload,
                model_id=row["model_id"],
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                latency_ms=latency_ms,
                data_snapshot_id=data_snapshot_id,
            )
            session.commit()

    def save_consensus_signal(
        self,
        signal: Any,
        *,
        latency_ms: float | None = None,
        data_snapshot_id: str | None = None,
    ) -> None:
        payload = self._coerce_mapping(signal)
        row = {
            "symbol": payload["symbol"],
            "timestamp": self._coerce_datetime(payload["timestamp"]),
            "final_direction": str(payload["final_direction"]),
            "final_confidence": float(payload["final_confidence"]),
            "technical_weight": float(payload["technical_weight"]),
            "regime_weight": float(payload["regime_weight"]),
            "sentiment_weight": float(payload["sentiment_weight"]),
            "crisis_mode": bool(payload.get("crisis_mode", False)),
            "agent_divergence": bool(payload.get("agent_divergence", False)),
            "transition_model": str(payload["transition_model"]),
            "model_id": str(payload["model_id"]),
            "schema_version": str(payload.get("schema_version", "1.0")),
        }

        with self.Session() as session:
            self._upsert(
                session,
                ConsensusSignalDB,
                row,
                key_fields=("symbol", "timestamp", "model_id"),
            )
            self._insert_prediction_log(
                session,
                agent="consensus",
                payload=payload,
                model_id=row["model_id"],
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                latency_ms=latency_ms,
                data_snapshot_id=data_snapshot_id,
            )
            session.commit()

    def save_model_card(self, card: dict[str, Any], model_id: str | None = None, agent: str | None = None) -> None:
        payload = self._coerce_mapping(card)
        now = datetime.now(timezone.utc)
        model_identifier = str(model_id or payload["model_id"])
        agent_name = str(agent or payload["agent"])
        performance = payload.get("performance_json", payload.get("performance"))
        row = {
            "model_id": model_identifier,
            "agent": agent_name,
            "model_family": str(payload.get("model_family", payload.get("family", "unknown"))),
            "version": str(payload.get("version", "1.0")),
            "created_at": self._coerce_datetime(payload.get("created_at", now)),
            "updated_at": self._coerce_datetime(payload.get("updated_at", now)),
            "metadata_json": self._to_json(payload),
            "performance_json": self._to_json(performance) if performance is not None else None,
            "status": str(payload.get("status", "active")),
        }

        with self.Session() as session:
            self._upsert(
                session,
                ModelCardDB,
                row,
                key_fields=("model_id",),
            )
            session.commit()

    def save_backtest_run(self, run: dict[str, Any]) -> None:
        payload = self._coerce_mapping(run)
        row = {
            "model_id": str(payload["model_id"]),
            "run_timestamp": self._coerce_datetime(payload["run_timestamp"]),
            "backtest_start": self._coerce_datetime(payload["backtest_start"]),
            "backtest_end": self._coerce_datetime(payload["backtest_end"]),
            "sharpe": self._optional_float(payload.get("sharpe")),
            "sortino": self._optional_float(payload.get("sortino")),
            "max_drawdown": self._optional_float(payload.get("max_drawdown")),
            "win_rate": self._optional_float(payload.get("win_rate")),
            "coverage": self._optional_float(payload.get("coverage")),
            "params_json": self._to_json(payload.get("params_json", payload.get("params"))) if payload.get("params_json", payload.get("params")) is not None else None,
            "notes": payload.get("notes"),
        }

        with self.Session() as session:
            session.add(BacktestRunDB(**row))
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

    def _insert_prediction_log(
        self,
        session,
        *,
        agent: str,
        payload: dict[str, Any],
        model_id: str,
        symbol: str | None,
        timestamp: datetime,
        latency_ms: float | None,
        data_snapshot_id: str | None,
    ) -> None:
        session.add(
            PredictionLogDB(
                agent=agent,
                symbol=symbol,
                timestamp=timestamp,
                prediction_json=self._to_json(payload),
                model_id=model_id,
                latency_ms=latency_ms,
                data_snapshot_id=data_snapshot_id,
            )
        )

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

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

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

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

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
        if isinstance(value, Enum):
            return value.value
        return value
