from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text

from src.db.connection import get_engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HypertableSpec:
    table: str
    time_column: str
    chunk_interval: str
    retention_interval: str | None = None


PHASE3_HYPERTABLE_SPECS: tuple[HypertableSpec, ...] = (
    HypertableSpec("observations", "timestamp", "1 week", "12 months"),
    HypertableSpec("trade_decisions", "timestamp", "1 week", None),
    HypertableSpec("orders", "created_at", "1 month", None),
    HypertableSpec("order_fills", "fill_timestamp", "1 month", None),
    HypertableSpec("portfolio_snapshots", "timestamp", "1 week", "6 months"),
    HypertableSpec("reward_logs", "timestamp", "1 week", "12 months"),
    HypertableSpec("deliberation_logs", "timestamp", "1 month", "6 months"),
)

PHASE3_PK_REFACTOR_TABLES: dict[str, str] = {
    "reward_logs": "event_id",
    "portfolio_snapshots": "event_id",
    "deliberation_logs": "event_id",
}


def apply_phase3_timescale_hypertables(
    *,
    database_url: str | None = None,
    engine=None,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Converts selected Phase 3 tables into Timescale hypertables.

    Notes:
    - PostgreSQL only; returns a skipped status for other dialects.
    - Requires TimescaleDB extension visibility on the target server.
    - Skips tables that have unique constraints incompatible with
      Timescale partitioning rules (time column must be part of every
      unique/PK constraint).
    """
    db_engine = engine or get_engine(database_url)
    if db_engine.dialect.name != "postgresql":
        return {
            "status": "skipped_non_postgres",
            "table_results": {},
            "errors": [],
        }

    table_results: dict[str, str] = {}
    errors: list[str] = []

    with db_engine.begin() as conn:
        if not _has_timescaledb_extension(conn):
            message = "timescaledb extension is not available on this PostgreSQL server."
            if strict:
                raise RuntimeError(message)
            return {
                "status": "skipped_no_timescaledb",
                "table_results": {},
                "errors": [message],
            }

        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        for spec in PHASE3_HYPERTABLE_SPECS:
            try:
                table_scope = conn.begin_nested() if hasattr(conn, "begin_nested") else nullcontext()
                with table_scope:
                    inspector = inspect(conn)
                    table_names = set(inspector.get_table_names())
                    if spec.table not in table_names:
                        table_results[spec.table] = "skipped_missing_table"
                        continue

                    if spec.table in PHASE3_PK_REFACTOR_TABLES:
                        _refactor_primary_key_for_timescale(conn, inspector, spec.table, spec.time_column)
                        inspector = inspect(conn)
                        table_names = set(inspector.get_table_names())

                    columns = {col["name"] for col in inspector.get_columns(spec.table)}
                    if spec.time_column not in columns:
                        table_results[spec.table] = "skipped_missing_time_column"
                        continue

                    incompatible = _find_incompatible_unique_constraints(inspector, spec.table, spec.time_column)
                    if incompatible:
                        detail = ",".join(incompatible)
                        table_results[spec.table] = f"skipped_incompatible_unique_constraints:{detail}"
                        continue

                    conn.execute(
                        text(
                            "SELECT create_hypertable("
                            f"'{spec.table}', '{spec.time_column}', "
                            f"chunk_time_interval => INTERVAL '{spec.chunk_interval}', "
                            "if_not_exists => TRUE, migrate_data => TRUE);"
                        )
                    )
                    if spec.retention_interval:
                        conn.execute(
                            text(
                                "SELECT add_retention_policy("
                                f"'{spec.table}', INTERVAL '{spec.retention_interval}', "
                                "if_not_exists => TRUE);"
                            )
                        )
                    table_results[spec.table] = "applied"
            except Exception as exc:  # pragma: no cover - connector/db dependent
                message = f"{spec.table}: {exc}"
                if strict:
                    raise RuntimeError(f"Failed Timescale migration for {spec.table}.") from exc
                errors.append(message)
                table_results[spec.table] = "error"
                logger.warning("Phase 3 Timescale migration skipped for %s: %s", spec.table, exc)

    applied_count = sum(1 for value in table_results.values() if value == "applied")
    total = len(PHASE3_HYPERTABLE_SPECS)
    if applied_count == total and not errors:
        status = "applied"
    elif applied_count == 0:
        status = "skipped" if not errors else "partial"
    else:
        status = "partial"
    return {
        "status": status,
        "table_results": table_results,
        "errors": errors,
    }


def format_phase3_timescale_result(result: dict[str, Any]) -> str:
    """Pretty JSON output for CLI and operator audit logs."""
    return json.dumps(result, indent=2, sort_keys=True)


def _has_timescaledb_extension(conn) -> bool:
    try:
        value = conn.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb' LIMIT 1;")
        ).scalar()
        return value is not None
    except Exception:
        return False


def _find_incompatible_unique_constraints(inspector, table_name: str, time_column: str) -> list[str]:
    """
    Returns unique/PK constraints that do not include the time column.
    Timescale requires all unique constraints to include partition columns.
    """
    incompatible: list[str] = []

    pk = inspector.get_pk_constraint(table_name) or {}
    pk_cols = pk.get("constrained_columns") or []
    if pk_cols and time_column not in pk_cols:
        incompatible.append(f"pk:{pk.get('name') or 'primary_key'}")

    for constraint in inspector.get_unique_constraints(table_name) or []:
        cols = constraint.get("column_names") or []
        if cols and time_column not in cols:
            incompatible.append(f"uq:{constraint.get('name') or '_'.join(cols)}")

    for index in inspector.get_indexes(table_name) or []:
        if not index.get("unique"):
            continue
        cols = index.get("column_names") or []
        if cols and time_column not in cols:
            incompatible.append(f"uidx:{index.get('name') or '_'.join(cols)}")

    return incompatible


def _refactor_primary_key_for_timescale(conn, inspector, table_name: str, time_column: str) -> None:
    """
    Refactors legacy single-column PKs into (event_id, time_column) PKs
    for append-only event tables so Timescale hypertable conversion can proceed.
    """
    event_id_column = PHASE3_PK_REFACTOR_TABLES[table_name]
    table_identifier = _quote_identifier(table_name)
    time_identifier = _quote_identifier(time_column)
    event_identifier = _quote_identifier(event_id_column)

    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if event_id_column not in columns:
        conn.execute(text(f"ALTER TABLE {table_identifier} ADD COLUMN {event_identifier} VARCHAR(64);"))

    conn.execute(
        text(
            f"UPDATE {table_identifier} "
            f"SET {event_identifier} = md5(random()::text || clock_timestamp()::text || ctid::text) "
            f"WHERE {event_identifier} IS NULL;"
        )
    )
    conn.execute(text(f"ALTER TABLE {table_identifier} ALTER COLUMN {event_identifier} SET NOT NULL;"))

    pk = inspector.get_pk_constraint(table_name) or {}
    pk_name = pk.get("name")
    pk_cols = pk.get("constrained_columns") or []
    target_pk = [event_id_column, time_column]
    if pk_cols == target_pk:
        return

    if pk_name:
        pk_identifier = _quote_identifier(pk_name)
        conn.execute(text(f"ALTER TABLE {table_identifier} DROP CONSTRAINT IF EXISTS {pk_identifier};"))

    conn.execute(text(f"ALTER TABLE {table_identifier} ADD PRIMARY KEY ({event_identifier}, {time_identifier});"))


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
