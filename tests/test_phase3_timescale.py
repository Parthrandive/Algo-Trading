from __future__ import annotations

from sqlalchemy import create_engine

import src.db.phase3_timescale as phase3_timescale


class _FakeResult:
    def __init__(self, value=None):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return _FakeResult(1)


class _FakeBegin:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDialect:
    name = "postgresql"


class _FakeEngine:
    def __init__(self, conn):
        self._conn = conn
        self.dialect = _FakeDialect()

    def begin(self):
        return _FakeBegin(self._conn)


class _FakeInspector:
    def __init__(self, *, incompatible_table: str | None = None):
        self._incompatible_table = incompatible_table

    def get_table_names(self):
        return [spec.table for spec in phase3_timescale.PHASE3_HYPERTABLE_SPECS]

    def get_columns(self, table_name):
        for spec in phase3_timescale.PHASE3_HYPERTABLE_SPECS:
            if spec.table == table_name:
                return [{"name": "id"}, {"name": spec.time_column}]
        return []

    def get_pk_constraint(self, table_name):
        for spec in phase3_timescale.PHASE3_HYPERTABLE_SPECS:
            if spec.table != table_name:
                continue
            if table_name == self._incompatible_table:
                return {"name": f"pk_{table_name}", "constrained_columns": ["id"]}
            return {"name": f"pk_{table_name}", "constrained_columns": ["id", spec.time_column]}
        return {}

    def get_unique_constraints(self, table_name):
        return []

    def get_indexes(self, table_name):
        return []

    def get_foreign_keys(self, table_name):
        return []


def test_apply_phase3_timescale_skips_non_postgres():
    engine = create_engine("sqlite:///:memory:")
    result = phase3_timescale.apply_phase3_timescale_hypertables(engine=engine)

    assert result["status"] == "skipped_non_postgres"
    assert result["table_results"] == {}


def test_find_incompatible_unique_constraints_detects_pk_and_unique_indexes():
    class _Inspector:
        def get_pk_constraint(self, table_name):
            return {"name": "pk_test", "constrained_columns": ["id"]}

        def get_unique_constraints(self, table_name):
            return [{"name": "uq_symbol", "column_names": ["symbol"]}]

        def get_indexes(self, table_name):
            return [{"name": "uidx_symbol", "column_names": ["symbol"], "unique": True}]

        def get_foreign_keys(self, table_name):
            return []

    incompatible = phase3_timescale._find_incompatible_unique_constraints(
        _Inspector(), "trade_decisions", "timestamp"
    )

    assert "pk:pk_test" in incompatible
    assert "uq:uq_symbol" in incompatible
    assert "uidx:uidx_symbol" in incompatible


def test_apply_phase3_timescale_emits_hypertable_ddl(monkeypatch):
    conn = _FakeConn()
    engine = _FakeEngine(conn)

    monkeypatch.setattr(phase3_timescale, "_has_timescaledb_extension", lambda _: True)
    monkeypatch.setattr(phase3_timescale, "inspect", lambda _: _FakeInspector())

    result = phase3_timescale.apply_phase3_timescale_hypertables(engine=engine)

    assert result["status"] == "applied"
    assert all(status == "applied" for status in result["table_results"].values())
    assert any("CREATE EXTENSION IF NOT EXISTS timescaledb" in stmt for stmt in conn.statements)
    assert any("create_hypertable('observations'" in stmt for stmt in conn.statements)
    assert any("create_hypertable('trade_decisions'" in stmt for stmt in conn.statements)
    assert any("create_hypertable('orders'" in stmt for stmt in conn.statements)
    assert any("add_retention_policy('observations'" in stmt for stmt in conn.statements)


def test_apply_phase3_timescale_skips_incompatible_unique_constraints(monkeypatch):
    conn = _FakeConn()
    engine = _FakeEngine(conn)

    monkeypatch.setattr(phase3_timescale, "_has_timescaledb_extension", lambda _: True)
    monkeypatch.setattr(
        phase3_timescale,
        "inspect",
        lambda _: _FakeInspector(incompatible_table="trade_decisions"),
    )

    result = phase3_timescale.apply_phase3_timescale_hypertables(engine=engine)

    assert result["table_results"]["trade_decisions"].startswith("skipped_incompatible_unique_constraints")
    assert not any("create_hypertable('trade_decisions'" in stmt for stmt in conn.statements)
