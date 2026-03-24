import logging

from sqlalchemy import text

from src.db.connection import get_engine
from src.db.models import Base

logger = logging.getLogger(__name__)

# TimescaleDB-only DDL (skipped on plain PostgreSQL)
TIMESCALE_DDL = [
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",
    "SELECT create_hypertable('ohlcv_bars', 'timestamp', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);",
    "SELECT create_hypertable('ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);",
    "SELECT create_hypertable('macro_indicators', 'timestamp', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);",
]

COMPRESSION_DDL = [
    """
    ALTER TABLE IF EXISTS ohlcv_bars SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'symbol,interval',
        timescaledb.compress_orderby = 'timestamp DESC'
    );
    """,
    "SELECT add_compression_policy('ohlcv_bars', INTERVAL '7 days', if_not_exists => TRUE);",
]

# Plain-PostgreSQL-safe DDL (always run)
INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON ohlcv_bars (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_corp_symbol_date ON corporate_actions (symbol, ex_date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_ts ON text_items (timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_symbol ON text_items (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_type ON text_items (item_type, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ingest_ts ON ingestion_log (run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ingest_symbol ON ingestion_log (symbol, run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_live_obs_symbol_ts ON live_market_observations (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_quality_status ON market_data_quality (dataset_type, status, train_ready);",
    "CREATE INDEX IF NOT EXISTS idx_tech_pred_sym_ts ON technical_predictions (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_regime_pred_sym_ts ON regime_predictions (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_sym_ts ON sentiment_scores (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_consensus_sym_ts ON consensus_signals (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_pred_log_agent_ts ON prediction_log (agent, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_backtest_model ON backtest_runs (model_id, run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_observations_symbol_ts ON observations (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_reward_logs_symbol_ts ON reward_logs (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_policies_policy_id ON rl_policies (policy_id, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rl_training_runs_policy_id ON rl_training_runs (policy_id, started_at DESC);",
]

POSTGRES_SCHEMA_REPAIR_DDL = [
    "ALTER TABLE IF EXISTS sentiment_scores ALTER COLUMN lane TYPE VARCHAR(16);",
]


def _execute_statement(conn, statement: str, *, optional: bool) -> None:
    try:
        conn.execute(text(statement))
        logger.debug(f"Executed: {statement[:50]}...")
    except Exception as exc:
        if optional:
            logger.warning(f"Optional DDL skipped: {statement[:50]}...: {exc}")
            return
        raise RuntimeError(f"Required DDL failed: {statement[:50]}...") from exc


def _has_timescaledb(conn) -> bool:
    """Check whether TimescaleDB extension is available on this server."""
    try:
        result = conn.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        )
        return result.fetchone() is not None
    except Exception:
        return False


def init_database(database_url: str | None = None):
    """
    Creates tables via ORM and configures database features.

    On TimescaleDB: creates hypertables, compression, and indexes.
    On plain PostgreSQL: creates regular tables, schema repairs, and indexes.
    """
    engine = get_engine(database_url)

    logger.info("Creating tables from ORM metadata...")
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        if _has_timescaledb(conn):
            logger.info("TimescaleDB detected; applying hypertables and compression...")
            for stmt in TIMESCALE_DDL:
                _execute_statement(conn, stmt, optional=False)
            for stmt in COMPRESSION_DDL:
                _execute_statement(conn, stmt, optional=True)
        else:
            logger.info("Plain PostgreSQL detected; skipping TimescaleDB DDL.")

        if conn.dialect.name == "postgresql":
            logger.info("Applying schema repair DDL...")
            for stmt in POSTGRES_SCHEMA_REPAIR_DDL:
                _execute_statement(conn, stmt, optional=False)

        logger.info("Creating indexes...")
        for stmt in INDEX_DDL:
            _execute_statement(conn, stmt, optional=False)

    logger.info("Schema initialization complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()
