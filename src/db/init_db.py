import logging
from sqlalchemy import text
from src.db.connection import get_engine
from src.db.models import Base

logger = logging.getLogger(__name__)

REQUIRED_DDL_STATEMENTS = [
    # 1. Extension
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",

    # 2. Hypertables
    "SELECT create_hypertable('ohlcv_bars', 'timestamp', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);",
    "SELECT create_hypertable('ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);",
    "SELECT create_hypertable('macro_indicators', 'timestamp', chunk_time_interval => INTERVAL '1 year', if_not_exists => TRUE);",

    # 3. Indexes
    "CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON ohlcv_bars (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_corp_symbol_date ON corporate_actions (symbol, ex_date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_ts ON text_items (timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_symbol ON text_items (symbol, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_text_type ON text_items (item_type, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ingest_ts ON ingestion_log (run_timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ingest_symbol ON ingestion_log (symbol, run_timestamp DESC);",
]

OPTIONAL_DDL_STATEMENTS = [
    # 4. Compression (Requires TimescaleDB community/enterprise, might fail on basic so wrap in try)
    """
    ALTER TABLE IF EXISTS ohlcv_bars SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'symbol,interval',
        timescaledb.compress_orderby = 'timestamp DESC'
    );
    """,
    "SELECT add_compression_policy('ohlcv_bars', INTERVAL '7 days', if_not_exists => TRUE);"
]

def _execute_statement(conn, statement: str, *, optional: bool) -> None:
    try:
        with conn.begin():
            conn.execute(text(statement))
        logger.debug(f"Executed: {statement[:50]}...")
    except Exception as exc:
        if optional:
            logger.warning(f"Optional DDL failed: {statement[:50]}...: {exc}")
            return
        raise RuntimeError(f"Required DDL failed: {statement[:50]}...") from exc

def init_database(database_url: str | None = None):
    """
    Creates tables via ORM and configures TimescaleDB specific features.
    """
    engine = get_engine(database_url)
    
    logger.info("Creating tables from ORM metadata...")
    Base.metadata.create_all(engine)
    
    logger.info("Executing TimescaleDB specific DDL...")
    with engine.connect() as conn:
        for statement in REQUIRED_DDL_STATEMENTS:
            _execute_statement(conn, statement, optional=False)
        for statement in OPTIONAL_DDL_STATEMENTS:
            _execute_statement(conn, statement, optional=True)
            
    logger.info("Schema initialization complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()
