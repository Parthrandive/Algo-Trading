import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence
from unittest.mock import Mock, patch

import pytest

from sqlalchemy import insert

from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.parsers import BaseParser
from src.agents.macro.pipeline import MacroIngestPipeline
from src.agents.macro.scheduler import MacroScheduler
from src.db.connection import get_engine, get_session
from src.db.models import Base, IngestionLog, MacroIndicatorDB
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType

@pytest.fixture
def mock_pipeline():
    pipeline = Mock(spec=MacroIngestPipeline)
    pipeline.run_ingest.return_value = []
    return pipeline

@pytest.fixture
def test_db_url(tmp_path):
    import os
    db_path = tmp_path / "test_macro.db"
    return f"sqlite:///{db_path}"

@pytest.fixture
def setup_db(test_db_url):
    engine = get_engine(test_db_url)
    Base.metadata.create_all(engine)
    yield test_db_url
    Base.metadata.drop_all(engine)

@pytest.fixture
def config_file(tmp_path):
    config_data = {
        "version": "macro-monitor-runtime-v1",
        "default_request_policy": {
            "retry": {"max_attempts": 3, "base_backoff_seconds": 0.1}
        },
        "indicator_configs": {
            "CPI": {
                "sources": [
                    {"retry": {"max_attempts": 2, "base_backoff_seconds": 0.1}}
                ]
            }
        }
    }
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config_data))
    return str(config_path)

def test_scheduler_reads_config(config_file, mock_pipeline, setup_db):
    client = Mock(spec=MacroClientInterface)
    parser = Mock(spec=BaseParser)
    registry = {MacroIndicatorType.CPI: (client, parser)}
    
    scheduler = MacroScheduler(config_file, mock_pipeline, registry, database_url=setup_db)
    
    # Check retry policy extraction
    policy = scheduler.get_retry_policy(MacroIndicatorType.CPI)
    assert policy["max_attempts"] == 2
    assert policy["base_backoff_seconds"] == 0.1
    
    # Indicator not in config uses default
    policy_default = scheduler.get_retry_policy(MacroIndicatorType.WPI)
    assert policy_default["max_attempts"] == 3

@patch("time.sleep")
def test_exponential_backoff_retry(mock_sleep, config_file, mock_pipeline, setup_db):
    client = Mock(spec=MacroClientInterface)
    parser = Mock(spec=BaseParser)
    registry = {MacroIndicatorType.CPI: (client, parser)}
    
    # Mock pipeline to fail once, then succeed
    mock_pipeline.run_ingest.side_effect = [RuntimeError("Fetch failed"), [Mock(spec=MacroIndicator)]]
    
    scheduler = MacroScheduler(config_file, mock_pipeline, registry, database_url=setup_db)
    date_range = DateRange(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 31, tzinfo=UTC))
    
    records = scheduler.run_job(MacroIndicatorType.CPI, date_range)
    assert len(records) == 1
    assert mock_pipeline.run_ingest.call_count == 2
    
def test_idempotent_ingest_guard(config_file, mock_pipeline, setup_db):
    client = Mock(spec=MacroClientInterface)
    parser = Mock(spec=BaseParser)
    registry = {MacroIndicatorType.CPI: (client, parser)}
    
    scheduler = MacroScheduler(config_file, mock_pipeline, registry, database_url=setup_db)
    
    date_range = DateRange(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 31, tzinfo=UTC))
    
    # Insert a dummy record into DB to simulate existing ingestion
    engine = get_engine(setup_db)
    Session = get_session(engine)
    with Session() as session:
        session.execute(
            insert(MacroIndicatorDB).values(
                indicator_name="CPI",
                timestamp=datetime(2026, 1, 15, tzinfo=UTC),
                value=5.0,
                unit="%",
                period="Monthly",
                source_type="official_api",
                ingestion_timestamp_utc=datetime.now(UTC),
                ingestion_timestamp_ist=datetime.now(UTC),
            )
        )
        session.commit()
        
    records = scheduler.run_job(MacroIndicatorType.CPI, date_range)
    # Pipeline should not be called because of idempotent guard
    mock_pipeline.run_ingest.assert_not_called()
    assert len(records) == 0

def test_provenance_logging(config_file, mock_pipeline, setup_db):
    client = Mock(spec=MacroClientInterface)
    parser = Mock(spec=BaseParser)
    registry = {MacroIndicatorType.CPI: (client, parser)}
    
    mock_pipeline.run_ingest.return_value = [Mock(spec=MacroIndicator), Mock(spec=MacroIndicator)]
    
    scheduler = MacroScheduler(config_file, mock_pipeline, registry, database_url=setup_db)
    date_range = DateRange(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 31, tzinfo=UTC))
    
    scheduler.run_job(MacroIndicatorType.CPI, date_range)
    
    # Check IngestionLog
    engine = get_engine(setup_db)
    Session = get_session(engine)
    with Session() as session:
        logs = session.query(IngestionLog).all()
        assert len(logs) == 1
        assert logs[0].symbol == "CPI"
        assert logs[0].status == "success"
        assert logs[0].records_ingested == 2
        assert "macro_v1.1" in logs[0].code_hash
