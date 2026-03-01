import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import insert

from src.agents.macro.freshness import MacroFreshnessChecker, WebhookAlerter
from src.db.connection import get_engine, get_session
from src.db.models import Base, MacroIndicatorDB

@pytest.fixture
def test_db_url(tmp_path):
    import os
    db_path = tmp_path / "test_macro_freshness.db"
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
        "week3_required_publish_set": ["CPI", "WPI", "IIP"],
        "indicator_configs": {
            "CPI": {"freshness_window_hours": 48},
            "WPI": {"freshness_window_hours": 48},
            "IIP": {"freshness_window_hours": 48}
        }
    }
    config_path = tmp_path / "test_freshness_config.json"
    config_path.write_text(json.dumps(config_data))
    return str(config_path)

def test_freshness_all_missing(config_file, setup_db, caplog):
    checker = MacroFreshnessChecker(config_file, database_url=setup_db)
    
    with caplog.at_level(logging.WARNING):
        report = checker.generate_report()
        
    assert report["total_required"] == 3
    assert report["missing"] == 3
    assert report["healthy"] == 0
    assert report["completion_percentage"] == 0.0
    
    assert "Missing Data: CPI" in caplog.text
    assert "SLA Breach: Completeness < 95%" in caplog.text

def test_freshness_mixed_state(config_file, setup_db, caplog):
    checker = MacroFreshnessChecker(config_file, database_url=setup_db)
    
    now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)
    
    # CPI is fresh (10 hours old)
    # WPI is stale (50 hours old, SLA is 48)
    # IIP is missing
    
    engine = get_engine(setup_db)
    Session = get_session(engine)
    with Session() as session:
        session.execute(
            insert(MacroIndicatorDB).values([
                {
                    "indicator_name": "CPI",
                    "timestamp": now - timedelta(hours=12),
                    "value": 5.0,
                    "unit": "%",
                    "period": "Monthly",
                    "source_type": "official_api",
                    "ingestion_timestamp_utc": now - timedelta(hours=10),
                    "ingestion_timestamp_ist": now - timedelta(hours=10),
                    "schema_version": "1.1",
                    "quality_status": "pass"
                },
                {
                    "indicator_name": "WPI",
                    "timestamp": now - timedelta(hours=60),
                    "value": 2.0,
                    "unit": "%",
                    "period": "Monthly",
                    "source_type": "official_api",
                    "ingestion_timestamp_utc": now - timedelta(hours=50),
                    "ingestion_timestamp_ist": now - timedelta(hours=50),
                    "schema_version": "1.1",
                    "quality_status": "pass"
                }
            ])
        )
        session.commit()
        
    with caplog.at_level(logging.WARNING):
        report = checker.generate_report(reference_time=now)
        
    assert report["missing"] == 1
    assert report["stale"] == 1
    assert report["healthy"] == 1
    assert report["completion_percentage"] == round((1 / 3) * 100, 1)
    
    assert report["details"]["CPI"]["status"] == "FRESH"
    assert report["details"]["WPI"]["status"] == "STALE"
    assert report["details"]["IIP"]["status"] == "MISSING"
    
    assert "Stale Data: WPI" in caplog.text

def test_freshness_all_healthy(config_file, setup_db):
    checker = MacroFreshnessChecker(config_file, database_url=setup_db)
    now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)
    
    engine = get_engine(setup_db)
    Session = get_session(engine)
    with Session() as session:
        records = []
        for ind in ["CPI", "WPI", "IIP"]:
            records.append({
                "indicator_name": ind,
                "timestamp": now - timedelta(hours=2),
                "value": 5.0,
                "unit": "%",
                "period": "Monthly",
                "source_type": "official_api",
                "ingestion_timestamp_utc": now - timedelta(hours=1),
                "ingestion_timestamp_ist": now - timedelta(hours=1),
                "schema_version": "1.1",
                "quality_status": "pass"
            })
        session.execute(insert(MacroIndicatorDB).values(records))
        session.commit()
        
    report = checker.generate_report(reference_time=now)
    assert report["missing"] == 0
    assert report["stale"] == 0
    assert report["healthy"] == 3
    assert report["completion_percentage"] == 100.0
