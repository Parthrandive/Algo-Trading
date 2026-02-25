"""
Unit tests for macro parsers (CPI, WPI, IIP, RBIBulletin).
"""

from datetime import UTC, datetime, timedelta
import pytest
from src.agents.macro.parsers import (
    CPIParser,
    WPIParser,
    IIPParser,
    RBIBulletinParser,
)
from src.schemas.macro_data import (
    MacroIndicatorType,
    QualityFlag,
)

@pytest.fixture
def cpi_parser():
    return CPIParser()

@pytest.fixture
def wpi_parser():
    return WPIParser()

@pytest.fixture
def iip_parser():
    return IIPParser()

@pytest.fixture
def bulletin_parser():
    return RBIBulletinParser()

def test_cpi_parser_success(cpi_parser):
    # Use a very recent date to pass latency check
    recent_date = (datetime.now(UTC) - timedelta(hours=24)).date().isoformat()
    raw = {
        "data": [
            {"date": recent_date, "value": 5.09}
        ]
    }
    records = cpi_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.CPI
    assert records[0].value == 5.09
    assert records[0].unit == "%"
    assert records[0].quality_status == QualityFlag.PASS

def test_cpi_parser_outlier(cpi_parser):
    raw = {
        "data": [
            {"date": "2026-01-01", "value": 25.0} # Outlier (max 15%)
        ]
    }
    records = cpi_parser.parse(raw)
    assert records[0].quality_status == QualityFlag.WARN

def test_cpi_parser_latency(cpi_parser):
    # Obs date is 3 days ago, SLA is 48h
    three_days_ago = (datetime.now(UTC) - timedelta(days=3)).date().isoformat()
    raw = {
        "data": [
            {"date": three_days_ago, "value": 5.0}
        ]
    }
    records = cpi_parser.parse(raw)
    assert records[0].quality_status == QualityFlag.WARN

def test_wpi_parser_success(wpi_parser):
    raw = {
        "data": [
            {"date": "2026-01-01", "value": 2.37}
        ]
    }
    records = wpi_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.WPI
    assert records[0].value == 2.37

def test_iip_parser_success(iip_parser):
    raw = {
        "data": [
            {"date": "2026-01-01", "value": 3.8}
        ]
    }
    records = iip_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.IIP
    assert records[0].value == 3.8

def test_bulletin_parser_success(bulletin_parser):
    raw = {
        "publications": [
            {"date": "2026-02-15", "title": "RBI Bulletin Feb"}
        ]
    }
    records = bulletin_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.RBI_BULLETIN
    assert records[0].value == 1.0
    assert records[0].unit == "count"
    assert records[0].period == "Irregular"

def test_parser_malformed_data(cpi_parser):
    raw = {
        "data": [
            {"date": "invalid-date", "value": "not-a-float"}
        ]
    }
    records = cpi_parser.parse(raw)
    assert len(records) == 0
