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

def test_bulletin_parser_html_extracts_dates(bulletin_parser):
    raw = {
        "html": "<html><body><h1>RBI Bulletin</h1><p>Published on 15 February 2026</p></body></html>"
    }
    records = bulletin_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.RBI_BULLETIN
    assert records[0].value == 1.0

def test_bulletin_parser_pdf_text_extracts_dates(bulletin_parser):
    raw = {
        "pdf_text": "RBI Bulletin monthly release date: 2026-02-15."
    }
    records = bulletin_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.RBI_BULLETIN

def test_parser_malformed_data(cpi_parser):
    raw = {
        "data": [
            {"date": "invalid-date", "value": "not-a-float"}
        ]
    }
    records = cpi_parser.parse(raw)
    assert len(records) == 0

from src.agents.macro.parsers import FIIDIIParser, FXReservesParser, BondSpreadParser

@pytest.fixture
def fiidii_parser():
    return FIIDIIParser()

@pytest.fixture
def fx_parser():
    return FXReservesParser()

@pytest.fixture
def bond_spread_parser():
    return BondSpreadParser()

def test_fiidii_parser_success(fiidii_parser):
    raw = {
        "date": "2026-02-26",
        "fii_flow": -1500.5,
        "dii_flow": 2000.0
    }
    records = fiidii_parser.parse(raw)
    assert len(records) == 2
    
    fii_record = next(r for r in records if r.indicator_name == MacroIndicatorType.FII_FLOW)
    assert fii_record.value == -1500.5
    assert fii_record.unit == "INR_Cr"
    assert fii_record.period == "Daily"
    
    dii_record = next(r for r in records if r.indicator_name == MacroIndicatorType.DII_FLOW)
    assert dii_record.value == 2000.0
    assert dii_record.unit == "INR_Cr"
    assert dii_record.period == "Daily"

def test_fx_parser_success(fx_parser):
    raw = {
        "date": "2026-02-20",
        "value": 680.5
    }
    records = fx_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.FX_RESERVES
    assert records[0].value == 680.5
    assert records[0].unit == "USD_Bn"
    assert records[0].period == "Weekly"

def test_bond_spread_parser_success(bond_spread_parser):
    raw = {
        "date": "2026-02-26",
        "india_10y_percent": 7.10,
        "us_10y_percent": 4.25
    }
    records = bond_spread_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.INDIA_US_10Y_SPREAD
    # (7.10 - 4.25) * 100 = 2.85 * 100 = 285.0
    assert records[0].value == 285.0
    assert records[0].unit == "bps"
    assert records[0].period == "Daily"

def test_fiidii_parser_partial_data(fiidii_parser):
    raw = {
        "date": "2026-02-26",
        "fii_flow": -1500.5
        # dii_flow missing
    }
    records = fiidii_parser.parse(raw)
    assert len(records) == 1
    assert records[0].indicator_name == MacroIndicatorType.FII_FLOW
    assert records[0].quality_status == QualityFlag.WARN

def test_fiidii_parser_missing_date(fiidii_parser):
    raw = {
        "fii_flow": -1500.5,
        "dii_flow": 2000.0
    }
    records = fiidii_parser.parse(raw)
    assert len(records) == 0

def test_bond_spread_parser_missing_legs(bond_spread_parser):
    raw = {
        "date": "2026-02-26",
        "india_10y_percent": 7.10
        # us_10y missing
    }
    records = bond_spread_parser.parse(raw)
    assert len(records) == 0

def test_fiidii_parser_outlier_warn(fiidii_parser):
    raw = {
        "date": "2026-02-26",
        "fii_flow": 250000.0,
        "dii_flow": 2000.0,
    }
    records = fiidii_parser.parse(raw)
    fii_record = next(r for r in records if r.indicator_name == MacroIndicatorType.FII_FLOW)
    assert fii_record.quality_status == QualityFlag.WARN

def test_fx_parser_outlier_warn(fx_parser):
    raw = {
        "date": "2026-02-20",
        "value": 1800.0,
    }
    records = fx_parser.parse(raw)
    assert len(records) == 1
    assert records[0].quality_status == QualityFlag.WARN

def test_bond_spread_parser_outlier_warn(bond_spread_parser):
    raw = {
        "date": "2026-02-26",
        "india_10y_percent": 40.0,
        "us_10y_percent": 1.0,
    }
    records = bond_spread_parser.parse(raw)
    assert len(records) == 1
    assert records[0].quality_status == QualityFlag.WARN
