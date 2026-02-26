import pytest
import shutil
from pathlib import Path
from datetime import UTC, datetime, timedelta

from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.pipeline import MacroIngestPipeline
from src.agents.macro.parsers import BaseParser
from src.agents.macro.recorder import MacroSilverRecorder
from src.schemas.macro_data import MacroIndicator, MacroIndicatorType, SourceType, QualityFlag
from src.agents.sentinel.failover_client import DegradationState

# ------------------------------------------------------------------------
# Test Mocks
# ------------------------------------------------------------------------

class MockClient(MacroClientInterface):
    """Mocks Bronze layer fetch responding with raw JSON."""
    
    def __init__(self, raw_data_to_return, should_fail=False):
        self._data = raw_data_to_return
        self.should_fail = should_fail
        
    @property
    def supported_indicators(self):
        return frozenset([MacroIndicatorType.CPI])
        
    def get_indicator(self, name, date_range):
        if self.should_fail:
            raise ConnectionError("Simulated network outage")
        return [self._data]


class MockParser(BaseParser):
    """Simple parser to unpack the mock raw JSON."""
    indicator_type = MacroIndicatorType.CPI
    unit = "%"
    period = "Monthly"
    source_type = SourceType.OFFICIAL_API
    
    def parse(self, raw_data):
        now = datetime.now(UTC)
        obs_date = datetime.now(UTC) - timedelta(days=5)
        
        # Simulated parsing error
        if "bad_key" in raw_data:
            raise ValueError("Malformed simulated payload")
            
        return [MacroIndicator(
            indicator_name=self.indicator_type,
            value=float(raw_data["val"]),
            unit=self.unit,
            period=self.period,
            timestamp=obs_date,
            source_type=self.source_type,
            ingestion_timestamp_utc=now,
            ingestion_timestamp_ist=now,
            schema_version="1.1",
            quality_status=QualityFlag.PASS,
        )]

# ------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------

@pytest.fixture
def tmp_silver_dir(tmp_path):
    yield tmp_path / "silver"

@pytest.fixture
def tmp_quarantine_dir(tmp_path):
    yield tmp_path / "quarantine"
    
@pytest.fixture
def recorder(tmp_silver_dir, tmp_quarantine_dir):
    return MacroSilverRecorder(
        base_dir=str(tmp_silver_dir),
        quarantine_dir=str(tmp_quarantine_dir)
    )

@pytest.fixture
def pipeline(recorder):
    return MacroIngestPipeline(recorder=recorder)

# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------

def test_pipeline_success(pipeline, recorder):
    """Bronze raw -> Bronze dict -> Silver parse -> Silver Parquet save."""
    client = MockClient({"val": 3.14})
    parser = MockParser()
    dr = DateRange(start=(datetime.now(UTC) - timedelta(days=30)).date(), end=datetime.now(UTC).date())
    
    assert pipeline.degradation_state == DegradationState.NORMAL
    
    records = pipeline.run_ingest(client, MacroIndicatorType.CPI, dr, parser)
    
    assert len(records) == 1
    assert records[0].value == 3.14
    assert pipeline.degradation_state == DegradationState.NORMAL

def test_pipeline_feed_network_failure(pipeline):
    """Client fetching failure triggers REDUCE_ONLY state."""
    client = MockClient({}, should_fail=True)
    parser = MockParser()
    dr = DateRange(start=(datetime.now(UTC) - timedelta(days=30)).date(), end=datetime.now(UTC).date())
    
    with pytest.raises(ConnectionError):
        pipeline.run_ingest(client, MacroIndicatorType.CPI, dr, parser)
        
    assert pipeline.degradation_state == DegradationState.REDUCE_ONLY

def test_pipeline_parse_failure(pipeline):
    """Parsing malformed payloads raises exception and triggers REDUCE_ONLY."""
    client = MockClient({"bad_key": True}) # simulate bad payload
    parser = MockParser()
    dr = DateRange(start=(datetime.now(UTC) - timedelta(days=30)).date(), end=datetime.now(UTC).date())
    
    with pytest.raises(ValueError):
        pipeline.run_ingest(client, MacroIndicatorType.CPI, dr, parser)
        
    assert pipeline.degradation_state == DegradationState.REDUCE_ONLY

def test_pipeline_recovery(pipeline):
    """Pipeline transitions REDUCE_ONLY -> NORMAL on next success."""
    # Step 1: Force failure
    client_fail = MockClient({}, should_fail=True)
    parser = MockParser()
    dr = DateRange(start=(datetime.now(UTC) - timedelta(days=30)).date(), end=datetime.now(UTC).date())
    
    with pytest.raises(ConnectionError):
        pipeline.run_ingest(client_fail, MacroIndicatorType.CPI, dr, parser)
        
    assert pipeline.degradation_state == DegradationState.REDUCE_ONLY
    
    # Step 2: Push success
    client_success = MockClient({"val": 99.9})
    records = pipeline.run_ingest(client_success, MacroIndicatorType.CPI, dr, parser)
    
    assert len(records) == 1
    assert pipeline.degradation_state == DegradationState.NORMAL
