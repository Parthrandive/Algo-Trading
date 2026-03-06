import os
import pytest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from src.agents.textual.textual_data_agent import TextualDataAgent, COMPLIANCE_LOG_PATH
from src.agents.textual.adapters import EconomicTimesAdapter
from src.agents.textual.cleaners import TextCleaner
from src.agents.textual.validators import TextualValidator
from src.agents.textual.exporters import TextualExporter
from src.schemas.text_sidecar import ComplianceStatus

@pytest.fixture
def mock_recorder():
    return MagicMock()

@pytest.fixture
def agent(mock_recorder):
    # Use default components but with mock recorder
    agent = TextualDataAgent.from_default_components(recorder=mock_recorder)
    return agent

def test_run_once_success(agent, mock_recorder):
    # Filter adapters to only one for simplicity in this test
    agent.adapters = [EconomicTimesAdapter()]
    
    batch = agent.run_once()
    
    assert len(batch.canonical_records) > 0
    assert len(batch.sidecar_records) > 0
    
    # Verify persistence call
    mock_recorder.save_text_items.assert_called_once()
    
    # Verify canonical records content
    record = batch.canonical_records[0]
    assert record.publisher == "Economic Times"
    assert record.quality_status.value == "pass"

def test_compliance_rejection_logging(agent):
    # EconomicTimesAdapter returns one record with license_ok=False (should be rejected)
    # as per my implementation in adapters.py
    agent.adapters = [EconomicTimesAdapter()]
    
    # Clear log before test
    if COMPLIANCE_LOG_PATH.exists():
        COMPLIANCE_LOG_PATH.unlink()
        
    batch = agent.run_once()
    
    # Based on adapters.py implementation, EconomicTimesAdapter returns 2 records:
    # 1. Official feed, license_ok=True -> pass
    # 2. Fallback scraper, license_ok=False -> reject
    
    assert len(batch.canonical_records) == 1
    assert len(batch.sidecar_records) == 2
    
    rejected = [s for s in batch.sidecar_records if s.compliance_status == ComplianceStatus.REJECT]
    assert len(rejected) == 1
    assert rejected[0].compliance_reason == "unlicensed_content"
    
    # Check log file
    assert COMPLIANCE_LOG_PATH.exists()
    with open(COMPLIANCE_LOG_PATH, "r") as f:
        log_content = f.read()
        assert "unlicensed_content" in log_content

def test_sidecar_metadata_population(agent):
    agent.adapters = [EconomicTimesAdapter()]
    batch = agent.run_once()
    
    sidecar = batch.sidecar_records[0]
    assert sidecar.source_type.value == "rss_feed"
    assert sidecar.source_route_detail.value == "official_feed"
    assert sidecar.compliance_status == ComplianceStatus.ALLOW
