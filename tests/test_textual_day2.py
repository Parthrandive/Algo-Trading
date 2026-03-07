import os
import pytest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from src.agents.textual.textual_data_agent import TextualDataAgent, COMPLIANCE_LOG_PATH
from src.agents.textual.adapters import EconomicTimesAdapter, RawTextRecord, TextSourceType, SourceRouteDetail
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
    assert record.quality_status.value in {"pass", "warn"}

def test_compliance_rejection_logging(agent):
    class RejectingAdapter:
        source_name = "economic_times"
        source_type = TextSourceType.RSS_FEED

        def fetch(self, *, as_of_utc=None):
            now = as_of_utc or datetime.now(UTC)
            return [
                RawTextRecord(
                    record_type="news_article",
                    source_name="economic_times",
                    source_id="et_pass_1",
                    timestamp=now,
                    content="Markets opened positive with strong breadth.",
                    payload={
                        "headline": "Markets Positive",
                        "publisher": "Economic Times",
                        "url": "https://economictimes.indiatimes.com/markets/pass-1",
                        "language": "en",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": True,
                    },
                    source_type=TextSourceType.RSS_FEED,
                    source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
                ),
                RawTextRecord(
                    record_type="news_article",
                    source_name="economic_times",
                    source_id="et_reject_1",
                    timestamp=now,
                    content="This record should be rejected due to licensing.",
                    payload={
                        "headline": "License Reject Sample",
                        "publisher": "Economic Times",
                        "url": "https://economictimes.indiatimes.com/markets/reject-1",
                        "language": "en",
                        "is_published": True,
                        "is_embargoed": False,
                        "license_ok": False,
                    },
                    source_type=TextSourceType.RSS_FEED,
                    source_route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
                ),
            ]

    agent.adapters = [RejectingAdapter()]
    
    # Clear log before test
    if COMPLIANCE_LOG_PATH.exists():
        COMPLIANCE_LOG_PATH.unlink()
        
    batch = agent.run_once()
    
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
