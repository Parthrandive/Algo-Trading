import json
import pytest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.textual.textual_data_agent import TextualDataAgent, COMPLIANCE_LOG_PATH
from src.agents.textual.adapters import EconomicTimesAdapter, RBIReportsAdapter, RawTextRecord, TextSourceType, SourceRouteDetail
from src.agents.textual.cleaners import TextCleaner
from src.agents.textual.validators import TextualValidator
from src.agents.textual.exporters import TextualExporter
from src.schemas.text_sidecar import ComplianceStatus


def _runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"


def _write_temp_runtime_config(tmp_path: Path, *, updates: dict) -> Path:
    runtime_config = json.loads(_runtime_config_path().read_text(encoding="utf-8"))
    runtime_config.update(updates)
    config_path = tmp_path / "textual_runtime_test.json"
    config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    return config_path

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


def test_rbi_uses_dbie_route_when_rss_is_out(tmp_path: Path):
    runtime_path = _write_temp_runtime_config(
        tmp_path,
        updates={
            "rbi_source_policy": {
                "rss_feed_url": "https://www.rbi.org.in/Scripts/rss.aspx",
                "dbie_download_url": "https://data.rbi.org.in/DBIE/#/",
                "fallback_scraper_url": "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx",
                "simulate_rss_outage": True,
                "simulate_dbie_outage": False,
                "enable_emergency_fallback_scraper": False,
            }
        },
    )
    agent = TextualDataAgent.from_default_components(runtime_path)
    agent.adapters = [adapter for adapter in agent.adapters if isinstance(adapter, RBIReportsAdapter)]

    batch = agent.run_once(as_of_utc=datetime(2026, 3, 7, 10, 0, tzinfo=UTC))
    assert len(batch.canonical_records) == 1
    assert batch.canonical_records[0].source_type.value == "official_api"
    assert str(batch.canonical_records[0].url).startswith("https://data.rbi.org.in/DBIE/#/")
    assert batch.sidecar_records[0].source_route_detail.value == "primary_api"
    assert batch.sidecar_records[0].compliance_status == ComplianceStatus.ALLOW


def test_rbi_emergency_scraper_rejected_when_not_in_outage_mode(tmp_path: Path):
    runtime_path = _write_temp_runtime_config(
        tmp_path,
        updates={
            "outage_emergency_mode": False,
            "rbi_source_policy": {
                "rss_feed_url": "https://www.rbi.org.in/Scripts/rss.aspx",
                "dbie_download_url": "https://data.rbi.org.in/DBIE/#/",
                "fallback_scraper_url": "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx",
                "simulate_rss_outage": True,
                "simulate_dbie_outage": True,
                "enable_emergency_fallback_scraper": True,
            },
        },
    )
    agent = TextualDataAgent.from_default_components(runtime_path)
    # Inject mock HTTP responses so we don't rely on live RBI site
    emergency_page = '<html><body><a href="/Scripts/BS_PressReleaseDisplay.aspx?prid=70123">Emergency RBI release</a></body></html>'
    for adapter in agent.adapters:
        if isinstance(adapter, RBIReportsAdapter):
            adapter._fetch_text = MagicMock(return_value=emergency_page)
    agent.adapters = [adapter for adapter in agent.adapters if isinstance(adapter, RBIReportsAdapter)]

    batch = agent.run_once(as_of_utc=datetime(2026, 3, 7, 10, 0, tzinfo=UTC))
    assert len(batch.canonical_records) == 0
    assert len(batch.sidecar_records) > 0
    sidecar = batch.sidecar_records[0]
    assert sidecar.compliance_status == ComplianceStatus.REJECT
    assert sidecar.compliance_reason == "fallback_requires_emergency"


def test_rbi_emergency_scraper_allowed_in_outage_mode(tmp_path: Path):
    runtime_path = _write_temp_runtime_config(
        tmp_path,
        updates={
            "outage_emergency_mode": True,
            "rbi_source_policy": {
                "rss_feed_url": "https://www.rbi.org.in/Scripts/rss.aspx",
                "dbie_download_url": "https://data.rbi.org.in/DBIE/#/",
                "fallback_scraper_url": "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx",
                "simulate_rss_outage": True,
                "simulate_dbie_outage": True,
                "enable_emergency_fallback_scraper": True,
            },
        },
    )
    agent = TextualDataAgent.from_default_components(runtime_path)
    emergency_page = '<html><body><a href="/Scripts/BS_PressReleaseDisplay.aspx?prid=70123">Emergency RBI release</a></body></html>'
    for adapter in agent.adapters:
        if isinstance(adapter, RBIReportsAdapter):
            adapter._fetch_text = MagicMock(return_value=emergency_page)
    agent.adapters = [adapter for adapter in agent.adapters if isinstance(adapter, RBIReportsAdapter)]

    batch = agent.run_once(as_of_utc=datetime(2026, 3, 7, 10, 0, tzinfo=UTC))
    print("DEBUG records:", batch.canonical_records)
    print("DEBUG sidecars:", batch.sidecar_records)
    assert len(batch.canonical_records) > 0
    assert batch.canonical_records[0].source_type.value == "fallback_scraper"
    sidecar = batch.sidecar_records[0]
    assert sidecar.source_route_detail.value == "fallback_scraper"
    assert sidecar.compliance_status == ComplianceStatus.ALLOW
