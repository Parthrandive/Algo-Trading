import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.textual.adapters import RawTextRecord
from src.agents.textual.textual_data_agent import TextualDataAgent
from src.agents.textual.validators import TextualValidator
from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import ComplianceStatus, SourceRouteDetail, TextSidecarMetadata
from src.utils.schema_registry import SchemaRegistry


def _runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"


def _load_runtime_config() -> dict:
    with _runtime_config_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_news_record(
    *,
    source_id: str = "nse_news_001",
    is_published: bool = True,
    is_embargoed: bool = False,
    license_ok: bool = True,
) -> RawTextRecord:
    payload = {
        "headline": "Nifty closes higher on strong banking momentum",
        "publisher": "NSE",
        "url": f"https://www.nseindia.com/news/{source_id}",
        "language": "en",
        "is_published": is_published,
        "is_embargoed": is_embargoed,
        "license_ok": license_ok,
        "confidence": 0.7,
        "manipulation_risk_score": 0.1,
        "quality_flags": ["pass"],
    }
    return RawTextRecord(
        record_type="news_article",
        source_name="nse_news",
        source_id=source_id,
        timestamp=datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
        content="Nifty closes higher on strong banking momentum",
        payload=payload,
        source_type=TextSourceType.RSS_FEED,
        source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
    )


def _sample_rbi_fallback_record(*, emergency_active: bool) -> RawTextRecord:
    payload = {
        "headline": "RBI emergency scrape sample",
        "publisher": "Reserve Bank of India",
        "url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=12345",
        "language": "en",
        "is_published": True,
        "is_embargoed": False,
        "license_ok": True,
        "confidence": 0.4,
        "manipulation_risk_score": 0.3,
        "quality_flags": ["fallback_scraper", "outage_emergency"],
    }
    if emergency_active:
        payload["fallback_emergency_active"] = True

    return RawTextRecord(
        record_type="news_article",
        source_name="rbi_reports",
        source_id=f"rbi_fallback_{'emergency' if emergency_active else 'blocked'}",
        timestamp=datetime(2026, 3, 6, 9, 30, tzinfo=UTC),
        content="Emergency fallback scrape captured RBI update.",
        payload=payload,
        source_type=TextSourceType.RSS_FEED,
        source_route_detail=SourceRouteDetail.FALLBACK_SCRAPER,
    )


def test_textual_runtime_config_day1_contract_freeze():
    config = _load_runtime_config()
    assert config["version"] == "textual-data-agent-runtime-v1"
    assert config["schema_version"] == "1.0"

    assert set(config["canonical_schema_keys"]) == {
        "news_article",
        "social_post",
        "earnings_transcript",
    }
    for schema_key in config["canonical_schema_keys"].values():
        assert SchemaRegistry.get_model(schema_key) is not None

    assert {entry["source_name"] for entry in config["source_allowlist"]} == {
        "nse_news",
        "economic_times",
        "newsapi_market",
        "rbi_reports",
        "earnings_transcripts",
        "x_posts",
    }
    source_allowed = {entry["source_name"]: bool(entry["allowed"]) for entry in config["source_allowlist"]}
    for source_name in ("nse_news", "economic_times", "newsapi_market", "earnings_transcripts", "x_posts", "rbi_reports"):
        assert source_allowed[source_name] is True
    for entry in config["source_allowlist"]:
        assert entry["allowed_routes"]
        assert entry["compliance_checks"]

    rbi_entry = next(item for item in config["source_allowlist"] if item["source_name"] == "rbi_reports")
    assert rbi_entry["fallback_emergency_only"] is True
    assert rbi_entry["allow_fallback_scraper"] is True
    assert rbi_entry["fallback_emergency_flag_field"] == "fallback_emergency_active"
    assert rbi_entry["preferred_source_sequence"] == [
        "rbi_official_rss_xml",
        "rbi_dbie_official_download",
        "fallback_scraper_emergency_only",
    ]
    assert rbi_entry["official_source_urls"]["rbi_rss_index"] == "https://www.rbi.org.in/Scripts/rss.aspx"
    assert rbi_entry["official_source_urls"]["rbi_dbie_catalog"] == "https://data.rbi.org.in/DBIE/#/"
    
    # Also verify my specific policy keys are present if they were added to the config.
    assert config["outage_emergency_mode"] is False
    assert config["rbi_source_policy"]["rss_feed_url"] == "https://www.rbi.org.in/Scripts/rss.aspx"
    assert config["rbi_source_policy"]["dbie_download_url"] == "https://data.rbi.org.in/DBIE/#/"


def test_textual_runtime_config_has_india_x_templates():
    config = _load_runtime_config()
    templates = config["x_query_templates"]
    assert templates["market_scope"] == "india_equities_and_macro"
    assert len(templates["keyword_templates"]) >= 5
    assert len(templates["semantic_intents"]) >= 3
    assert "india" in " ".join(templates["keyword_templates"]).lower()


def test_text_sidecar_requires_reason_on_reject():
    with pytest.raises(ValidationError):
        TextSidecarMetadata(
            source_type=TextSourceType.RSS_FEED,
            source_id="rejected_item_1",
            ingestion_timestamp_utc=datetime.now(UTC),
            source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
            quality_flags=["compliance_reject"],
            manipulation_risk_score=0.2,
            confidence=0.0,
            ttl_seconds=0,
            compliance_status=ComplianceStatus.REJECT,
            compliance_reason=None,
        )


def test_textual_validator_rejects_embargoed_content():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _sample_news_record(is_embargoed=True)
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is None
    assert sidecar_record.compliance_status == ComplianceStatus.REJECT
    assert sidecar_record.compliance_reason == "embargoed_content"
    assert sidecar_record.ttl_seconds == 0


def test_rbi_fallback_requires_emergency_flag():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _sample_rbi_fallback_record(emergency_active=False)
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is None
    assert sidecar_record.compliance_status == ComplianceStatus.REJECT
    assert sidecar_record.compliance_reason == "fallback_requires_emergency"


def test_rbi_fallback_allowed_in_emergency_mode():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _sample_rbi_fallback_record(emergency_active=True)
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is not None
    assert sidecar_record.compliance_status == ComplianceStatus.ALLOW
    assert sidecar_record.compliance_reason is None


def test_textual_validator_allows_and_validates_canonical_payload():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _sample_news_record()
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is not None
    assert canonical_record.__class__.__name__ == "NewsArticle"
    assert sidecar_record.compliance_status == ComplianceStatus.ALLOW
    assert sidecar_record.compliance_reason is None
    assert sidecar_record.ttl_seconds == 21600
    assert sidecar_record.source_type == TextSourceType.RSS_FEED
    assert sidecar_record.source_route_detail == SourceRouteDetail.OFFICIAL_FEED


def test_textual_agent_run_once_returns_contract_safe_batch():
    agent = TextualDataAgent.from_default_components(_runtime_config_path())
    batch = agent.run_once(as_of_utc=datetime(2026, 3, 2, 10, 0, tzinfo=UTC))

    assert isinstance(batch.canonical_records, list)
    assert isinstance(batch.sidecar_records, list)
    for sidecar in batch.sidecar_records:
        assert isinstance(sidecar.confidence, float)
        assert isinstance(sidecar.ttl_seconds, int)
        assert isinstance(sidecar.manipulation_risk_score, float)

def test_textual_agent_smoke_test_with_default_components():
    """Verifies that the agent runs with all default components and produces the expected mock data."""
    agent = TextualDataAgent.from_default_components(_runtime_config_path())
    
    # Mock RBI adapter HTTP getter to return empty string, forcing the deterministic fallback record
    for adapter in agent.adapters:
        if adapter.source_name == "rbi_reports":
            adapter._has_custom_http_get = True
            adapter._http_get = lambda url, headers=None: ""
        if adapter.source_name == "newsapi_market":
            adapter._api_key = ""

    batch = agent.run_once(as_of_utc=datetime(2026, 3, 2, 10, 0, tzinfo=UTC))

    # We expect 4 or 5 canonical records (NSE [1 or 2], ET-pass, RBI fallback, Transcript, X-pass)
    # The actual count depends on how NSE handles its fallback vs primary.
    assert len(batch.canonical_records) >= 4
    assert len(batch.sidecar_records) >= 6

    # Verify a few types to be sure we have representation
    record_types = [r.model_dump()["source_type"] for r in batch.canonical_records]
    assert record_types.count("rss_feed") >= 3
    assert record_types.count("official_api") >= 1
    assert record_types.count("social_media") == 1
