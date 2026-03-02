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
        "rbi_reports",
        "earnings_transcripts",
        "x_posts",
    }
    for entry in config["source_allowlist"]:
        assert entry["allowed"] is True
        assert entry["allowed_routes"]
        assert entry["compliance_checks"]


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


def test_textual_agent_skeleton_runs_with_empty_adapter_stubs():
    agent = TextualDataAgent.from_default_components(_runtime_config_path())
    batch = agent.run_once(as_of_utc=datetime(2026, 3, 2, 10, 0, tzinfo=UTC))

    assert batch.canonical_records == []
    assert batch.sidecar_records == []
