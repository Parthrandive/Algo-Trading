from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.agents.textual.adapters import RawTextRecord, TextSourceType, SourceRouteDetail
from src.agents.textual.textual_data_agent import (
    SIDECAR_ARTIFACT_PATH,
    TextualDataAgent,
)
from src.agents.textual.validators import TextualValidator
from src.schemas.text_sidecar import ComplianceStatus


def _runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"


def _build_news_record(*, source_id: str, content: str, timestamp: datetime) -> RawTextRecord:
    return RawTextRecord(
        record_type="news_article",
        source_name="nse_news",
        source_id=source_id,
        timestamp=timestamp,
        content=content,
        payload={
            "headline": "Test Headline",
            "publisher": "NSE",
            "url": f"https://www.nseindia.com/news/{source_id}",
            "language": "en",
            "is_published": True,
            "is_embargoed": False,
            "license_ok": True,
            "quality_status": "pass",
            "confidence": 0.8,
            "manipulation_risk_score": 0.1,
        },
        source_type=TextSourceType.RSS_FEED,
        source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
    )


def test_quality_gate_rejects_missing_required_fields():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _build_news_record(
        source_id="",
        content="",
        timestamp=datetime(2026, 3, 7, 9, 0, tzinfo=UTC),
    )
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical, sidecar = validator.validate_record(record, payload)

    assert canonical is None
    assert sidecar.compliance_status == ComplianceStatus.REJECT
    assert sidecar.compliance_reason == "missing_required_fields"
    assert "missing_field:source_id" in sidecar.quality_flags
    assert "missing_field:content" in sidecar.quality_flags


def test_quality_gate_flags_stale_timestamp():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    stale_time = datetime.now(UTC) - timedelta(hours=36)
    record = _build_news_record(
        source_id="stale_news",
        content="NIFTY update with enough content length for quality checks.",
        timestamp=stale_time,
    )
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical, sidecar = validator.validate_record(record, payload)

    assert canonical is not None
    assert sidecar.compliance_status == ComplianceStatus.ALLOW
    assert "stale_timestamp" in sidecar.quality_flags


def test_agent_deduplicates_and_persists_sidecar_artifact():
    if SIDECAR_ARTIFACT_PATH.exists():
        SIDECAR_ARTIFACT_PATH.unlink()

    class DuplicateAdapter:
        source_name = "nse_news"

        def fetch(self, *, as_of_utc=None):
            now = as_of_utc or datetime.now(UTC)
            return [
                _build_news_record(
                    source_id="dup_a",
                    content="NIFTY index closes higher amid banking strength and broad gains.",
                    timestamp=now,
                ),
                _build_news_record(
                    source_id="dup_b",
                    content="NIFTY index closes higher amid banking strength and broad gains.",
                    timestamp=now,
                ),
            ]

    agent = TextualDataAgent.from_default_components(_runtime_config_path())
    agent.adapters = [DuplicateAdapter()]
    batch = agent.run_once(as_of_utc=datetime.now(UTC) - timedelta(minutes=5))

    assert len(batch.canonical_records) == 1
    assert len(batch.sidecar_records) == 2
    duplicate_sidecar = [s for s in batch.sidecar_records if s.compliance_reason == "duplicate_record"]
    assert len(duplicate_sidecar) == 1

    assert SIDECAR_ARTIFACT_PATH.exists()
    artifact = json.loads(SIDECAR_ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact["total_records"] == 2
