from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from src.agents.textual.adapters import NSENewsAdapter, RBIReportsAdapter, RawTextRecord
from src.agents.textual.cleaners import TextCleaner
from src.agents.textual.textual_data_agent import TextualDataAgent
from src.agents.textual.validators import TextualValidator
from src.schemas.text_data import SourceType as TextSourceType
from src.schemas.text_sidecar import ComplianceStatus, SourceRouteDetail

IST = ZoneInfo("Asia/Kolkata")


def _runtime_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"


def _workspace_tmp_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "test_tmp" / f"day6_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _http_get_from_map(payload_map: dict[str, str]):
    def _getter(url: str, headers: dict[str, str] | None = None) -> str:
        _ = headers
        if url not in payload_map:
            raise ValueError(f"unexpected url: {url}")
        return payload_map[url]

    return _getter


def _build_news_record(
    *,
    source_id: str,
    content: str,
    timestamp: datetime,
    ingestion_timestamp_utc: datetime,
    route: SourceRouteDetail = SourceRouteDetail.OFFICIAL_FEED,
) -> RawTextRecord:
    return RawTextRecord(
        record_type="news_article",
        source_name="nse_news",
        source_id=source_id,
        timestamp=timestamp,
        content=content,
        payload={
            "headline": "NSE Test Headline",
            "publisher": "NSE",
            "url": f"https://www.nseindia.com/news/{source_id}",
            "language": "en",
            "ingestion_timestamp_utc": ingestion_timestamp_utc,
            "ingestion_timestamp_ist": ingestion_timestamp_utc.astimezone(IST),
            "is_published": True,
            "is_embargoed": False,
            "license_ok": True,
            "quality_status": "pass",
            "confidence": 0.8,
            "manipulation_risk_score": 0.1,
        },
        source_type=TextSourceType.RSS_FEED,
        source_route_detail=route,
    )


def _build_social_record(
    *,
    source_id: str,
    content: str,
    timestamp: datetime,
    ingestion_timestamp_utc: datetime,
) -> RawTextRecord:
    return RawTextRecord(
        record_type="social_post",
        source_name="x_posts",
        source_id=source_id,
        timestamp=timestamp,
        content=content,
        payload={
            "platform": "X",
            "likes": 200,
            "shares": 10,
            "url": f"https://x.com/test/status/{source_id}",
            "language": "en",
            "ingestion_timestamp_utc": ingestion_timestamp_utc,
            "ingestion_timestamp_ist": ingestion_timestamp_utc.astimezone(IST),
            "is_published": True,
            "is_embargoed": False,
            "license_ok": True,
            "quality_status": "pass",
            "confidence": 0.7,
            "manipulation_risk_score": 0.05,
        },
        source_type=TextSourceType.SOCIAL_MEDIA,
        source_route_detail=SourceRouteDetail.PRIMARY_API,
    )


def _build_earnings_record(*, source_id: str, extraction_quality_score: float) -> RawTextRecord:
    now = datetime.now(UTC) - timedelta(minutes=2)
    return RawTextRecord(
        record_type="earnings_transcript",
        source_name="earnings_transcripts",
        source_id=source_id,
        timestamp=now,
        content="Quarterly earnings transcript with stable growth and margin commentary.",
        payload={
            "symbol": "DUMMY.NS",
            "quarter": "Q1",
            "year": 2026,
            "url": f"https://example.com/transcript/{source_id}",
            "language": "en",
            "is_published": True,
            "is_embargoed": False,
            "license_ok": True,
            "quality_status": "pass",
            "confidence": 0.8,
            "manipulation_risk_score": 0.0,
            "extraction_quality_score": extraction_quality_score,
        },
        source_type=TextSourceType.OFFICIAL_API,
        source_route_detail=SourceRouteDetail.OFFICIAL_FEED,
    )


def test_day6_ingestion_provenance_and_timestamp_alignment():
    class DeterministicAdapter:
        source_name = "day6_deterministic"

        def fetch(self, *, as_of_utc=None):
            run_time = as_of_utc or datetime.now(UTC)
            event_time = run_time - timedelta(minutes=15)
            return [
                _build_news_record(
                    source_id="day6_news_1",
                    content="NIFTY closes higher after RBI liquidity guidance.",
                    timestamp=event_time,
                    ingestion_timestamp_utc=run_time,
                ),
                _build_social_record(
                    source_id="day6_social_1",
                    content="India market sentiment on NIFTY stays positive after RBI cues.",
                    timestamp=event_time,
                    ingestion_timestamp_utc=run_time,
                ),
            ]

    as_of = datetime.now(UTC) - timedelta(minutes=5)
    agent = TextualDataAgent.from_default_components(_runtime_config_path())
    agent.adapters = [DeterministicAdapter()]

    batch = agent.run_once(as_of_utc=as_of)

    assert len(batch.canonical_records) == 2
    assert len(batch.sidecar_records) == 2

    sidecar_by_source_id = {sidecar.source_id: sidecar for sidecar in batch.sidecar_records}
    for canonical_record in batch.canonical_records:
        sidecar = sidecar_by_source_id.get(canonical_record.source_id)
        assert sidecar is not None
        assert sidecar.compliance_status == ComplianceStatus.ALLOW
        assert sidecar.source_route_detail in {
            SourceRouteDetail.PRIMARY_API,
            SourceRouteDetail.OFFICIAL_FEED,
            SourceRouteDetail.FALLBACK_SCRAPER,
        }
        assert sidecar.source_type == canonical_record.source_type
        assert sidecar.ingestion_timestamp_utc == canonical_record.ingestion_timestamp_utc
        assert canonical_record.ingestion_timestamp_utc >= canonical_record.timestamp.astimezone(UTC)
        assert (
            canonical_record.ingestion_timestamp_ist.astimezone(UTC) == canonical_record.ingestion_timestamp_utc
        )


def test_day6_leakage_guard_rejects_future_timestamp():
    validator = TextualValidator.from_config_path(_runtime_config_path())
    future_time = datetime.now(UTC) + timedelta(minutes=20)
    record = _build_news_record(
        source_id="day6_future_leakage",
        content="NIFTY guidance that should be rejected due to future timestamp.",
        timestamp=future_time,
        ingestion_timestamp_utc=datetime.now(UTC),
    )
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is None
    assert sidecar_record.compliance_status == ComplianceStatus.REJECT
    assert sidecar_record.compliance_reason == "malformed_timestamp"
    assert "quality_gate_reject" in sidecar_record.quality_flags
    assert "timestamp_in_future" in sidecar_record.quality_flags


@pytest.mark.parametrize(
    ("score", "expected_flag", "expected_confidence"),
    [
        (0.92, "pdf_extraction_pass", 0.92),
        (0.75, "pdf_extraction_warn", 0.75),
        (0.40, "pdf_extraction_low_quality", 0.32),
    ],
)
def test_day6_pdf_quality_thresholds(score: float, expected_flag: str, expected_confidence: float):
    validator = TextualValidator.from_config_path(_runtime_config_path())
    record = _build_earnings_record(source_id=f"day6_pdf_{score}", extraction_quality_score=score)
    payload = TextualDataAgent._build_canonical_payload(record)
    canonical_record, sidecar_record = validator.validate_record(record, payload)

    assert canonical_record is not None
    assert sidecar_record.compliance_status == ComplianceStatus.ALLOW
    assert expected_flag in sidecar_record.quality_flags
    assert sidecar_record.confidence == pytest.approx(expected_confidence)


def test_day6_hinglish_normalization_sanity():
    cleaner = TextCleaner()
    now = datetime(2026, 3, 7, 10, 0, tzinfo=UTC)
    raw_record = RawTextRecord(
        record_type="social_post",
        source_name="x_posts",
        source_id="day6_hinglish_1",
        timestamp=now,
        content="Bhai paisa kya lagta hai, nifty aaj rally karega.",
        payload={
            "platform": "X",
            "likes": 50,
            "shares": 3,
            "url": "https://x.com/test/status/day6_hinglish_1",
            "is_published": True,
            "is_embargoed": False,
            "license_ok": True,
        },
        source_type=TextSourceType.SOCIAL_MEDIA,
        source_route_detail=SourceRouteDetail.PRIMARY_API,
    )

    cleaned = cleaner.clean(raw_record)

    assert cleaned.payload["language"] == "code_mixed"
    assert "code_mixed_detected" in cleaned.payload["quality_flags"]
    normalized = str(cleaned.payload.get("normalized_content", ""))
    assert "brother" in normalized
    assert "money" in normalized
    assert "what" in normalized
    assert "seems" in normalized


def test_day6_failure_mode_rate_limit_activates_nse_fallback_route():
    fallback_rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title><![CDATA[Nifty moves up after policy cues]]></title>
      <description><![CDATA[Fallback route news from ET RSS.]]></description>
      <link>https://example.com/nse-day6-fallback</link>
      <guid>nse-day6-guid</guid>
      <pubDate>Sat, 07 Mar 2026 10:00:00 +0530</pubDate>
    </item>
  </channel>
</rss>
"""
    payload_map = {
        NSENewsAdapter._NSE_ANNOUNCEMENT_URL: '{"status":429,"message":"rate limited"}',
        NSENewsAdapter._ET_FALLBACK_URL: fallback_rss,
    }
    adapter = NSENewsAdapter(
        max_items=5,
        http_get=_http_get_from_map(payload_map),
        cache_root=_workspace_tmp_dir() / "cache",
    )

    records = adapter.fetch(as_of_utc=datetime.now(UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_route_detail == SourceRouteDetail.FALLBACK_SCRAPER
    assert "fallback_scraper" in list(record.payload.get("quality_flags", []))


def test_day6_failure_mode_source_outage_activates_rbi_emergency_fallback():
    emergency_page = """
<html>
  <body>
    <a href="/Scripts/BS_PressReleaseDisplay.aspx?prid=70123">Emergency RBI release</a>
  </body>
</html>
"""
    payload_map = {
        RBIReportsAdapter._RSS_INDEX_URL: "",
        RBIReportsAdapter._DBIE_CATALOG_URL: "",
        RBIReportsAdapter._EMERGENCY_SCRAPER_URL: emergency_page,
    }
    for url in RBIReportsAdapter._FEED_URLS:
        payload_map[url] = ""

    adapter = RBIReportsAdapter(
        max_items=5,
        allow_emergency_fallback=True,
        http_get=_http_get_from_map(payload_map),
        cache_root=_workspace_tmp_dir() / "cache",
    )

    records = adapter.fetch(as_of_utc=datetime.now(UTC))

    assert len(records) == 1
    record = records[0]
    assert record.source_route_detail == SourceRouteDetail.FALLBACK_SCRAPER
    assert bool(record.payload.get("fallback_emergency_active")) is True
    assert "outage_emergency" in list(record.payload.get("quality_flags", []))
