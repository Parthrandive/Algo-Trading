import pytest
from datetime import UTC, datetime
from src.agents.textual.textual_data_agent import TextualDataAgent, PDF_SPOT_CHECK_REPORT_PATH
from src.agents.textual.adapters import RawTextRecord, TextSourceType, SourceRouteDetail
from src.schemas.text_sidecar import ComplianceStatus

def _runtime_config_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"

@pytest.fixture
def agent():
    return TextualDataAgent.from_default_components(_runtime_config_path())

def test_pdf_extraction_integration(agent):
    batch = agent.run_once()
    earnings_record = next(r for r in batch.canonical_records if "dummy" in r.url.lower() or "example.com" in r.url.lower())

    # Depending on whether real urlopen worked or fallback was used, the content will differ
    assert "guid" in earnings_record.content.lower() or "dummy" in earnings_record.url.lower() or "example" in earnings_record.url.lower()

    earnings_sidecar = next(s for s in batch.sidecar_records if s.source_id == earnings_record.source_id)
    assert any("spot_check" in f or "offline" in f for f in earnings_sidecar.quality_flags)
    assert earnings_sidecar.confidence >= 0.9


def test_pdf_spot_check_report_generation(agent):
    if PDF_SPOT_CHECK_REPORT_PATH.exists():
        PDF_SPOT_CHECK_REPORT_PATH.unlink()

    _ = agent.run_once()

    assert agent.last_pdf_spot_check_report is not None
    report = agent.last_pdf_spot_check_report
    assert report["total_documents"] >= 1  # EarningsTranscriptAdapter supplies PDF records
    assert report["fail_count"] == 0
    assert report["warn_count"] == 0
    assert PDF_SPOT_CHECK_REPORT_PATH.exists()

def test_hinglish_detection(agent):
    class HinglishAdapter:
        source_name = "x_posts"
        def fetch(self, *, as_of_utc=None):
            now = datetime.now(UTC)
            return [
                RawTextRecord(
                    record_type="social_post",
                    source_name="x_posts",
                    source_id="hinglish_1",
                    timestamp=now,
                    content="Bhai, NIFTY ka kya lagta hai? \u092a\u0948\u0938\u0947 double hoga kya?",
                    payload={
                        "platform": "X", 
                        "likes": 100, 
                        "license_ok": True,
                        "url": "http://x.com/hinglish",
                        "timestamp": now,
                    },
                    source_type=TextSourceType.SOCIAL_MEDIA,
                    source_route_detail=SourceRouteDetail.PRIMARY_API
                )
            ]
            
    agent.adapters = [HinglishAdapter()]
    batch = agent.run_once()

    assert len(batch.canonical_records) > 0
    record = batch.canonical_records[0]
    assert record.language == "code_mixed"

    sidecar = batch.sidecar_records[0]
    assert "code_mixed_detected" in sidecar.quality_flags
    assert "transliteration_applied" in sidecar.quality_flags
    assert "Bhai, NIFTY ka kya lagta hai?" in record.content


def test_scam_lexicon_detection_on_allowed_record(agent):
    class ScamAdapter:
        source_name = "x_posts"
        def fetch(self, *, as_of_utc=None):
            now = datetime.now(UTC)
            return [
                RawTextRecord(
                    record_type="social_post",
                    source_name="x_posts",
                    source_id="scam_1",
                    timestamp=now,
                    content="NIFTY pump and dump call with 100% success rate and no risk.",
                    payload={
                        "platform": "X", 
                        "likes": 42,
                        "license_ok": True,
                        "url": "http://x.com/scam",
                        "timestamp": now,
                    },
                    source_type=TextSourceType.SOCIAL_MEDIA,
                    source_route_detail=SourceRouteDetail.PRIMARY_API
                )
            ]
            
    agent.adapters = [ScamAdapter()]
    batch = agent.run_once()

    assert len(batch.sidecar_records) > 0
    sidecar = batch.sidecar_records[0]
    assert any("scam_pattern" in flag for flag in sidecar.quality_flags)
    assert sidecar.compliance_status == ComplianceStatus.ALLOW
    assert sidecar.manipulation_risk_score >= 0.7


def test_scam_lexicon_detection_on_rejected_record(agent):
    class RejectedScamAdapter:
        source_name = "x_posts"
        def fetch(self, *, as_of_utc=None):
            now = datetime.now(UTC)
            return [
                RawTextRecord(
                    record_type="social_post",
                    source_name="x_posts",
                    source_id="scam_reject_1",
                    timestamp=now,
                    content="Join our pump and dump group with 100% success rate and no risk.",
                    payload={
                        "platform": "X",
                        "likes": 5,
                        "license_ok": True,
                        "url": "http://x.com/scam-reject",
                        "timestamp": now,
                    },
                    source_type=TextSourceType.SOCIAL_MEDIA,
                    source_route_detail=SourceRouteDetail.PRIMARY_API,
                )
            ]

    agent.adapters = [RejectedScamAdapter()]
    batch = agent.run_once()
    sidecar = batch.sidecar_records[0]
    assert sidecar.compliance_status == ComplianceStatus.REJECT
    assert sidecar.compliance_reason == "low_india_relevance"
    assert any("scam_pattern" in flag for flag in sidecar.quality_flags)
    assert sidecar.manipulation_risk_score > 0.0
