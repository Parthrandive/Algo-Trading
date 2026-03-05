import pytest
from datetime import UTC, datetime
from src.agents.textual.textual_data_agent import TextualDataAgent
from src.agents.textual.adapters import RawTextRecord, TextSourceType, SourceRouteDetail

def _runtime_config_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"

@pytest.fixture
def agent():
    return TextualDataAgent.from_default_components(_runtime_config_path())

def test_pdf_extraction_integration(agent):
    # RBIReportsAdapter uses PDFExtractor
    batch = agent.run_once()
    rbi_record = next(r for r in batch.canonical_records if r.source_id == "rbi_report_feb_2026")
    
    # Check if text was extracted by our mock service
    assert "RBI Bulletin" in rbi_record.content
    
    # Check if extraction quality score exists in sidecar
    rbi_sidecar = next(s for s in batch.sidecar_records if s.source_id == "rbi_report_feb_2026")
    # Our mock gives 0.95 for this content length
    assert rbi_sidecar.confidence <= 0.95 

def test_hinglish_detection(agent):
    # Create a mock adapter for Hinglish
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
                    content="Bhai, NIFTY ka kya lagta hai? Profit hoga kya?",
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
    # Content should be cleaned
    assert "Bhai, NIFTY ka kya lagta hai?" in record.content

def test_scam_lexicon_detection(agent):
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
                    content="Join our pump and dump group for 100% success rate! No risk.",
                    payload={
                        "platform": "X", 
                        "likes": 10, 
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
    # pump and dump, 100% success rate, no risk -> 3 patterns * 0.25 = 0.75
    assert sidecar.manipulation_risk_score >= 0.75
