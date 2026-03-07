import pytest
from datetime import UTC, datetime

from src.agents.textual.textual_data_agent import TextualDataAgent
from src.agents.textual.adapters import XPostAdapter, RawTextRecord, TextSourceType, SourceRouteDetail
from src.schemas.text_sidecar import ComplianceStatus

def _runtime_config_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "textual_data_agent_runtime_v1.json"

@pytest.fixture
def agent():
    return TextualDataAgent.from_default_components(_runtime_config_path())

def test_x_post_relevance_and_negative_filters(agent):
    # Only use X adapter for this test
    adapter = XPostAdapter()
    agent.adapters = [adapter]
    
    batch = agent.run_once()
    
    # XPostAdapter returns 3 records:
    # 1. India market (allow)
    # 2. Crypto giveaway (reject: negative_filter_match)
    # 3. US market (reject: low_india_relevance)
    
    assert len(batch.canonical_records) == 1
    assert batch.canonical_records[0].source_id == "x_post_india_market_1"
    
    rejections = [s for s in batch.sidecar_records if s.compliance_status == ComplianceStatus.REJECT]
    assert len(rejections) == 2
    
    reasons = [r.compliance_reason for r in rejections]
    assert "negative_filter_match" in reasons
    assert "low_india_relevance" in reasons

def test_reliability_confidence_weighting(agent):
    # Test confidence adjustments for different source types
    validator = agent.validator
    
    def news_record(source_name, route):
        return RawTextRecord(
            record_type="news_article",
            source_name=source_name,
            source_id="test_news",
            timestamp=datetime.now(UTC),
            content="India market news mentioning NIFTY.",
            payload={
                "url": "http://test.com", 
                "timestamp": datetime.now(UTC),
                "headline": "Test Headline",
                "publisher": "Test Publisher",
                "license_ok": True,
            },
            source_type=TextSourceType.RSS_FEED,
            source_route_detail=route
        )

    def social_record(source_name, route, likes=100):
        return RawTextRecord(
            record_type="social_post",
            source_name=source_name,
            source_id="test_social",
            timestamp=datetime.now(UTC),
            content="India market social news mentioning NIFTY.",
            payload={
                "url": "http://test.com", 
                "timestamp": datetime.now(UTC),
                "likes": likes,
                "platform": "X",
                "license_ok": True,
            },
            source_type=TextSourceType.SOCIAL_MEDIA,
            source_route_detail=route
        )

    # RSS Feed (official) -> Boost (+0.2)
    rec_rss = news_record("nse_news", SourceRouteDetail.OFFICIAL_FEED)
    _, sidecar_rss = validator.validate_record(rec_rss, agent._build_canonical_payload(rec_rss))
    # Base 0.5 + 0.2 = 0.7
    assert sidecar_rss.confidence == 0.7

    # X Post (high likes) -> Social Penalty (-0.1) + High Likes Boost (+0.1) = 0.5
    rec_x = social_record("x_posts", SourceRouteDetail.PRIMARY_API, likes=2000)
    _, sidecar_x = validator.validate_record(rec_x, agent._build_canonical_payload(rec_x))
    assert sidecar_x.confidence == 0.5

    # Fallback Scraper News -> Boost (+0.2 RSS) - Penalty (-0.3 Fallback) = 0.4
    rec_fb = news_record("economic_times", SourceRouteDetail.FALLBACK_SCRAPER)
    _, sidecar_fb = validator.validate_record(rec_fb, agent._build_canonical_payload(rec_fb))
    assert sidecar_fb.confidence == pytest.approx(0.4)

def test_burst_detection_logic(agent):
    # Setup 5 allowed X posts to trigger burst detection
    class HighVolumeXAdapter:
        def fetch(self, *, as_of_utc=None):
            return [
                RawTextRecord(
                    record_type="social_post",
                    source_name="x_posts",
                    source_id=f"burst_{i}",
                    timestamp=datetime.now(UTC),
                    content=f"NIFTY is going up! #India #Market signal_{i}",
                    payload={
                        "likes": 100, 
                        "url": "http://x.com", 
                        "is_published": True, 
                        "license_ok": True,
                        "platform": "X",
                        "timestamp": datetime.now(UTC),
                    },
                    source_type=TextSourceType.SOCIAL_MEDIA,
                    source_route_detail=SourceRouteDetail.PRIMARY_API
                ) for i in range(5)
            ]

    agent.adapters = [HighVolumeXAdapter()]
    batch = agent.run_once()
    
    assert len(batch.canonical_records) == 5
    # The first 4 should be normal (before burst threshold is met in processing? No, burst is batch level)
    for sidecar in batch.sidecar_records:
        assert "high_volume_burst" in sidecar.quality_flags
        # Confidence: 0.5 - 0.1 (social) - 0.1 (low likes < 1000? No, 100 is not < 10 but not > 1000 either)
        # 0.5 - 0.1 = 0.4.
        # Manipulation Risk: 0.0 (base) + 0.2 (burst) = 0.2.
        assert sidecar.manipulation_risk_score == pytest.approx(0.2)
