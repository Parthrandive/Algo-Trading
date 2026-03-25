from datetime import UTC, datetime

from src.agents.sentiment.schemas import SentimentLane, SentimentLabel
from src.agents.sentiment.sentiment_agent import SentimentAgent


def test_fast_lane_scores_indian_market_headline_with_source_ttl():
    agent = SentimentAgent.from_default_components()

    prediction = agent.score_realtime(
        {
            "source_id": "headline_fast_1",
            "headline": "Nifty rally gains strength after RBI commentary and earnings beat",
            "source_type": "rss_feed",
            "symbol": "NIFTY50",
            "timestamp": datetime.now(UTC),
        }
    )

    assert prediction.lane == SentimentLane.FAST
    assert prediction.label == SentimentLabel.POSITIVE
    assert prediction.ttl_seconds == 14_400
    assert prediction.latency_ms is not None
    assert prediction.latency_ms <= 100.0


def test_fast_lane_social_payload_uses_shorter_ttl():
    agent = SentimentAgent.from_default_components()

    prediction = agent.score_realtime(
        {
            "source_id": "headline_fast_2",
            "headline": "Bearish chatter spikes on X after weak guidance",
            "source_type": "social_media",
            "platform": "X",
            "symbol": "INFY.NS",
            "timestamp": datetime.now(UTC),
        }
    )

    assert prediction.lane == SentimentLane.FAST
    assert prediction.ttl_seconds == 3_600
