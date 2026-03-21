from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.consensus import AgentSignal, ConsensusAgent, ConsensusInput
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.db.models import Base, TextItemDB
from src.db.phase2_recorder import Phase2Recorder


def _build_sqlite_recorder() -> Phase2Recorder:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Phase2Recorder(engine=engine, session_factory=session_factory)


def test_negative_sentiment_flood_does_not_flip_consensus_direction():
    agent = ConsensusAgent()
    payload = ConsensusInput(
        technical=AgentSignal(name="technical", score=0.95, confidence=0.88),
        regime=AgentSignal(name="regime", score=0.82, confidence=0.84),
        sentiment=AgentSignal(name="sentiment", score=-0.35, confidence=0.92),
        volatility=0.22,
        macro_differential=0.10,
        rbi_signal=0.05,
        sentiment_quantile=0.25,
        crisis_probability=0.10,
        generated_at_utc=datetime.now(timezone.utc),
    )

    result = agent.run(payload)
    assert result.score > 0.0


def test_expired_cached_sentiment_triggers_reduced_risk_fallback():
    recorder = _build_sqlite_recorder()
    agent = SentimentAgent.from_default_components(phase2_recorder=recorder, persist_predictions=True)
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    recorder.save_sentiment_score(
        {
            "symbol": "INFY.NS",
            "timestamp": old_time,
            "lane": "fast",
            "source_id": "old_sentiment_row",
            "source_type": "rss_feed",
            "sentiment_class": "positive",
            "sentiment_score": 0.6,
            "confidence": 0.8,
            "source_count": 1,
            "ttl_seconds": 3600,
            "freshness_flag": "fresh",
            "score_timestamp": old_time,
            "model_id": "keyword_rule_v1",
        }
    )

    cached = agent.get_cached_sentiment("INFY.NS", as_of_utc=datetime.now(timezone.utc))

    assert cached.reduced_risk_mode is True
    assert cached.fallback_mode == "technical_only_reduced_risk"


def test_hinglish_realtime_payload_returns_valid_contract():
    agent = SentimentAgent.from_default_components()
    prediction = agent.score_realtime(
        {
            "source_id": "hinglish_1",
            "headline": "Nifty aaj strong lag raha hai but volume weak hai",
            "source_type": "social_media",
            "platform": "X",
            "symbol": "NIFTY50",
            "timestamp": datetime.now(timezone.utc),
        }
    )

    assert prediction.score == prediction.score
    assert prediction.confidence == prediction.confidence
    assert prediction.symbol == "NIFTY50"
