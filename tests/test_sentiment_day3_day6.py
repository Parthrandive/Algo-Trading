from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.sentiment.cache import InMemorySentimentCache
from src.agents.sentiment.models import FinBERTSentimentModel, KeywordSentimentModel
from src.agents.sentiment.schemas import CacheFreshnessState, SentimentLabel, SentimentLane, SentimentQualityStatus
from src.agents.sentiment.sentiment_agent import SentimentAgent


class FailingCache:
    def get(self, cache_key: str, *, as_of_utc=None, include_expired: bool = False):
        raise RuntimeError("cache read failed")

    def set(self, cache_key: str, entry):
        raise RuntimeError("cache write failed")


def _agent_with_inmemory_cache() -> SentimentAgent:
    return SentimentAgent(
        fast_model=KeywordSentimentModel(),
        slow_model=FinBERTSentimentModel.bootstrap(enable_hf_pipeline=False),
        cache=InMemorySentimentCache(),
        fast_ttl_seconds=100,
        slow_ttl_seconds=500,
        fast_stale_after_seconds=20,
        slow_stale_after_seconds=100,
        stale_downweight_factor=0.5,
    )


def test_cache_policy_fresh_stale_expired_paths():
    agent = _agent_with_inmemory_cache()
    t0 = datetime.now(UTC)

    first = agent.score(
        "Nifty rally continues with strong breadth",
        source_id="s1",
        lane=SentimentLane.FAST,
        as_of_utc=t0,
    )
    fresh = agent.score(
        "Nifty rally continues with strong breadth",
        source_id="s1",
        lane=SentimentLane.FAST,
        as_of_utc=t0 + timedelta(seconds=10),
    )
    stale = agent.score(
        "Nifty rally continues with strong breadth",
        source_id="s1",
        lane=SentimentLane.FAST,
        as_of_utc=t0 + timedelta(seconds=40),
    )
    expired = agent.score(
        "Nifty rally continues with strong breadth",
        source_id="s1",
        lane=SentimentLane.FAST,
        as_of_utc=t0 + timedelta(seconds=150),
    )

    assert first.cache_hit is False
    assert fresh.cache_hit is True
    assert fresh.freshness_state == CacheFreshnessState.FRESH

    assert stale.cache_hit is True
    assert stale.freshness_state == CacheFreshnessState.STALE
    assert stale.downweighted is True
    assert abs(stale.score) <= abs(fresh.score)

    assert expired.cache_hit is False
    assert expired.freshness_state == CacheFreshnessState.EXPIRED


def test_payload_manipulation_filters_trigger_quality_downgrade():
    agent = _agent_with_inmemory_cache()
    payload = {
        "source_id": "x_spam_1",
        "content": "PUMP AND DUMP!!! 100% success rate, no risk, guaranteed returns!!!",
        "language": "code_mixed",
        "manipulation_risk_score": 0.4,
    }

    prediction = agent.score_textual_payload(payload, lane=SentimentLane.FAST)

    assert prediction.quality_status in {SentimentQualityStatus.WARN, SentimentQualityStatus.FAIL}
    assert prediction.manipulation_risk_score >= 0.35
    assert any("scam_pattern" in flag for flag in prediction.manipulation_flags)


def test_price_mismatch_circuit_breaker_downgrades_signal():
    agent = _agent_with_inmemory_cache()
    payload = {
        "source_id": "headline_1",
        "headline": "Nifty rally and bullish upgrade expected",
        "price_return": -0.05,
    }

    prediction = agent.score_textual_payload(payload, lane=SentimentLane.FAST)

    assert prediction.label == SentimentLabel.NEUTRAL
    assert prediction.score == 0.0
    assert "sentiment_price_mismatch" in prediction.manipulation_flags


def test_batch_scoring_deduplicates_repeated_source_payloads():
    agent = _agent_with_inmemory_cache()
    payloads = [
        {"source_id": "dup_src", "headline": "Rupee weakens on oil spike"},
        {"source_id": "dup_src", "headline": "Rupee weakens on oil spike"},
        {"source_id": "uniq_src", "headline": "RBI holds rates steady"},
    ]

    predictions = agent.score_textual_records(payloads, lane=SentimentLane.FAST, dedupe=True)
    assert len(predictions) == 2


def test_cache_failure_triggers_reduced_risk_mode():
    agent = SentimentAgent(
        fast_model=KeywordSentimentModel(),
        slow_model=FinBERTSentimentModel.bootstrap(enable_hf_pipeline=False),
        cache=FailingCache(),
        fast_ttl_seconds=60,
        slow_ttl_seconds=120,
    )

    prediction = agent.score(
        "Nifty outlook is mixed",
        source_id="cache_fail_src",
        lane=SentimentLane.FAST,
    )

    assert prediction.reduced_risk_mode is True
    assert prediction.fallback_mode == "technical_only_reduced_risk"
    assert prediction.quality_status == SentimentQualityStatus.FAIL


def test_daily_zt_aggregate_uses_sentiment_and_macro_inputs():
    agent = _agent_with_inmemory_cache()
    p1 = agent.score("Nifty rally gains strength", source_id="a", lane=SentimentLane.FAST)
    p2 = agent.score("Markets turn weak after guidance cut", source_id="b", lane=SentimentLane.FAST)

    aggregate = agent.compute_daily_z_t(
        [p1, p2],
        macro_features={"cpi_surprise": 0.2, "fx_stress": -0.1, "policy_bias": 0.15},
    )

    assert aggregate.sample_size == 2
    assert -1.0 <= aggregate.weighted_sentiment_score <= 1.0
    assert -1.0 <= aggregate.z_t <= 1.0
    assert aggregate.quality_status in {
        SentimentQualityStatus.PASS,
        SentimentQualityStatus.WARN,
        SentimentQualityStatus.FAIL,
    }
