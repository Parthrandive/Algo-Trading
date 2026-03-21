from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Mapping, TYPE_CHECKING

from src.agents.sentiment.cache_policy import resolve_ttl_seconds
from src.agents.sentiment.schemas import SentimentLane, SentimentPrediction

if TYPE_CHECKING:
    from src.agents.sentiment.sentiment_agent import SentimentAgent


class FastLaneSentimentScorer:
    def __init__(
        self,
        agent: "SentimentAgent",
        *,
        source_ttls: Mapping[str, int] | None = None,
        target_latency_ms: float = 100.0,
    ):
        self.agent = agent
        self.source_ttls = dict(source_ttls or {})
        self.target_latency_ms = max(1.0, float(target_latency_ms))

    def score_headline(
        self,
        payload: Mapping[str, Any],
        *,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        started = perf_counter()
        prediction = self.agent.score_textual_payload(payload, lane=SentimentLane.FAST, as_of_utc=as_of_utc)
        latency_ms = (perf_counter() - started) * 1000.0
        ttl_seconds = resolve_ttl_seconds(
            payload,
            fallback_seconds=self.agent.fast_ttl_seconds,
            source_ttls=self.source_ttls,
        )
        headline_timestamp = payload.get("timestamp")
        if isinstance(headline_timestamp, datetime):
            if headline_timestamp.tzinfo is not None:
                headline_timestamp = headline_timestamp.astimezone(UTC)
        else:
            headline_timestamp = None

        provenance = dict(prediction.provenance)
        provenance["latency_target_met"] = latency_ms <= self.target_latency_ms
        return prediction.model_copy(
            update={
                "ttl_seconds": ttl_seconds,
                "symbol": payload.get("symbol"),
                "source_type": payload.get("source_type"),
                "headline_timestamp_utc": headline_timestamp,
                "score_timestamp_utc": prediction.generated_at_utc,
                "latency_ms": latency_ms,
                "provenance": provenance,
            }
        )
