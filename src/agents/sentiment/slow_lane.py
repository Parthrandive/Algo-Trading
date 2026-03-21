from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from src.agents.sentiment.cache_policy import resolve_ttl_seconds
from src.agents.sentiment.schemas import (
    DailySentimentAggregate,
    NightlySentimentBatchResult,
    SentimentLane,
    SentimentPrediction,
)

if TYPE_CHECKING:
    from src.agents.sentiment.sentiment_agent import SentimentAgent


class SlowLaneSentimentScorer:
    def __init__(
        self,
        agent: "SentimentAgent",
        *,
        source_ttls: Mapping[str, int] | None = None,
    ):
        self.agent = agent
        self.source_ttls = dict(source_ttls or {})

    def score_payload(
        self,
        payload: Mapping[str, Any],
        *,
        as_of_utc: datetime | None = None,
    ) -> SentimentPrediction:
        prediction = self.agent.score_textual_payload(payload, lane=SentimentLane.SLOW, as_of_utc=as_of_utc)
        ttl_seconds = resolve_ttl_seconds(
            payload,
            fallback_seconds=self.agent.slow_ttl_seconds,
            source_ttls=self.source_ttls,
        )
        headline_timestamp = payload.get("timestamp")
        if isinstance(headline_timestamp, datetime):
            if headline_timestamp.tzinfo is not None:
                headline_timestamp = headline_timestamp.astimezone(UTC)
        else:
            headline_timestamp = None
        return prediction.model_copy(
            update={
                "ttl_seconds": ttl_seconds,
                "symbol": payload.get("symbol"),
                "source_type": payload.get("source_type"),
                "headline_timestamp_utc": headline_timestamp,
                "score_timestamp_utc": prediction.generated_at_utc,
            }
        )

    def run_nightly_batch(
        self,
        payloads: Iterable[Mapping[str, Any]],
        *,
        macro_by_symbol: Mapping[str, Mapping[str, float]] | None = None,
        as_of_utc: datetime | None = None,
        lookback_hours: int = 24,
    ) -> NightlySentimentBatchResult:
        started = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        predictions: list[SentimentPrediction] = []
        grouped_predictions: dict[str, list[SentimentPrediction]] = defaultdict(list)

        for payload in payloads:
            prediction = self.score_payload(payload, as_of_utc=started)
            predictions.append(prediction)
            symbol = str(payload.get("symbol") or "MARKET")
            grouped_predictions[symbol].append(prediction)

        symbol_aggregates: list[DailySentimentAggregate] = []
        for symbol, symbol_predictions in grouped_predictions.items():
            aggregate = self.agent.compute_daily_z_t(
                symbol_predictions,
                macro_features=(macro_by_symbol or {}).get(symbol),
                as_of_utc=started,
            ).model_copy(update={"symbol": None if symbol == "MARKET" else symbol})
            symbol_aggregates.append(aggregate)

        market_predictions = predictions if predictions else []
        market_aggregate = self.agent.compute_daily_z_t(
            market_predictions,
            macro_features=(macro_by_symbol or {}).get("MARKET"),
            as_of_utc=started,
        ).model_copy(update={"symbol": None})

        return NightlySentimentBatchResult(
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            lookback_hours=lookback_hours,
            document_predictions=tuple(predictions),
            symbol_aggregates=tuple(aggregate for aggregate in symbol_aggregates if aggregate.symbol is not None),
            market_aggregate=market_aggregate,
        )
