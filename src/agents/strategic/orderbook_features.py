from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Sequence

from src.agents.strategic.config import OrderBookFeatureConfig


class FeatureQuality(str, Enum):
    PASS = "pass"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    bid_levels: tuple[float, ...]
    ask_levels: tuple[float, ...]
    bid_arrival_rate: float
    ask_arrival_rate: float
    source_quality: str = "pass"


@dataclass(frozen=True)
class OrderBookFeatures:
    symbol: str
    timestamp: datetime
    imbalance: float
    queue_pressure: float
    quality_flag: FeatureQuality
    schema_version: str
    source_quality: str
    degraded: bool
    degradation_reason: str | None = None


class OrderBookFeaturePipeline:
    """
    Tier 1-C feature computation with non-blocking safe degradation.
    """

    def __init__(
        self,
        *,
        config: OrderBookFeatureConfig | None = None,
        schema_version: str = "2.0",
    ) -> None:
        self.config = config or OrderBookFeatureConfig()
        self.schema_version = schema_version
        self._last_good: dict[str, OrderBookFeatures] = {}
        self._events: list[dict[str, object]] = []

    def compute(
        self,
        snapshot: OrderBookSnapshot,
        *,
        now: datetime | None = None,
    ) -> OrderBookFeatures:
        now = (now or snapshot.timestamp).astimezone(UTC)
        snapshot_ts = snapshot.timestamp.astimezone(UTC)
        age_seconds = max(0.0, (now - snapshot_ts).total_seconds())

        top_n = int(self.config.top_n_levels)
        bids = tuple(snapshot.bid_levels[:top_n])
        asks = tuple(snapshot.ask_levels[:top_n])

        quality = FeatureQuality.PASS
        degradation_reason: str | None = None

        if age_seconds > float(self.config.staleness_threshold_seconds):
            quality = FeatureQuality.STALE
            degradation_reason = "stale_snapshot"
        elif len(bids) < top_n or len(asks) < top_n:
            quality = FeatureQuality.INCOMPLETE
            degradation_reason = "missing_book_levels"
        elif snapshot.source_quality.strip().lower() != "pass":
            quality = FeatureQuality.DEGRADED
            degradation_reason = "source_quality_degraded"

        if quality in {FeatureQuality.STALE, FeatureQuality.INCOMPLETE}:
            fallback = self._last_good.get(snapshot.symbol)
            if fallback is not None:
                features = OrderBookFeatures(
                    symbol=snapshot.symbol,
                    timestamp=snapshot_ts,
                    imbalance=fallback.imbalance,
                    queue_pressure=fallback.queue_pressure,
                    quality_flag=quality,
                    schema_version=self.schema_version,
                    source_quality=snapshot.source_quality,
                    degraded=True,
                    degradation_reason=degradation_reason,
                )
            else:
                features = OrderBookFeatures(
                    symbol=snapshot.symbol,
                    timestamp=snapshot_ts,
                    imbalance=0.0,
                    queue_pressure=0.0,
                    quality_flag=quality,
                    schema_version=self.schema_version,
                    source_quality=snapshot.source_quality,
                    degraded=True,
                    degradation_reason=degradation_reason,
                )
        else:
            imbalance = _compute_imbalance(bids, asks)
            queue_pressure = _compute_queue_pressure(
                snapshot.bid_arrival_rate,
                snapshot.ask_arrival_rate,
            )
            if quality == FeatureQuality.DEGRADED:
                downweight = float(self.config.degraded_downweight)
                imbalance *= downweight
                queue_pressure *= downweight
            features = OrderBookFeatures(
                symbol=snapshot.symbol,
                timestamp=snapshot_ts,
                imbalance=imbalance,
                queue_pressure=queue_pressure,
                quality_flag=quality,
                schema_version=self.schema_version,
                source_quality=snapshot.source_quality,
                degraded=quality != FeatureQuality.PASS,
                degradation_reason=degradation_reason,
            )
            self._last_good[snapshot.symbol] = features

        self._events.append(
            {
                "timestamp": now.isoformat(),
                "symbol": snapshot.symbol,
                "quality": features.quality_flag.value,
                "degraded": features.degraded,
                "reason": features.degradation_reason,
            }
        )
        return features

    def recent_events(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(0, limit) :])


def _compute_imbalance(bids: Sequence[float], asks: Sequence[float]) -> float:
    bid_qty = float(sum(max(0.0, level) for level in bids))
    ask_qty = float(sum(max(0.0, level) for level in asks))
    denom = bid_qty + ask_qty
    if denom <= 0.0:
        return 0.0
    return (bid_qty - ask_qty) / denom


def _compute_queue_pressure(bid_arrival_rate: float, ask_arrival_rate: float) -> float:
    bid = float(bid_arrival_rate)
    ask = float(ask_arrival_rate)
    denom = abs(bid) + abs(ask)
    if denom <= 0.0:
        return 0.0
    return (bid - ask) / denom
