from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.strategic.config import OrderBookFeatureConfig
from src.agents.strategic.orderbook_features import (
    FeatureQuality,
    OrderBookFeaturePipeline,
    OrderBookSnapshot,
)


def _snapshot(*, ts: datetime, bids: tuple[float, ...], asks: tuple[float, ...], quality: str = "pass") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="RELIANCE.NS",
        timestamp=ts,
        bid_levels=bids,
        ask_levels=asks,
        bid_arrival_rate=120.0,
        ask_arrival_rate=80.0,
        source_quality=quality,
    )


def test_orderbook_feature_pipeline_computes_imbalance_and_queue_pressure():
    pipeline = OrderBookFeaturePipeline(config=OrderBookFeatureConfig(top_n_levels=3))
    now = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
    features = pipeline.compute(_snapshot(ts=now, bids=(100.0, 90.0, 80.0), asks=(70.0, 60.0, 50.0)))

    assert features.quality_flag == FeatureQuality.PASS
    assert features.degraded is False
    assert features.imbalance > 0.0
    assert features.queue_pressure > 0.0


def test_incomplete_levels_use_last_good_snapshot():
    pipeline = OrderBookFeaturePipeline(config=OrderBookFeatureConfig(top_n_levels=3))
    now = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
    good = pipeline.compute(_snapshot(ts=now, bids=(100.0, 90.0, 80.0), asks=(70.0, 60.0, 50.0)))

    degraded = pipeline.compute(
        _snapshot(ts=now + timedelta(milliseconds=300), bids=(100.0,), asks=(90.0,), quality="pass"),
    )
    assert degraded.quality_flag == FeatureQuality.INCOMPLETE
    assert degraded.degraded is True
    assert degraded.imbalance == good.imbalance
    assert degraded.queue_pressure == good.queue_pressure


def test_stale_snapshot_degrades_without_blocking():
    pipeline = OrderBookFeaturePipeline(
        config=OrderBookFeatureConfig(top_n_levels=3, staleness_threshold_seconds=1),
    )
    now = datetime(2026, 4, 16, 9, 30, tzinfo=UTC)
    stale = pipeline.compute(
        _snapshot(ts=now, bids=(10.0, 9.0, 8.0), asks=(7.0, 6.0, 5.0)),
        now=now + timedelta(seconds=2),
    )
    assert stale.quality_flag == FeatureQuality.STALE
    assert stale.degraded is True
    assert stale.degradation_reason == "stale_snapshot"
