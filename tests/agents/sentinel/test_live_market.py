from datetime import datetime
from zoneinfo import ZoneInfo

from src.agents.sentinel.live_market import finalize_live_observation_to_bar
from src.schemas.market_data import LiveMarketObservation, ObservationKind, QualityFlag, SourceType


def test_quote_observation_does_not_finalize_bar():
    obs = LiveMarketObservation(
        symbol="RELIANCE.NS",
        timestamp=datetime(2026, 2, 16, 10, 0, tzinfo=ZoneInfo("UTC")),
        source_type=SourceType.FALLBACK_SCRAPER,
        source_name="nsepython",
        observation_kind=ObservationKind.QUOTE,
        asset_type="equity",
        last_price=100.0,
        volume=1000,
    )

    assert finalize_live_observation_to_bar(obs, latest_finalized_bar_timestamp=None) is None


def test_explicit_final_bar_observation_can_finalize():
    obs = LiveMarketObservation(
        symbol="RELIANCE.NS",
        timestamp=datetime(2026, 2, 16, 10, 20, tzinfo=ZoneInfo("UTC")),
        source_type=SourceType.BROKER_API,
        source_name="broker_rest",
        observation_kind=ObservationKind.FINAL_BAR,
        asset_type="equity",
        interval="1h",
        last_price=101.0,
        open=100.0,
        high=102.0,
        low=99.5,
        close=101.0,
        volume=1200,
        bar_timestamp=datetime(2026, 2, 16, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")),
        is_final_bar=True,
        quality_status=QualityFlag.PASS,
    )

    bar = finalize_live_observation_to_bar(obs, latest_finalized_bar_timestamp=None)
    assert bar is not None
    assert bar.interval == "1h"
    assert bar.close == 101.0
