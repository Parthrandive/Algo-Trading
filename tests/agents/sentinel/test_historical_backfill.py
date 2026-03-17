from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine

from src.agents.sentinel.historical_backfill import HistoricalBackfillService
from src.schemas.market_data import Bar, SourceType


class DummyRecorder:
    def save_bars(self, bars):
        self.bars = list(bars)

    def save_market_data_quality(self, records):
        self.records = list(records)


class StaticProvider:
    def __init__(self, name: str, bars: list[Bar]):
        self.name = name
        self.source_type = SourceType.FALLBACK_SCRAPER
        self._bars = bars

    def fetch_bars(self, symbol: str, start_date: datetime, end_date: datetime, interval: str):
        return list(self._bars)


def _bars(count: int) -> list[Bar]:
    ist = ZoneInfo("Asia/Kolkata")
    base = datetime(2026, 2, 16, 9, 15, tzinfo=ist)
    result = []
    for idx in range(count):
        ts = base.replace(hour=9 + idx)
        result.append(
            Bar(
                symbol="RELIANCE.NS",
                timestamp=ts,
                source_type=SourceType.FALLBACK_SCRAPER,
                interval="1h",
                open=100 + idx,
                high=101 + idx,
                low=99 + idx,
                close=100.5 + idx,
                volume=1000 + idx,
            )
        )
    return result


def test_fetch_merged_history_tries_next_provider_when_first_is_shallow(monkeypatch):
    monkeypatch.setattr(
        "src.agents.sentinel.historical_backfill.ensure_market_schema",
        lambda database_url=None: None,
    )
    monkeypatch.setattr(
        "src.agents.sentinel.historical_backfill.get_engine",
        lambda database_url=None: create_engine("sqlite:///:memory:"),
    )

    service = HistoricalBackfillService(
        providers=[
            StaticProvider("shallow", _bars(2)),
            StaticProvider("deep", _bars(5)),
        ],
        silver_recorder=DummyRecorder(),
    )
    service.thresholds = service.thresholds.__class__(
        min_rows_by_interval={"1h": 5},
        min_history_days=1,
        min_coverage_pct=50.0,
        max_zero_volume_ratio=1.0,
    )

    merged, sources_used, errors, duplicate_count = service.fetch_merged_history(
        symbol="RELIANCE.NS",
        start_date=datetime(2026, 2, 16, 0, 0, tzinfo=ZoneInfo("UTC")),
        end_date=datetime(2026, 2, 16, 23, 59, tzinfo=ZoneInfo("UTC")),
        interval="1h",
    )

    assert len(merged) == 5
    assert sources_used == ["shallow", "deep"]
    assert errors == []
    assert duplicate_count >= 2

