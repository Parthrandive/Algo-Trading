from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import show_latest_data
from src.schemas.market_data import Bar, SourceType


class FakeHistoricalBackfillService:
    def __init__(self, bars: list[Bar]):
        self._bars = bars

    def backfill_symbol(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1h") -> dict:
        return {
            "symbol": symbol,
            "bars_fetched": len(self._bars),
            "bars_added": 0,
            "quality": {
                "status": "train_ready",
                "train_ready": True,
                "coverage_pct": 100.0,
            },
        }


class FakeLiveMarketService:
    def poll_symbol(self, symbol: str) -> dict:
        return {
            "status": "SUCCESS",
            "symbol": symbol,
            "last_price": 123.45,
            "volume": 321,
            "source_name": "fake_live",
            "timestamp": "2026-02-19T09:30:00+00:00",
            "freshness_status": "fresh",
            "source_status": "ok",
            "message": None,
        }


def test_new_symbol_autofetch_fetches_live_and_writes_silver(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(show_latest_data, "PROJECT_ROOT", tmp_path)

    requested_symbol = "INTEGRATIONNEW"
    normalized_symbol = "INTEGRATIONNEW.NS"
    bars = [
        Bar(
            symbol=normalized_symbol,
            timestamp=datetime(2026, 2, 19, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")),
            source_type=SourceType.OFFICIAL_API,
            interval="1h",
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1000,
        )
    ]

    monkeypatch.setattr(show_latest_data, "_get_latest_timestamp", lambda _symbol: None)
    monkeypatch.setattr(
        show_latest_data,
        "_build_historical_backfill_service",
        lambda: FakeHistoricalBackfillService(bars),
    )
    monkeypatch.setattr(
        show_latest_data,
        "_get_historical_bars",
        lambda symbol, start_date, end_date, interval: pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timestamp": bars[0].timestamp.astimezone(ZoneInfo("UTC")),
                    "open": bars[0].open,
                    "high": bars[0].high,
                    "low": bars[0].low,
                    "close": bars[0].close,
                    "volume": bars[0].volume,
                }
            ]
        ),
    )
    monkeypatch.setattr(show_latest_data, "_build_live_market_service", lambda: FakeLiveMarketService())
    monkeypatch.setattr(show_latest_data, "_run_preprocessing", lambda _symbol: None)

    exit_code = show_latest_data.run([requested_symbol, "--from", "2026-02-18", "--to", "2026-02-20", "--json"])
    assert exit_code == show_latest_data.EXIT_SUCCESS

    # With DB-backed persistence, bars are saved to PostgreSQL, not parquet files.
    # So silver_files (parquet) will be empty, and bars_saved (parquet count) will be 0.

    output = capsys.readouterr().out
    assert "SUCCESS: Received Live Observation" in output
    assert "Fetch persistence summary:" in output
    assert "bars_fetched: 1" in output
    assert "bars_saved: 0" in output  # Data goes to DB, not parquet

    json_start = output.rfind('{\n  "requested_symbol"')
    assert json_start != -1
    payload = json.loads(output[json_start:])
    assert payload["historical"]["status"] == "SUCCESS"
    assert payload["historical"]["bars_fetched"] == 1
    assert payload["historical"]["bars_saved"] == 0  # DB-backed, no parquet
    assert payload["live"]["status"] == "SUCCESS"
