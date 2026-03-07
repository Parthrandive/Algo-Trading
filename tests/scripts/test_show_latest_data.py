from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import show_latest_data
from src.schemas.market_data import Bar, SourceType, Tick


class FakeHistoricalClient:
    def __init__(self, bars: list[Bar]):
        self._bars = bars

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1h") -> list[Bar]:
        return self._bars


class FakeLiveClient:
    def get_stock_quote(self, symbol: str) -> Tick:
        return Tick(
            symbol=symbol,
            timestamp=datetime(2026, 2, 19, 9, 30, tzinfo=ZoneInfo("UTC")),
            source_type=SourceType.OFFICIAL_API,
            price=123.45,
            volume=321,
        )


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

    monkeypatch.setattr(show_latest_data, "_build_failover_client", lambda *args, **kwargs: FakeHistoricalClient(bars))
    monkeypatch.setattr(show_latest_data, "choose_client_order", lambda _: [("FakeLive", lambda: FakeLiveClient())])

    exit_code = show_latest_data.run([requested_symbol, "--from", "2026-02-18", "--to", "2026-02-20", "--json"])
    assert exit_code == show_latest_data.EXIT_SUCCESS

    # With DB-backed persistence, bars are saved to PostgreSQL, not parquet files.
    # So silver_files (parquet) will be empty, and bars_saved (parquet count) will be 0.

    output = capsys.readouterr().out
    assert "SUCCESS: Received Live Tick" in output
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
