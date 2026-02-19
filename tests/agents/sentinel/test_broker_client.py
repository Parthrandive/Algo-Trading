from datetime import datetime, timezone

from src.agents.sentinel.broker_client import BrokerAPIClient
from src.schemas.market_data import SourceType


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return self._responses.pop(0)


def test_broker_quote_parsing():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "symbol": "RELIANCE",
                    "last_price": 1432.5,
                    "volume": 12345,
                    "bid": 1432.4,
                    "ask": 1432.6,
                    "timestamp": "2026-02-19T09:15:00+05:30",
                }
            )
        ]
    )
    client = BrokerAPIClient(base_url="https://broker.example.com/v1", session=session)
    tick = client.get_stock_quote("RELIANCE")

    assert tick.source_type == SourceType.BROKER_API
    assert tick.price == 1432.5
    assert tick.volume == 12345
    assert tick.timestamp.tzinfo is not None


def test_broker_historical_parsing():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "bars": [
                        {
                            "time": "2026-02-19T09:15:00+05:30",
                            "o": 100.0,
                            "h": 102.0,
                            "l": 99.5,
                            "c": 101.5,
                            "v": 5000,
                        }
                    ]
                }
            )
        ]
    )
    client = BrokerAPIClient(base_url="https://broker.example.com/v1", session=session)
    bars = client.get_historical_data(
        symbol="RELIANCE",
        start_date=datetime(2026, 2, 18, tzinfo=timezone.utc),
        end_date=datetime(2026, 2, 20, tzinfo=timezone.utc),
        interval="1h",
    )

    assert len(bars) == 1
    assert bars[0].source_type == SourceType.BROKER_API
    assert bars[0].interval == "1h"
