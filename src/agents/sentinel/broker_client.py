from __future__ import annotations

from datetime import UTC, datetime
from typing import List, Optional

import requests

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, QualityFlag, SourceType, Tick
from src.utils.resilience import rate_limit, retry_with_backoff


class BrokerAPIClient(NSEClientInterface):
    """
    Generic broker REST connector.
    Expected endpoints:
      - GET {base_url}/quote?symbol=...
      - GET {base_url}/historical?symbol=...&start=...&end=...&interval=...
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        timeout_seconds: float = 10.0,
        session: Optional[requests.Session] = None,
        quote_endpoint: str = "/quote",
        historical_endpoint: str = "/historical",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.access_token = access_token
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.quote_endpoint = quote_endpoint
        self.historical_endpoint = historical_endpoint
        self.source_type = SourceType.BROKER_API

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _parse_timestamp(self, raw_value: object) -> datetime:
        if isinstance(raw_value, datetime):
            dt = raw_value
        elif isinstance(raw_value, (int, float)):
            dt = datetime.fromtimestamp(float(raw_value), tz=UTC)
        elif isinstance(raw_value, str):
            dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        else:
            dt = datetime.now(UTC)

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _coalesce(data: dict, keys: list[str], default=None):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return default

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=5, period=1)
    def get_stock_quote(self, symbol: str) -> Tick:
        response = self.session.get(
            f"{self.base_url}{self.quote_endpoint}",
            params={"symbol": symbol},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        price = float(self._coalesce(payload, ["price", "last_price", "ltp"]))
        volume = int(self._coalesce(payload, ["volume", "last_volume", "traded_volume"], default=0))
        bid = self._coalesce(payload, ["bid", "best_bid"])
        ask = self._coalesce(payload, ["ask", "best_ask"])
        ts = self._parse_timestamp(self._coalesce(payload, ["timestamp", "ts", "last_trade_time"]))

        return Tick(
            symbol=symbol,
            timestamp=ts,
            source_type=self.source_type,
            price=price,
            volume=volume,
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            quality_status=QualityFlag.PASS,
        )

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=2, period=1)
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> List[Bar]:
        response = self.session.get(
            f"{self.base_url}{self.historical_endpoint}",
            params={
                "symbol": symbol,
                "start": start_date.astimezone(UTC).isoformat(),
                "end": end_date.astimezone(UTC).isoformat(),
                "interval": interval,
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            rows = payload.get("bars", [])
        else:
            rows = payload

        bars: list[Bar] = []
        for row in rows:
            ts = self._parse_timestamp(self._coalesce(row, ["timestamp", "ts", "time"]))
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts,
                    source_type=self.source_type,
                    interval=str(self._coalesce(row, ["interval"], default=interval)),
                    open=float(self._coalesce(row, ["open", "o"])),
                    high=float(self._coalesce(row, ["high", "h"])),
                    low=float(self._coalesce(row, ["low", "l"])),
                    close=float(self._coalesce(row, ["close", "c"])),
                    volume=int(self._coalesce(row, ["volume", "v"], default=0)),
                    quality_status=QualityFlag.PASS,
                )
            )
        return bars
