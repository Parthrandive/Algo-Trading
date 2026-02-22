from __future__ import annotations

from datetime import UTC, datetime
from typing import List, Optional

import requests

from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Bar, CorporateAction, CorporateActionType, QualityFlag, SourceType, Tick
from src.utils.resilience import rate_limit, retry_with_backoff


class BrokerAPIClient(NSEClientInterface):
    """
    Generic broker REST connector.
    Expected endpoints:
      - GET {base_url}/quote?symbol=...
      - GET {base_url}/historical?symbol=...&start=...&end=...&interval=...
      - GET {base_url}/corporate-actions?symbol=...&start=...&end=...
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
        corporate_actions_endpoint: str = "/corporate-actions",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.access_token = access_token
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.quote_endpoint = quote_endpoint
        self.historical_endpoint = historical_endpoint
        self.corporate_actions_endpoint = corporate_actions_endpoint
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

    @staticmethod
    def _parse_action_type(raw_value: object) -> Optional[CorporateActionType]:
        if raw_value is None:
            return None
        normalized = str(raw_value).strip().lower().replace("_", " ").replace("-", " ")
        if "dividend" in normalized:
            return CorporateActionType.DIVIDEND
        if "split" in normalized or "sub division" in normalized or "subdivision" in normalized:
            return CorporateActionType.SPLIT
        if "bonus" in normalized:
            return CorporateActionType.BONUS
        if "rights" in normalized:
            return CorporateActionType.RIGHTS
        return None

    @staticmethod
    def _normalize_ratio(raw_value: object) -> Optional[str]:
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            return f"{raw_value}:1"
        normalized = str(raw_value).strip().replace("/", ":")
        if ":" in normalized:
            return normalized
        try:
            ratio_value = float(normalized)
            return f"{ratio_value}:1"
        except ValueError:
            return None

    def _build_action_from_row(self, symbol: str, row: dict) -> Optional[CorporateAction]:
        action_type = self._parse_action_type(
            self._coalesce(
                row,
                ["action_type", "type", "event_type", "purpose", "category", "ca_type"],
            )
        )
        if action_type is None:
            return None

        ex_date_raw = self._coalesce(row, ["ex_date", "exDate", "exdate", "effective_date", "date"])
        if ex_date_raw is None:
            return None
        ex_date = self._parse_timestamp(ex_date_raw)

        record_date_raw = self._coalesce(row, ["record_date", "recordDate", "recorddate"])
        record_date = self._parse_timestamp(record_date_raw) if record_date_raw is not None else None

        ratio = self._normalize_ratio(
            self._coalesce(
                row,
                ["ratio", "ratio_str", "split_ratio", "bonus_ratio", "rights_ratio"],
            )
        )
        value_raw = self._coalesce(row, ["value", "amount", "dividend", "cash_dividend"])
        value = float(value_raw) if value_raw is not None else None
        timestamp = self._parse_timestamp(
            self._coalesce(row, ["timestamp", "ts", "updated_at", "ingested_at"], default=datetime.now(UTC))
        )

        try:
            return CorporateAction(
                symbol=symbol,
                timestamp=timestamp,
                source_type=self.source_type,
                action_type=action_type,
                ratio=ratio,
                value=value,
                ex_date=ex_date,
                record_date=record_date,
                quality_status=QualityFlag.PASS,
            )
        except Exception:
            return None

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

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @rate_limit(calls=2, period=1)
    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[CorporateAction]:
        response = self.session.get(
            f"{self.base_url}{self.corporate_actions_endpoint}",
            params={
                "symbol": symbol,
                "start": start_date.astimezone(UTC).isoformat(),
                "end": end_date.astimezone(UTC).isoformat(),
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            rows = (
                payload.get("actions")
                or payload.get("corporate_actions")
                or payload.get("data")
                or payload.get("records")
                or []
            )
        else:
            rows = payload

        actions: list[CorporateAction] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            action = self._build_action_from_row(symbol, row)
            if action is None:
                continue
            if start_date.astimezone(UTC) <= action.ex_date <= end_date.astimezone(UTC):
                actions.append(action)

        actions.sort(key=lambda item: item.ex_date)
        return actions
