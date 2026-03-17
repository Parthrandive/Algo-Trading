"""
BondSpreadClient — India-US 10-Year bond spread computation client.

Covers:
  INDIA_US_10Y_SPREAD — Daily, bps

Formula (from runtime config):
  spread_bps = (india_10y_percent - us_10y_percent) * 100

Source legs:
  India 10Y : RBI Statistics (rbi_india_10y_leg)
  US 10Y    : FRED DGS10     (fred_us_10y_leg)
  US 10Y FB : US Treasury    (treasury_us_10y_fallback)

Freshness SLA: daily_market_close_plus_6h (6-hour window).

This client is responsible only for the *computed* spread indicator.
The two raw legs (INDIA_10Y, US_10Y) are fetched by RBIClient / a separate
US rates client — this client will call them internally once wired.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
from src.agents.macro.clients.fred_client import FredSeriesSpec, fetch_series_history
from src.agents.macro.parsers import BondSpreadParser
from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    QualityFlag,
    SourceType,
)

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_SUPPORTED: frozenset[MacroIndicatorType] = frozenset(
    [MacroIndicatorType.INDIA_US_10Y_SPREAD]
)

_INDIA_10Y_FALLBACK = FredSeriesSpec(
    series_id="INDIRLTLT01STM",
    frequency="Monthly",
    source_label="fred:INDIRLTLT01STM",
)

_US_10Y_FALLBACK = FredSeriesSpec(
    series_id="DGS10",
    frequency="Daily",
    source_label="fred:DGS10",
)


class BondSpreadClient:
    """
    Concrete client for India-US 10-Year Bond Spread.

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    The parser computes spread_bps = (india_10y_pct - us_10y_pct) * 100.
    A custom ``raw_fetcher`` can be injected for live/contract tests.

    Global proxy justification:
        India-US 10Y spread is a direct measure of capital flow pressure
        and is explicitly required in the Week 3 publish set (CP3).
    """

    def __init__(
        self,
        raw_fetcher: Callable[[DateRange], dict[str, Any]] | None = None,
    ) -> None:
        self._raw_fetcher = raw_fetcher
        self._parser = BondSpreadParser()

    @property
    def supported_indicators(self) -> frozenset[MacroIndicatorType]:
        return _SUPPORTED

    def get_indicator(
        self,
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> Sequence[MacroIndicator]:
        """
        Compute and return India-US 10Y spread records for the given date range.

        Returns schema-normalized spread records for the requested range.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"BondSpreadClient only supports INDIA_US_10Y_SPREAD, got '{name}'."
            )

        logger.info(
            "BondSpreadClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )
        if self._raw_fetcher is not None:
            used_fallback = False
            payload = self._raw_fetcher(date_range)
        else:
            used_fallback = True
            payload = self._build_historical_payload(date_range)
        self._parser.source_type = (
            SourceType.FALLBACK_SCRAPER if used_fallback else SourceType.OFFICIAL_API
        )
        records = list(self._parser.parse(payload))
        if used_fallback:
            records = [
                record.model_copy(update={"quality_status": QualityFlag.WARN})
                for record in records
            ]
        return records

    @staticmethod
    def _build_historical_payload(date_range: DateRange) -> dict[str, Any]:
        """
        Build spread payload from deep-history fallback legs.

        - India 10Y: monthly observations (FRED-hosted OECD series).
        - US 10Y: daily observations (FRED DGS10).
        """
        try:
            india_rows = fetch_series_history(_INDIA_10Y_FALLBACK, date_range)
            us_rows = fetch_series_history(_US_10Y_FALLBACK, date_range)
        except Exception as exc:
            logger.warning(
                "Bond spread fallback series fetch failed for %s -> %s (%s).",
                date_range.start,
                date_range.end,
                exc,
            )
            return {"data": []}

        if not india_rows or not us_rows:
            return {"data": []}

        us_by_date = {row["date"]: float(row["value"]) for row in us_rows}
        us_sorted_dates = sorted(us_by_date.keys())

        records: list[dict[str, Any]] = []
        latest_us = None
        us_idx = 0

        for india_row in sorted(india_rows, key=lambda r: r["date"]):
            obs_date = india_row["date"]
            while us_idx < len(us_sorted_dates) and us_sorted_dates[us_idx] <= obs_date:
                latest_us = us_by_date[us_sorted_dates[us_idx]]
                us_idx += 1
            if latest_us is None:
                continue

            records.append(
                {
                    "date": obs_date,
                    "india_10y_percent": float(india_row["value"]),
                    "us_10y_percent": float(latest_us),
                }
            )

        return {"data": records}

    def _make_stub_record(
        self,
        india_10y_pct: float,
        us_10y_pct: float,
        observation_date: datetime,
        quality_status: QualityFlag = QualityFlag.PASS,
    ) -> MacroIndicator:
        """
        Compute spread and return a provenance-tagged record (for tests only).

        spread_bps = (india_10y_pct - us_10y_pct) * 100
        """
        spread_bps = round((india_10y_pct - us_10y_pct) * 100, 4)
        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=MacroIndicatorType.INDIA_US_10Y_SPREAD,
            value=spread_bps,
            unit="bps",
            period="Daily",
            timestamp=observation_date,
            source_type=SourceType.OFFICIAL_API,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
        )
