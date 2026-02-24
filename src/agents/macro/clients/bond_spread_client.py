"""
BondSpreadClient — stub for India-US 10-Year bond spread computation.

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
from typing import Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
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


class BondSpreadClient:
    """
    Concrete client stub for India-US 10-Year Bond Spread.

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    When implemented (Day 4) this client will:
    1. Fetch the India 10Y yield from RBI Statistics.
    2. Fetch the US 10Y yield from FRED (fallback: US Treasury site).
    3. Compute spread_bps = (india_10y_pct - us_10y_pct) * 100.
    4. Return a ``MacroIndicator`` for ``INDIA_US_10Y_SPREAD``.

    Global proxy justification:
        India-US 10Y spread is a direct measure of capital flow pressure
        and is explicitly required in the Week 3 publish set (CP3).
    """

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

        Raises
        ------
        NotImplementedError
            Always in this stub.
        ValueError
            If ``name`` is not ``INDIA_US_10Y_SPREAD``.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"BondSpreadClient only supports INDIA_US_10Y_SPREAD, got '{name}'."
            )

        logger.info(
            "BondSpreadClient.get_indicator called: indicator=%s range=%s — stub",
            name.value,
            date_range,
        )
        raise NotImplementedError(
            "BondSpreadClient.get_indicator — "
            "RBI + FRED fetch + spread computation not implemented until Day 4."
        )

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
