"""
NSEDIIFIIClient — stub for NSE / NSDL FII and DII daily flow data.

Covers:
  FII_FLOW  — Daily, INR_Cr  (NSDL FPI Reports / NSE FII-DII page)
  DII_FLOW  — Daily, INR_Cr  (NSDL FPI Reports / NSE FII-DII page)

Source URLs (from runtime config):
  Primary  : https://www.fpi.nsdl.co.in/web/Reports/Latest.aspx
  Fallback : https://www.nseindia.com/reports/fii-dii

Freshness SLA: data available by EOD + 4 hours (daily_eod_plus_4h schedule).
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
    [
        MacroIndicatorType.FII_FLOW,
        MacroIndicatorType.DII_FLOW,
    ]
)


class NSEDIIFIIClient:
    """
    Concrete client stub for NSE/NSDL FII and DII daily flow data.

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    Note: FII and DII flows are published on the same page (NSDL/NSE),
    so this single client handles both indicators efficiently with one
    underlying HTTP round-trip once implemented.

    Day 4 will wire in the real ``FIIDIIParser``.
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
        Fetch FII or DII flow values for the given date range.

        Raises
        ------
        NotImplementedError
            Always in this stub.
        ValueError
            If ``name`` not in ``supported_indicators``.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"NSEDIIFIIClient does not support indicator '{name}'. "
                f"Supported: {sorted(i.value for i in _SUPPORTED)}"
            )

        logger.info(
            "NSEDIIFIIClient.get_indicator called: indicator=%s range=%s — stub",
            name.value,
            date_range,
        )
        raise NotImplementedError(
            f"NSEDIIFIIClient.get_indicator({name.value!r}) — "
            "HTTP fetch + FIIDIIParser not implemented until Day 4."
        )

    def _make_stub_record(
        self,
        name: MacroIndicatorType,
        value: float,
        observation_date: datetime,
        quality_status: QualityFlag = QualityFlag.PASS,
        source_type: SourceType = SourceType.OFFICIAL_API,
    ) -> MacroIndicator:
        """Return a fully-provenance-tagged stub record (for tests only)."""
        if name not in _SUPPORTED:
            raise ValueError(f"NSEDIIFIIClient cannot produce stub for {name.value!r}")
        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=name,
            value=value,
            unit="INR_Cr",
            period="Daily",
            timestamp=observation_date,
            source_type=source_type,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
        )
