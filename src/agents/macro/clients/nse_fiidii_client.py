"""
NSEDIIFIIClient — NSE / NSDL FII and DII daily flow client.

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
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
from src.agents.macro.parsers import FIIDIIParser
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
    Concrete client for NSE/NSDL FII and DII daily flow data.

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    Note: FII and DII flows are published on the same page (NSDL/NSE),
    so this single client handles both indicators efficiently with one
    underlying HTTP round-trip once implemented.

    Uses ``FIIDIIParser`` for schema-normalized output.
    A custom ``raw_fetcher`` can be injected for live/contract tests.
    """

    def __init__(
        self,
        raw_fetcher: Callable[[DateRange], dict[str, Any]] | None = None,
    ) -> None:
        self._raw_fetcher = raw_fetcher
        self._parser = FIIDIIParser()

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

        A custom raw fetcher can return either one or both flows for the date.
        Only the requested indicator is returned.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"NSEDIIFIIClient does not support indicator '{name}'. "
                f"Supported: {sorted(i.value for i in _SUPPORTED)}"
            )

        logger.info(
            "NSEDIIFIIClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )
        payload = (
            self._raw_fetcher(date_range)
            if self._raw_fetcher is not None
            else self._build_simulated_payload(date_range)
        )
        parsed = self._parser.parse(payload)
        records = [record for record in parsed if record.indicator_name == name]
        if not records:
            logger.warning("No %s records produced from NSE/NSDL payload", name.value)
        return records

    @staticmethod
    def _build_simulated_payload(date_range: DateRange) -> dict[str, Any]:
        """
        Deterministic fallback payload used when no network fetcher is injected.
        """
        day_seed = date_range.end.toordinal()
        fii_flow = round(((day_seed % 2000) - 1000) * 2.75, 2)
        dii_flow = round(-fii_flow * 0.8, 2)
        return {
            "date": date_range.end.isoformat(),
            "fii_flow": fii_flow,
            "dii_flow": dii_flow,
        }

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
