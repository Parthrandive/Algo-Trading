"""
RBIClient — stub for Reserve Bank of India data.

Covers:
  FX_RESERVES    — Weekly, USD_Bn   (RBI Weekly Statistical Supplement)
  RBI_BULLETIN   — Irregular, count (publication event marker, value=1.0)
  INDIA_10Y      — Daily, %         (RBI government securities yield)

Source URLs (from runtime config):
  FX_RESERVES    : https://www.rbi.org.in/scripts/WSSView.aspx?Id=1
  RBI_BULLETIN   : https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx
  INDIA_10Y      : https://www.rbi.org.in/scripts/Statistics.aspx

This stub raises ``NotImplementedError`` for actual HTTP calls.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
from src.agents.macro.parsers import RBIBulletinParser
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
        MacroIndicatorType.FX_RESERVES,
        MacroIndicatorType.RBI_BULLETIN,
        MacroIndicatorType.INDIA_10Y,
    ]
)

_UNITS: dict[MacroIndicatorType, str] = {
    MacroIndicatorType.FX_RESERVES: "USD_Bn",
    MacroIndicatorType.RBI_BULLETIN: "count",  # event marker; value always 1.0
    MacroIndicatorType.INDIA_10Y: "%",
}

_PERIODS: dict[MacroIndicatorType, str] = {
    MacroIndicatorType.FX_RESERVES: "Weekly",
    MacroIndicatorType.RBI_BULLETIN: "Irregular",
    MacroIndicatorType.INDIA_10Y: "Daily",
}


class RBIClient:
    """
    Concrete client stub for RBI data (FX Reserves, RBI Bulletins, India 10Y).

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).
    """

    def __init__(self) -> None:
        self._parsers = {
            MacroIndicatorType.RBI_BULLETIN: RBIBulletinParser(),
        }

    @property
    def supported_indicators(self) -> frozenset[MacroIndicatorType]:
        return _SUPPORTED

    def get_indicator(
        self,
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> Sequence[MacroIndicator]:
        """
        Fetch RBI indicator values for the given date range.

        Raises
        ------
        NotImplementedError
            Always in this stub.
        ValueError
            If ``name`` not in ``supported_indicators``.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"RBIClient does not support indicator '{name}'. "
                f"Supported: {sorted(i.value for i in _SUPPORTED)}"
            )

        logger.info(
            "RBIClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )

        if name == MacroIndicatorType.RBI_BULLETIN:
            # Simulated raw payload for RBI Bulletin
            simulated_raw = {
                "publications": [
                    {
                        "date": datetime.combine(date_range.end, datetime.min.time()).isoformat(),
                        "title": "Simulated RBI Bulletin"
                    }
                ]
            }
            return self._parsers[name].parse(simulated_raw)

        raise NotImplementedError(
            f"RBIClient.get_indicator({name.value!r}) — "
            "HTTP fetch + parser not implemented yet (scheduled for Day 4 for this indicator)."
        )

    def _make_stub_record(
        self,
        name: MacroIndicatorType,
        value: float,
        observation_date: datetime,
        quality_status: QualityFlag = QualityFlag.PASS,
    ) -> MacroIndicator:
        """Return a fully-provenance-tagged stub record (for tests only)."""
        if name not in _SUPPORTED:
            raise ValueError(f"RBIClient cannot produce stub for {name.value!r}")

        # RBI_BULLETIN must always encode as value=1.0 (publication event marker)
        if name == MacroIndicatorType.RBI_BULLETIN:
            value = 1.0

        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=name,
            value=value,
            unit=_UNITS[name],
            period=_PERIODS[name],
            timestamp=observation_date,
            source_type=SourceType.OFFICIAL_API,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
        )
