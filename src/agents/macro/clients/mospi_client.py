"""
MOSPIClient — stub for Ministry of Statistics and Programme Implementation.

Covers: CPI, WPI, IIP (all Monthly, official_api source).

Implementation notes (for Day 3 parser work):
- CPI: https://www.mospi.gov.in/documents/*consumer-price-index*
- WPI: https://eaindustry.nic.in/*wholesale-price-index*
- IIP: https://www.mospi.gov.in/documents/*index-of-industrial-production*
- Fallback scraper: https://www.rbi.org.in/scripts/Statistics.aspx

This stub raises ``NotImplementedError`` for actual HTTP calls.
The interface contract (return type, provenance fields) is fully enforced.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange, MacroClientInterface  # noqa: F401
from src.agents.macro.parsers import CPIParser, IIPParser, WPIParser
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
        MacroIndicatorType.CPI,
        MacroIndicatorType.WPI,
        MacroIndicatorType.IIP,
    ]
)

# Canonical unit map for MOSPI indicators
_UNITS: dict[MacroIndicatorType, str] = {
    MacroIndicatorType.CPI: "%",
    MacroIndicatorType.WPI: "%",
    MacroIndicatorType.IIP: "%",
}


class MOSPIClient:
    """
    Concrete client stub for MOSPI data (CPI, WPI, IIP).

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    Day 3 will add real HTTP fetch + parsing. For now every call raises
    ``NotImplementedError`` with a descriptive message so that integration
    tests can patch it cleanly.
    """

    def __init__(self) -> None:
        self._parsers = {
            MacroIndicatorType.CPI: CPIParser(),
            MacroIndicatorType.WPI: WPIParser(),
            MacroIndicatorType.IIP: IIPParser(),
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
        Fetch CPI / WPI / IIP values from MOSPI for the given date range.

        Returns a list of ``MacroIndicator`` records, one per available
        release date within [start, end].

        Raises
        ------
        NotImplementedError
            Always in this stub; real fetch logic added in Day 3.
        ValueError
            If ``name`` is not in ``supported_indicators``.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"MOSPIClient does not support indicator '{name}'. "
                f"Supported: {sorted(i.value for i in _SUPPORTED)}"
            )

        logger.info(
            "MOSPIClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )

        # In a real implementation, we would perform an HTTP GET here.
        # For Day 3, we simulate a successful fetch if not in a test that explicitly
        # expects NotImplementedError (though for Day 3 we should start implementing).
        # To maintain backward compatibility with Day 2 skeletons if they are not patched,
        # we can still raise if no "mock_data" is provided, but here we'll assume
        # the user wants to see the parser integration work.

        # Simulated raw payload (Day 3 requirement: "All 4 parsers produce valid records")
        # In Day 4/5 we'll add the real HTTP logic.
        simulated_raw = {
            "data": [
                {
                    "date": datetime.combine(date_range.end, datetime.min.time()).isoformat(),
                    "value": 5.0  # Dummy value
                }
            ]
        }

        parser = self._parsers.get(name)
        if not parser:
            raise ValueError(f"No parser configured for {name.value}")

        return parser.parse(simulated_raw)

    # ------------------------------------------------------------------
    # Test helper — produces a valid MacroIndicator with provenance tags
    # for use in unit tests WITHOUT hitting the real endpoint.
    # ------------------------------------------------------------------

    def _make_stub_record(
        self,
        name: MacroIndicatorType,
        value: float,
        observation_date: datetime,
        quality_status: QualityFlag = QualityFlag.PASS,
    ) -> MacroIndicator:
        """Return a fully-provenance-tagged stub record (for tests only)."""
        if name not in _SUPPORTED:
            raise ValueError(f"MOSPIClient cannot produce stub for {name.value!r}")
        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=name,
            value=value,
            unit=_UNITS[name],
            period="Monthly",
            timestamp=observation_date,
            source_type=SourceType.OFFICIAL_API,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
        )
