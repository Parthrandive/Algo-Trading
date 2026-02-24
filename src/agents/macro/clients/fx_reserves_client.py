"""
FXReservesClient — stub for RBI Foreign Exchange Reserves data.

Covers:
  FX_RESERVES  — Weekly, USD_Bn

Source (from runtime config):
  Primary : https://www.rbi.org.in/scripts/WSSView.aspx?Id=1
            (RBI Weekly Statistical Supplement, released Fridays)

Freshness SLA: 24 hours after Friday release (friday_release_plus_24h).

Note: Although FX Reserves data is published by RBI, this is kept as a
separate client stub (per Day 2 backlog) because its fetch + parse logic
is distinct from the RBIClient (which covers Bulletins and the India 10Y
yield from the Statistics page).
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
    [MacroIndicatorType.FX_RESERVES]
)


class FXReservesClient:
    """
    Concrete client stub for RBI FX Reserves (Weekly Statistical Supplement).

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    Release cadence: every Friday, covers the week ending that Friday.
    Day 4 will add the real ``FXReservesParser`` and HTTP fetch.
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
        Fetch FX Reserves records for the given date range.

        Raises
        ------
        NotImplementedError
            Always in this stub.
        ValueError
            If ``name`` is not ``FX_RESERVES``.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"FXReservesClient only supports FX_RESERVES, got '{name}'."
            )

        logger.info(
            "FXReservesClient.get_indicator called: indicator=%s range=%s — stub",
            name.value,
            date_range,
        )
        raise NotImplementedError(
            "FXReservesClient.get_indicator — "
            "RBI WSS fetch + FXReservesParser not implemented until Day 4."
        )

    def _make_stub_record(
        self,
        value_usd_bn: float,
        observation_date: datetime,
        quality_status: QualityFlag = QualityFlag.PASS,
    ) -> MacroIndicator:
        """Return a fully-provenance-tagged stub record (for tests only)."""
        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=MacroIndicatorType.FX_RESERVES,
            value=value_usd_bn,
            unit="USD_Bn",
            period="Weekly",
            timestamp=observation_date,
            source_type=SourceType.OFFICIAL_API,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
        )
