"""
FXReservesClient — RBI Foreign Exchange Reserves client.

Covers:
  FX_RESERVES  — Weekly, USD_Bn

Source (from runtime config):
  Primary : https://www.rbi.org.in/scripts/WSSView.aspx?Id=1
            (RBI Weekly Statistical Supplement, released Fridays)

Freshness SLA: 24 hours after Friday release (friday_release_plus_24h).

Note: Although FX Reserves data is published by RBI, this is kept as a
separate client (per Day 2 backlog) because its fetch + parse logic
is distinct from the RBIClient (which covers Bulletins and the India 10Y
yield from the Statistics page).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
from src.agents.macro.clients.fred_client import FredSeriesSpec, fetch_series_history
from src.agents.macro.parsers import FXReservesParser
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

_FX_FRED_FALLBACK = FredSeriesSpec(
    series_id="TRESEGINM052N",
    frequency="Monthly",
    source_label="fred:TRESEGINM052N",
    # FRED IMF series is in USD million; convert to USD billion for schema unit.
    unit_scale=0.001,
)


class FXReservesClient:
    """
    Concrete client for RBI FX Reserves (Weekly Statistical Supplement).

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).

    Release cadence: every Friday, covers the week ending that Friday.
    Uses ``FXReservesParser`` for normalized output.
    A custom ``raw_fetcher`` can be injected for live/contract tests.
    """

    def __init__(
        self,
        raw_fetcher: Callable[[DateRange], dict[str, Any]] | None = None,
    ) -> None:
        self._raw_fetcher = raw_fetcher
        self._parser = FXReservesParser()

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

        Returns schema-normalized FX reserves records for the requested range.
        """
        if name not in _SUPPORTED:
            raise ValueError(
                f"FXReservesClient only supports FX_RESERVES, got '{name}'."
            )

        logger.info(
            "FXReservesClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )
        if self._raw_fetcher is not None:
            payload = self._raw_fetcher(date_range)
            used_fallback = False
        else:
            payload, used_fallback = self._try_real_then_historical_fallback(date_range)

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
    def _try_real_then_historical_fallback(date_range: DateRange) -> tuple[dict[str, Any], bool]:
        """Try official RBI latest scrape; fall back to deep FRED historical series."""
        from src.agents.macro.clients.rbi_fx_reserves_scraper import fetch_real_fx_reserves

        rows: list[dict[str, Any]] = []
        used_fallback = False

        try:
            logger.info("Calling real RBI FX Reserves scraper...")
            latest = fetch_real_fx_reserves(date_range)
            rows.append(
                {
                    "date": latest["date"],
                    "value": latest["value"],
                }
            )
        except Exception as e:
            logger.warning(
                "RBI FX Reserves latest scrape failed (%s). Falling back to FRED historical only.",
                e,
            )
            used_fallback = True

        try:
            fallback_rows = fetch_series_history(_FX_FRED_FALLBACK, date_range)
            for row in fallback_rows:
                rows.append({"date": row["date"], "value": row["value"]})
            if fallback_rows:
                used_fallback = True
        except Exception as exc:
            logger.error("FX fallback series fetch failed: %s", exc)

        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            dedup[row["date"]] = row
        ordered = [dedup[key] for key in sorted(dedup.keys())]
        return {"data": ordered}, used_fallback

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
