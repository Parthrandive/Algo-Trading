"""
RBIClient — Reserve Bank of India macro data client.

Covers:
  FX_RESERVES    — Weekly, USD_Bn   (RBI Weekly Statistical Supplement)
  RBI_BULLETIN   — Irregular, count (publication event marker, value=1.0)
  INDIA_10Y      — Daily, %         (RBI government securities yield)

Source URLs (from runtime config):
  FX_RESERVES    : https://www.rbi.org.in/scripts/WSSView.aspx?Id=1
  RBI_BULLETIN   : https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx
  INDIA_10Y      : https://www.rbi.org.in/scripts/Statistics.aspx

Uses parser-driven normalization with injectable fetchers.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.agents.macro.client import DateRange
from src.agents.macro.parsers import FXReservesParser, RBIBulletinParser
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
    Concrete client for RBI data (FX Reserves, RBI Bulletins, India 10Y).

    Satisfies ``MacroClientInterface`` (Protocol, duck-typed).
    """

    def __init__(
        self,
        raw_fetchers: dict[MacroIndicatorType, Callable[[DateRange], dict[str, Any]]] | None = None,
    ) -> None:
        self._raw_fetchers = raw_fetchers or {}
        self._parsers = {
            MacroIndicatorType.FX_RESERVES: FXReservesParser(),
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

        Returns schema-normalized records for supported RBI indicators.
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

        payload, used_fallback = self._get_payload(name, date_range)

        if name in self._parsers:
            parser = self._parsers[name]
            parser.source_type = (
                SourceType.FALLBACK_SCRAPER if used_fallback else SourceType.OFFICIAL_API
            )
            records = list(parser.parse(payload))
        else:
            # INDIA_10Y is a direct value series (%); mapped inline to MacroIndicator.
            source_type = SourceType.FALLBACK_SCRAPER if used_fallback else SourceType.OFFICIAL_API
            quality_status = QualityFlag.WARN if used_fallback else QualityFlag.PASS
            records = [self._build_india_10y_record(payload, source_type, quality_status)]

        if used_fallback:
            records = [
                record.model_copy(update={"quality_status": QualityFlag.WARN})
                for record in records
            ]
        return records

    def _get_payload(
        self,
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> Tuple[dict[str, Any], bool]:
        if name in self._raw_fetchers:
            return self._raw_fetchers[name](date_range), False

        if name == MacroIndicatorType.RBI_BULLETIN:
            from src.agents.macro.clients.rbi_bulletin_scraper import fetch_real_rbi_bulletin

            logger.info("Calling real RBI Bulletin scraper...")
            try:
                return fetch_real_rbi_bulletin(date_range), False
            except Exception as exc:
                logger.warning(
                    "RBI Bulletin fetch produced no usable rows for range %s (%s).",
                    date_range,
                    exc,
                )
                return {"publications": []}, True

        return self._build_simulated_payload(name, date_range), True

    @staticmethod
    def _build_simulated_payload(
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> dict[str, Any]:
        if name == MacroIndicatorType.RBI_BULLETIN:
            return {
                "publications": [
                    {
                        "date": date_range.end.isoformat(),
                        "title": "RBI Bulletin (simulated)",
                    }
                ]
            }
        if name == MacroIndicatorType.FX_RESERVES:
            day_seed = date_range.end.toordinal() % 20
            return {
                "date": date_range.end.isoformat(),
                "value": round(620.0 + day_seed * 1.7, 2),
            }
        if name == MacroIndicatorType.INDIA_10Y:
            day_seed = date_range.end.toordinal()
            return {
                "date": date_range.end.isoformat(),
                "value": round(6.8 + (day_seed % 9) * 0.03, 3),
            }
        raise ValueError(f"Unsupported RBI payload request: {name.value}")

    @staticmethod
    def _build_india_10y_record(
        raw_data: dict[str, Any],
        source_type: SourceType = SourceType.OFFICIAL_API,
        quality_status: QualityFlag = QualityFlag.PASS,
    ) -> MacroIndicator:
        raw_date = raw_data.get("date")
        if not isinstance(raw_date, str) or not raw_date.strip():
            raise ValueError("Missing/invalid 'date' for INDIA_10Y payload")
        raw_value = raw_data.get("value")
        if raw_value is None:
            raise ValueError("Missing 'value' for INDIA_10Y payload")

        ts = datetime.fromisoformat(raw_date).replace(tzinfo=UTC)
        now_utc = datetime.now(UTC)
        return MacroIndicator(
            indicator_name=MacroIndicatorType.INDIA_10Y,
            value=float(raw_value),
            unit="%",
            period="Daily",
            timestamp=ts,
            source_type=source_type,
            ingestion_timestamp_utc=now_utc,
            ingestion_timestamp_ist=now_utc.astimezone(IST),
            schema_version="1.1",
            quality_status=quality_status,
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
