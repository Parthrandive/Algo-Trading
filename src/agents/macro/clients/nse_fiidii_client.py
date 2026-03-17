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

import csv
import logging
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
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

_FINANCIAL_YEAR_RE = re.compile(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$")
_FII_TURNOVER_NET_COLUMN = "All India Equity - Net"
_DEFAULT_FII_TURNOVER_CSV_CANDIDATES = (
    Path("data/macro/FII Turnover.csv"),
    Path("data/macro/FII_Turnover.csv"),
    Path.home() / "Downloads" / "FII Turnover.csv",
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
        fii_turnover_csv_path: str | Path | None = None,
    ) -> None:
        self._raw_fetcher = raw_fetcher
        self._parser = FIIDIIParser()
        self._fii_turnover_csv_path = (
            Path(fii_turnover_csv_path).expanduser() if fii_turnover_csv_path else None
        )

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

        used_fallback = False
        parsed: list[MacroIndicator] = []
        if self._raw_fetcher is not None:
            payload = self._raw_fetcher(date_range)
        else:
            from src.agents.macro.clients.nse_fiidii_scraper import fetch_real_fii_dii
            try:
                logger.info("Calling real NSE FII/DII scraper...")
                payload = fetch_real_fii_dii(date_range)
            except Exception as exc:
                logger.warning(
                    "NSE FII/DII fetch returned no usable data for range %s (%s).",
                    date_range,
                    exc,
                )
                payload = None

        if payload is not None:
            self._parser.source_type = (
                SourceType.FALLBACK_SCRAPER if used_fallback else SourceType.OFFICIAL_API
            )
            parsed = list(self._parser.parse(payload))
            if used_fallback:
                parsed = [
                    record.model_copy(update={"quality_status": QualityFlag.WARN})
                    for record in parsed
                ]

        records = [record for record in parsed if record.indicator_name == name]
        if name == MacroIndicatorType.FII_FLOW:
            annual_records = self._load_annual_fii_records(date_range)
            if annual_records:
                by_timestamp = {record.timestamp: record for record in annual_records}
                for record in records:
                    # Prefer higher-frequency live record when timestamps collide.
                    by_timestamp[record.timestamp] = record
                records = [by_timestamp[k] for k in sorted(by_timestamp.keys())]

        if not records:
            logger.warning("No %s records produced from NSE/NSDL payload", name.value)
        return records

    @staticmethod
    def _parse_financial_year_end(raw_year: str) -> date | None:
        match = _FINANCIAL_YEAR_RE.match(str(raw_year or "").strip())
        if not match:
            return None
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if end_year != start_year + 1:
            return None
        return date(end_year, 3, 31)

    @staticmethod
    def _parse_float(raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        token = str(raw_value).strip()
        if not token or token == "-":
            return None
        try:
            return float(token.replace(",", ""))
        except ValueError:
            return None

    def _resolve_fii_turnover_csv_path(self) -> Path | None:
        if self._fii_turnover_csv_path and self._fii_turnover_csv_path.exists():
            return self._fii_turnover_csv_path

        env_path = os.getenv("FII_TURNOVER_CSV_PATH")
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                return candidate

        for candidate in _DEFAULT_FII_TURNOVER_CSV_CANDIDATES:
            if candidate.exists():
                return candidate

        return None

    def _load_annual_fii_records(self, date_range: DateRange) -> list[MacroIndicator]:
        csv_path = self._resolve_fii_turnover_csv_path()
        if csv_path is None:
            return []

        records: list[MacroIndicator] = []
        now_utc = datetime.now(UTC)

        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    year_token = str(row.get("Year", "")).strip()
                    observation_date = self._parse_financial_year_end(year_token)
                    if observation_date is None:
                        continue
                    if observation_date < date_range.start or observation_date > date_range.end:
                        continue

                    value = self._parse_float(row.get(_FII_TURNOVER_NET_COLUMN))
                    if value is None:
                        continue

                    records.append(
                        MacroIndicator(
                            indicator_name=MacroIndicatorType.FII_FLOW,
                            value=value,
                            unit="INR_Cr",
                            period="Annual",
                            timestamp=datetime(
                                observation_date.year,
                                observation_date.month,
                                observation_date.day,
                                tzinfo=UTC,
                            ),
                            source_type=SourceType.MANUAL_OVERRIDE,
                            ingestion_timestamp_utc=now_utc,
                            ingestion_timestamp_ist=now_utc.astimezone(IST),
                            schema_version="1.1",
                            quality_status=QualityFlag.WARN,
                        )
                    )
        except Exception as exc:
            logger.warning("Failed to parse annual FII turnover CSV at %s (%s).", csv_path, exc)
            return []

        records.sort(key=lambda item: item.timestamp)
        if records:
            logger.info(
                "Loaded %d annual FII fallback record(s) from %s for %s -> %s",
                len(records),
                csv_path,
                date_range.start,
                date_range.end,
            )
        return records

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
