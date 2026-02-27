"""
Parsers for India macro indicators and RBI Bulletins.

Normalizes raw payloads into the MacroIndicator schema v1.1.
Includes data quality checks for missingness, outliers, and latency.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    QualityFlag,
    SourceType,
)

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Outlier bounds for quality checks
_BOUNDS = {
    MacroIndicatorType.CPI: (-5.0, 15.0),   # %
    MacroIndicatorType.WPI: (-10.0, 20.0),  # %
    MacroIndicatorType.IIP: (-20.0, 20.0),  # %
    MacroIndicatorType.FII_FLOW: (-100000.0, 100000.0),  # INR_Cr
    MacroIndicatorType.DII_FLOW: (-100000.0, 100000.0),  # INR_Cr
    MacroIndicatorType.FX_RESERVES: (0.0, 1500.0),  # USD_Bn
    MacroIndicatorType.INDIA_US_10Y_SPREAD: (-1000.0, 2000.0),  # bps
}

# Latency SLAs in hours
_SLAS = {
    MacroIndicatorType.CPI: 48,
    MacroIndicatorType.WPI: 48,
    MacroIndicatorType.IIP: 48,
    MacroIndicatorType.RBI_BULLETIN: 24,
    MacroIndicatorType.FII_FLOW: 4,
    MacroIndicatorType.DII_FLOW: 4,
    MacroIndicatorType.FX_RESERVES: 24,
    MacroIndicatorType.INDIA_US_10Y_SPREAD: 6,
}

_DAILY_RELEASE_AT_EOD = frozenset(
    {
        MacroIndicatorType.FII_FLOW,
        MacroIndicatorType.DII_FLOW,
        MacroIndicatorType.INDIA_US_10Y_SPREAD,
    }
)

_WEEKLY_RELEASE_AT_EOD = frozenset({MacroIndicatorType.FX_RESERVES})


def _latency_anchor(name: MacroIndicatorType, observation_date: datetime) -> datetime:
    """Infer release anchor from observation date before SLA lag is applied."""
    if name in _DAILY_RELEASE_AT_EOD or name in _WEEKLY_RELEASE_AT_EOD:
        return observation_date.replace(hour=23, minute=59, second=59, microsecond=0)
    return observation_date


def _parse_date_token(date_token: str) -> datetime | None:
    date_token = date_token.strip()
    iso_candidate = date_token.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass

    for fmt in ("%d %B %Y", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(date_token, fmt).replace(tzinfo=UTC)
            return parsed
        except ValueError:
            continue
    return None


def _extract_dates_from_text(text: str) -> list[datetime]:
    """Extract publication dates from HTML/PDF text blobs."""
    patterns = (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
    )
    seen: set[datetime] = set()
    dates: list[datetime] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            parsed = _parse_date_token(match.group(0))
            if parsed is None:
                continue
            key = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            if key in seen:
                continue
            seen.add(key)
            dates.append(key)
    return dates


def check_quality(
    name: MacroIndicatorType,
    value: float,
    observation_date: datetime,
    ingestion_date: datetime,
    *,
    missing_required_fields: bool = False,
) -> QualityFlag:
    """
    Perform basic data quality checks.
    
    Checks:
    1. Outliers: Is the value within reasonable bounds?
    2. Latency: Is the ingestion date too far from the observation date?
    3. Missingness: Are required sibling fields missing from the raw payload?
    """
    if missing_required_fields:
        logger.warning("Missingness detected for %s payload", name.value)
        return QualityFlag.WARN

    # Outlier check
    if name in _BOUNDS:
        low, high = _BOUNDS[name]
        if not (low <= value <= high):
            logger.warning(
                "Outlier detected for %s: value=%s (bounds: [%s, %s])",
                name.value, value, low, high
            )
            return QualityFlag.WARN

    # Latency check
    if name in _SLAS:
        sla_hours = _SLAS[name]
        latency = ingestion_date - _latency_anchor(name, observation_date)
        if latency > timedelta(hours=sla_hours):
            logger.warning(
                "Latency SLA breached for %s: latency=%s (SLA: %sh)",
                name.value, latency, sla_hours
            )
            return QualityFlag.WARN

    return QualityFlag.PASS

class BaseParser:
    """Base class for all macro parsers."""
    
    indicator_type: MacroIndicatorType
    unit: str
    period: str
    source_type: SourceType = SourceType.OFFICIAL_API

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """Parse raw dictionary into MacroIndicator records."""
        raise NotImplementedError("Subclasses must implement parse()")

class MOSPIParser(BaseParser):
    """Base for MOSPI-style indicator parsers (CPI, WPI, IIP)."""
    
    period = "Monthly"

    def _create_record(
        self, 
        value: float, 
        observation_date: datetime,
        now: datetime,
    ) -> MacroIndicator:
        quality = check_quality(self.indicator_type, value, observation_date, now)
        
        return MacroIndicator(
            indicator_name=self.indicator_type,
            value=value,
            unit=self.unit,
            period=self.period,
            timestamp=observation_date,
            source_type=self.source_type,
            ingestion_timestamp_utc=now,
            ingestion_timestamp_ist=now.astimezone(IST),
            schema_version="1.1",
            quality_status=quality,
        )

class CPIParser(MOSPIParser):
    indicator_type = MacroIndicatorType.CPI
    unit = "%"

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        Expected schema (example):
        {
            "data": [
                {"date": "2026-01-01", "value": 5.1}
            ]
        }
        """
        results = []
        now = datetime.now(UTC)
        for item in raw_data.get("data", []):
            try:
                val = float(item["value"])
                dt = datetime.fromisoformat(item["date"]).replace(tzinfo=UTC)
                results.append(self._create_record(val, dt, now))
            except (KeyError, ValueError, TypeError) as e:
                logger.error("Failed to parse CPI item %s: %s", item, e)
        return results

class WPIParser(MOSPIParser):
    indicator_type = MacroIndicatorType.WPI
    unit = "%"

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        results = []
        now = datetime.now(UTC)
        for item in raw_data.get("data", []):
            try:
                val = float(item["value"])
                dt = datetime.fromisoformat(item["date"]).replace(tzinfo=UTC)
                results.append(self._create_record(val, dt, now))
            except (KeyError, ValueError, TypeError) as e:
                logger.error("Failed to parse WPI item %s: %s", item, e)
        return results

class IIPParser(MOSPIParser):
    indicator_type = MacroIndicatorType.IIP
    unit = "%"

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        results = []
        now = datetime.now(UTC)
        for item in raw_data.get("data", []):
            try:
                val = float(item["value"])
                dt = datetime.fromisoformat(item["date"]).replace(tzinfo=UTC)
                results.append(self._create_record(val, dt, now))
            except (KeyError, ValueError, TypeError) as e:
                logger.error("Failed to parse IIP item %s: %s", item, e)
        return results

class RBIBulletinParser(BaseParser):
    indicator_type = MacroIndicatorType.RBI_BULLETIN
    unit = "count"
    period = "Irregular"

    @staticmethod
    def _extract_publication_entries(raw_data: dict[str, Any]) -> list[dict[str, str]]:
        publications = raw_data.get("publications")
        if isinstance(publications, list):
            result: list[dict[str, str]] = []
            for item in publications:
                if not isinstance(item, dict):
                    continue
                date_token = item.get("date")
                if not isinstance(date_token, str):
                    continue
                title = item.get("title", "RBI Bulletin")
                result.append({"date": date_token, "title": str(title)})
            return result

        text_chunks: list[str] = []
        for key in ("html", "pdf_text", "text"):
            value = raw_data.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(value)
        pdf_bytes = raw_data.get("pdf_bytes")
        if isinstance(pdf_bytes, (bytes, bytearray)):
            decoded = bytes(pdf_bytes).decode("utf-8", errors="ignore").strip()
            if decoded:
                text_chunks.append(decoded)

        extracted_dates: list[datetime] = []
        for chunk in text_chunks:
            extracted_dates.extend(_extract_dates_from_text(chunk))

        return [
            {"date": dt.date().isoformat(), "title": "RBI Bulletin (derived)"}
            for dt in extracted_dates
        ]

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        RBI Bulletins are event markers. Value is always 1.0.
        Supports either:
        1. Structured schema:
        {
            "publications": [
                {"date": "2026-02-15", "title": "RBI Bulletin February 2026"}
            ]
        }
        2. Raw HTML/PDF text blobs:
        {
            "html": "... RBI Bulletin ... 15 February 2026 ..."
        }
        """
        results = []
        now = datetime.now(UTC)
        publications = self._extract_publication_entries(raw_data)
        for pub in publications:
            try:
                parsed = _parse_date_token(pub["date"])
                if parsed is None:
                    raise ValueError(f"Unsupported date format: {pub['date']!r}")
                dt = parsed
                quality = check_quality(self.indicator_type, 1.0, dt, now)
                
                results.append(MacroIndicator(
                    indicator_name=self.indicator_type,
                    value=1.0,
                    unit=self.unit,
                    period=self.period,
                    timestamp=dt,
                    source_type=self.source_type,
                    ingestion_timestamp_utc=now,
                    ingestion_timestamp_ist=now.astimezone(IST),
                    schema_version="1.1",
                    quality_status=quality,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.error("Failed to parse RBI Bulletin item %s: %s", pub, e)
        return results

class FIIDIIParser(BaseParser):
    unit = "INR_Cr"
    period = "Daily"
    
    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        Expected schema:
        {
            "date": "2026-02-26",
            "fii_flow": -1500.5,
            "dii_flow": 2000.0
        }
        """
        results = []
        now = datetime.now(UTC)
        
        try:
            dt_str = raw_data.get("date")
            if not dt_str:
                raise ValueError("Missing date")
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=UTC)

            has_fii = "fii_flow" in raw_data
            has_dii = "dii_flow" in raw_data
            if not has_fii and not has_dii:
                raise ValueError("At least one of fii_flow or dii_flow is required")

            missing_pair = not (has_fii and has_dii)

            if has_fii:
                fii_val = float(raw_data["fii_flow"])
                quality_fii = check_quality(
                    MacroIndicatorType.FII_FLOW,
                    fii_val,
                    dt,
                    now,
                    missing_required_fields=missing_pair,
                )
                results.append(MacroIndicator(
                    indicator_name=MacroIndicatorType.FII_FLOW,
                    value=fii_val,
                    unit=self.unit,
                    period=self.period,
                    timestamp=dt,
                    source_type=self.source_type,
                    ingestion_timestamp_utc=now,
                    ingestion_timestamp_ist=now.astimezone(IST),
                    schema_version="1.1",
                    quality_status=quality_fii,
                ))
                
            if has_dii:
                dii_val = float(raw_data["dii_flow"])
                quality_dii = check_quality(
                    MacroIndicatorType.DII_FLOW,
                    dii_val,
                    dt,
                    now,
                    missing_required_fields=missing_pair,
                )
                results.append(MacroIndicator(
                    indicator_name=MacroIndicatorType.DII_FLOW,
                    value=dii_val,
                    unit=self.unit,
                    period=self.period,
                    timestamp=dt,
                    source_type=self.source_type,
                    ingestion_timestamp_utc=now,
                    ingestion_timestamp_ist=now.astimezone(IST),
                    schema_version="1.1",
                    quality_status=quality_dii,
                ))
                
        except (ValueError, TypeError) as e:
            logger.error("Failed to parse FIIDII data %s: %s", raw_data, e)
            
        return results

class FXReservesParser(BaseParser):
    indicator_type = MacroIndicatorType.FX_RESERVES
    unit = "USD_Bn"
    period = "Weekly"
    
    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        Expected schema:
        {
            "date": "2026-02-20",
            "value": 680.5
        }
        """
        results = []
        now = datetime.now(UTC)
        
        try:
            dt_str = raw_data.get("date")
            if not dt_str:
                raise ValueError("Missing date")
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
            
            val = float(raw_data["value"])
            quality = check_quality(self.indicator_type, val, dt, now)
            
            results.append(MacroIndicator(
                indicator_name=self.indicator_type,
                value=val,
                unit=self.unit,
                period=self.period,
                timestamp=dt,
                source_type=self.source_type,
                ingestion_timestamp_utc=now,
                ingestion_timestamp_ist=now.astimezone(IST),
                schema_version="1.1",
                quality_status=quality,
            ))
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse FX Reserves data %s: %s", raw_data, e)
            
        return results

class BondSpreadParser(BaseParser):
    indicator_type = MacroIndicatorType.INDIA_US_10Y_SPREAD
    unit = "bps"
    period = "Daily"
    
    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        Expected schema:
        {
            "date": "2026-02-26",
            "india_10y_percent": 7.10,
            "us_10y_percent": 4.25
        }
        """
        results = []
        now = datetime.now(UTC)
        
        try:
            dt_str = raw_data.get("date")
            if not dt_str:
                raise ValueError("Missing date")
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
            
            ind_val = float(raw_data["india_10y_percent"])
            us_val = float(raw_data["us_10y_percent"])
            
            spread_bps = round((ind_val - us_val) * 100, 4)
            quality = check_quality(self.indicator_type, spread_bps, dt, now)
            
            results.append(MacroIndicator(
                indicator_name=self.indicator_type,
                value=spread_bps,
                unit=self.unit,
                period=self.period,
                timestamp=dt,
                source_type=self.source_type,
                ingestion_timestamp_utc=now,
                ingestion_timestamp_ist=now.astimezone(IST),
                schema_version="1.1",
                quality_status=quality,
            ))
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse Bond Spread data %s: %s", raw_data, e)
            
        return results
