"""
Parsers for India macro indicators and RBI Bulletins.

Normalizes raw payloads into the MacroIndicator schema v1.1.
Includes data quality checks for missingness, outliers, and latency.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    QualityFlag,
    SourceType,
)

logger = logging.getLogger(__name__)

# Outlier bounds for quality checks
_BOUNDS = {
    MacroIndicatorType.CPI: (-5.0, 15.0),  # %
    MacroIndicatorType.WPI: (-10.0, 20.0), # %
    MacroIndicatorType.IIP: (-20.0, 20.0), # %
}

# Latency SLAs in hours
_SLAS = {
    MacroIndicatorType.CPI: 48,
    MacroIndicatorType.WPI: 48,
    MacroIndicatorType.IIP: 48,
    MacroIndicatorType.RBI_BULLETIN: 24,
}

def check_quality(
    name: MacroIndicatorType,
    value: float,
    observation_date: datetime,
    ingestion_date: datetime,
) -> QualityFlag:
    """
    Perform basic data quality checks.
    
    Checks:
    1. Outliers: Is the value within reasonable bounds?
    2. Latency: Is the ingestion date too far from the observation date?
    """
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
        latency = ingestion_date - observation_date
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
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        
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

    def parse(self, raw_data: dict[str, Any]) -> Sequence[MacroIndicator]:
        """
        RBI Bulletins are event markers. Value is always 1.0.
        Expected schema:
        {
            "publications": [
                {"date": "2026-02-15", "title": "RBI Bulletin February 2026"}
            ]
        }
        """
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        results = []
        now = datetime.now(UTC)
        for pub in raw_data.get("publications", []):
            try:
                dt = datetime.fromisoformat(pub["date"]).replace(tzinfo=UTC)
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
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
        results = []
        now = datetime.now(UTC)
        
        try:
            dt_str = raw_data.get("date")
            if not dt_str:
                raise ValueError("Missing date")
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
            
            if "fii_flow" in raw_data:
                fii_val = float(raw_data["fii_flow"])
                quality_fii = check_quality(MacroIndicatorType.FII_FLOW, fii_val, dt, now)
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
                
            if "dii_flow" in raw_data:
                dii_val = float(raw_data["dii_flow"])
                quality_dii = check_quality(MacroIndicatorType.DII_FLOW, dii_val, dt, now)
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
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
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
        from zoneinfo import ZoneInfo
        IST = ZoneInfo("Asia/Kolkata")
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
