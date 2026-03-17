"""
AkShareClient — akshare-backed macro client for India/US proxy indicators.

Covers:
  REPO_RATE  — RBI policy rate decisions (Monthly, %)
  US_10Y     — US 10Y yield proxy from akshare (Daily, %)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.agents.macro.client import DateRange
from src.agents.macro.parsers import check_quality
from src.schemas.macro_data import (
    MacroIndicator,
    MacroIndicatorType,
    SourceType,
)

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_SUPPORTED: frozenset[MacroIndicatorType] = frozenset(
    [MacroIndicatorType.REPO_RATE, MacroIndicatorType.US_10Y]
)

_UNITS: dict[MacroIndicatorType, str] = {
    MacroIndicatorType.REPO_RATE: "%",
    MacroIndicatorType.US_10Y: "%",
}

_PERIODS: dict[MacroIndicatorType, str] = {
    MacroIndicatorType.REPO_RATE: "Monthly",
    MacroIndicatorType.US_10Y: "Daily",
}

_REPO_RATE_VALUE_COLUMNS = ("今值", "现值", "value", "Value", "最新值")
_US10Y_VALUE_COLUMNS = (
    "美国国债收益率10年",
    "美国10年期国债收益率",
    "美国10Y",
    "US10Y",
    "10Y",
    "yield_10y",
)
_DATE_COLUMNS = ("日期", "date", "Date", "DATE", "时间", "发布时间")


class AkShareClient:
    """
    Concrete client for akshare macro endpoints.

    Optional `raw_fetchers` injection keeps unit tests offline and deterministic.
    """

    def __init__(
        self,
        raw_fetchers: dict[MacroIndicatorType, Callable[[DateRange], pd.DataFrame]] | None = None,
    ) -> None:
        self._raw_fetchers = raw_fetchers or {}

    @property
    def supported_indicators(self) -> frozenset[MacroIndicatorType]:
        return _SUPPORTED

    def get_indicator(
        self,
        name: MacroIndicatorType,
        date_range: DateRange,
    ) -> Sequence[MacroIndicator]:
        if name not in _SUPPORTED:
            raise ValueError(
                f"AkShareClient does not support indicator '{name}'. "
                f"Supported: {sorted(i.value for i in _SUPPORTED)}"
            )

        logger.info(
            "AkShareClient.get_indicator called: indicator=%s range=%s",
            name.value,
            date_range,
        )

        frame = self._fetch_dataframe(name, date_range)
        if frame.empty:
            return []

        normalized = self._normalize_dataframe(name, frame, date_range)
        now_utc = datetime.now(UTC)
        records: list[MacroIndicator] = []
        for _, row in normalized.iterrows():
            ts = row["timestamp"]
            val = float(row["value"])
            quality = check_quality(name, val, ts, now_utc)
            records.append(
                MacroIndicator(
                    indicator_name=name,
                    value=val,
                    unit=_UNITS[name],
                    period=_PERIODS[name],
                    timestamp=ts,
                    source_type=SourceType.OFFICIAL_API,
                    ingestion_timestamp_utc=now_utc,
                    ingestion_timestamp_ist=now_utc.astimezone(IST),
                    schema_version="1.1",
                    quality_status=quality,
                )
            )
        return records

    def _fetch_dataframe(self, name: MacroIndicatorType, date_range: DateRange) -> pd.DataFrame:
        if name in self._raw_fetchers:
            return self._raw_fetchers[name](date_range).copy()

        ak = self._import_akshare()
        try:
            if name == MacroIndicatorType.REPO_RATE:
                frame = ak.macro_bank_india_interest_rate()
            else:
                # Some akshare versions accept period as positional/keyword, others do not.
                try:
                    frame = ak.bond_zh_us_rate(period="10年")
                except TypeError:
                    frame = ak.bond_zh_us_rate()
        except Exception as exc:
            raise RuntimeError(f"akshare fetch failed for {name.value}: {exc}") from exc

        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(f"akshare returned non-DataFrame payload for {name.value}")
        return frame.copy()

    @staticmethod
    def _import_akshare():
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "akshare is required for AkShareClient. Install with `pip install akshare>=1.18.0`."
            ) from exc
        return ak

    def _normalize_dataframe(
        self,
        name: MacroIndicatorType,
        frame: pd.DataFrame,
        date_range: DateRange,
    ) -> pd.DataFrame:
        date_col = self._resolve_date_column(frame)
        value_col = self._resolve_value_column(name, frame, date_col)

        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(frame[date_col], errors="coerce", utc=True),
                "value": self._coerce_numeric(frame[value_col]),
            }
        ).dropna(subset=["timestamp", "value"])

        # Inclusive date filter.
        out = out[
            (out["timestamp"].dt.date >= date_range.start)
            & (out["timestamp"].dt.date <= date_range.end)
        ]

        out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return out.reset_index(drop=True)

    @staticmethod
    def _resolve_date_column(frame: pd.DataFrame) -> str:
        for candidate in _DATE_COLUMNS:
            if candidate in frame.columns:
                return candidate

        for col in frame.columns:
            if "date" in str(col).lower():
                return str(col)
        raise ValueError("Unable to locate date column in akshare payload")

    def _resolve_value_column(
        self,
        name: MacroIndicatorType,
        frame: pd.DataFrame,
        date_col: str,
    ) -> str:
        candidates = _REPO_RATE_VALUE_COLUMNS if name == MacroIndicatorType.REPO_RATE else _US10Y_VALUE_COLUMNS
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate

        if name == MacroIndicatorType.US_10Y:
            us10y_heuristic = [
                col
                for col in frame.columns
                if col != date_col and ("10" in str(col)) and ("美" in str(col) or "us" in str(col).lower())
            ]
            if us10y_heuristic:
                return str(us10y_heuristic[0])

        for col in frame.columns:
            if col == date_col:
                continue
            numeric = self._coerce_numeric(frame[col])
            if numeric.notna().any():
                return str(col)
        raise ValueError(f"Unable to locate numeric value column for {name.value}")

    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        # Handles values like "6.50%" or "6,50" by stripping formatting noise.
        cleaned = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce")
