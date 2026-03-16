from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.agents.sentinel.market_utils import (
    ASSET_TYPE_EQUITY,
    expected_session_timestamps,
    infer_asset_type,
    interval_to_timedelta,
    normalize_timestamp,
)


@dataclass(frozen=True)
class HistoricalQualityThresholds:
    min_rows_by_interval: dict[str, int] = field(
        default_factory=lambda: {
            "1h": 1200,
            "1d": 300,
        }
    )
    min_history_days: int = 180
    min_coverage_pct: float = 60.0
    max_zero_volume_ratio: float = 0.35

    def min_rows(self, interval: str) -> int:
        return int(self.min_rows_by_interval.get(interval, 300))


def _frame_for_quality(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "volume" not in result.columns:
        result["volume"] = 0
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce").fillna(0)
    return result


def _non_equity_expected_rows(frame: pd.DataFrame, interval: str) -> int:
    if frame.empty:
        return 0
    delta = interval_to_timedelta(interval)
    start = normalize_timestamp(frame["timestamp"].min().to_pydatetime())
    end = normalize_timestamp(frame["timestamp"].max().to_pydatetime())
    if end < start:
        return 0
    return int(((end - start) // delta) + 1)


def compute_symbol_quality(
    *,
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    session_rules,
    thresholds: HistoricalQualityThresholds,
    duplicate_count: int | None = None,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
) -> dict[str, Any]:
    raw = _frame_for_quality(frame)
    asset_type = infer_asset_type(symbol)

    if raw.empty:
        return {
            "symbol": symbol,
            "interval": interval,
            "asset_type": asset_type,
            "first_timestamp": None,
            "last_timestamp": None,
            "row_count": 0,
            "duplicate_count": int(duplicate_count or 0),
            "expected_rows": 0,
            "missing_intervals": 0,
            "gap_count": 0,
            "largest_gap_intervals": 0,
            "zero_volume_ratio": None,
            "coverage_pct": 0.0,
            "history_days": 0,
            "train_ready": False,
            "status": "failed",
            "quality_flags": ["no_rows"],
        }

    duplicates = (
        int(duplicate_count)
        if duplicate_count is not None
        else int(raw.duplicated(subset=["timestamp"]).sum())
    )
    deduped = raw.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    first_ts = normalize_timestamp(deduped["timestamp"].min().to_pydatetime())
    last_ts = normalize_timestamp(deduped["timestamp"].max().to_pydatetime())
    history_days = max(1, (last_ts.date() - first_ts.date()).days + 1)

    if requested_start is None:
        requested_start = first_ts
    if requested_end is None:
        requested_end = last_ts

    expected_timestamps = expected_session_timestamps(
        session_rules=session_rules,
        start=requested_start,
        end=requested_end,
        interval=interval,
        asset_type=asset_type,
    )
    expected_rows = len(expected_timestamps) if expected_timestamps else _non_equity_expected_rows(deduped, interval)
    missing_intervals = max(0, expected_rows - len(deduped))

    interval_delta = interval_to_timedelta(interval)
    diffs = deduped["timestamp"].diff().dropna()
    if diffs.empty:
        gap_steps = pd.Series(dtype="int64")
    else:
        gap_steps = ((diffs / interval_delta).round().astype("int64") - 1).clip(lower=0)
    gap_count = int((gap_steps > 0).sum())
    largest_gap = int(gap_steps.max()) if not gap_steps.empty else 0

    zero_volume_ratio = None
    if asset_type == ASSET_TYPE_EQUITY:
        zero_volume_ratio = float((deduped["volume"] <= 0).mean()) if len(deduped) else 0.0

    coverage_pct = 100.0 if expected_rows == 0 else min(100.0, (100.0 * len(deduped) / expected_rows))

    flags: list[str] = []
    if duplicates > 0:
        flags.append("duplicates_detected")
    if len(deduped) < thresholds.min_rows(interval):
        flags.append("below_min_rows")
    if history_days < thresholds.min_history_days:
        flags.append("below_min_history_days")
    if coverage_pct < thresholds.min_coverage_pct:
        flags.append("below_min_coverage")
    if zero_volume_ratio is not None and zero_volume_ratio > thresholds.max_zero_volume_ratio:
        flags.append("high_zero_volume_ratio")

    train_ready = not flags or flags == ["duplicates_detected"]
    status = "train_ready" if train_ready else "partial"
    if len(deduped) == 0:
        status = "failed"

    return {
        "symbol": symbol,
        "interval": interval,
        "asset_type": asset_type,
        "first_timestamp": first_ts.isoformat(),
        "last_timestamp": last_ts.isoformat(),
        "row_count": int(len(deduped)),
        "duplicate_count": duplicates,
        "expected_rows": int(expected_rows),
        "missing_intervals": int(missing_intervals),
        "gap_count": gap_count,
        "largest_gap_intervals": largest_gap,
        "zero_volume_ratio": None if zero_volume_ratio is None else round(float(zero_volume_ratio), 6),
        "coverage_pct": round(float(coverage_pct), 4),
        "history_days": int(history_days),
        "train_ready": bool(train_ready),
        "status": status,
        "quality_flags": flags,
    }
