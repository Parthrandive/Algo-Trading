from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

ASSET_TYPE_EQUITY = "equity"
ASSET_TYPE_FOREX = "forex"
ASSET_TYPE_INDEX = "index"
ASSET_TYPE_UNKNOWN = "unknown"

_CURRENCY_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "CNY",
    "EUR",
    "GBP",
    "HKD",
    "INR",
    "JPY",
    "NZD",
    "SGD",
    "USD",
}

_INTERVAL_TO_DURATION = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "10m": timedelta(minutes=10),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def infer_asset_type(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ASSET_TYPE_UNKNOWN
    if normalized.startswith("^"):
        return ASSET_TYPE_INDEX
    if normalized.endswith("=X"):
        return ASSET_TYPE_FOREX
    if len(normalized) == 6 and normalized.isalpha():
        base = normalized[:3]
        quote = normalized[3:]
        if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
            return ASSET_TYPE_FOREX
    if normalized.endswith(".NS") or normalized.replace("-", "").isalnum():
        return ASSET_TYPE_EQUITY
    return ASSET_TYPE_UNKNOWN


def normalize_timestamp(value: datetime, *, default_timezone: ZoneInfo = UTC) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=default_timezone).astimezone(UTC)
    return value.astimezone(UTC)


def interval_to_timedelta(interval: str) -> timedelta:
    try:
        return _INTERVAL_TO_DURATION[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval '{interval}'.") from exc


def session_slots_for_date(session_rules, trading_date: date, interval: str) -> list[datetime]:
    """
    Returns expected bar start timestamps in UTC for a single NSE session date.

    The final partial session bucket is included so 1h bars produce:
    09:15, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15 IST.
    """
    interval_delta = interval_to_timedelta(interval)
    local_start = datetime.combine(trading_date, session_rules.regular_open, tzinfo=session_rules.tz)
    local_close = datetime.combine(trading_date, session_rules.regular_close, tzinfo=session_rules.tz)

    slots: list[datetime] = []
    cursor = local_start
    while cursor < local_close:
        slots.append(cursor.astimezone(UTC))
        cursor += interval_delta
    return slots


def expected_session_timestamps(
    *,
    session_rules,
    start: datetime,
    end: datetime,
    interval: str,
    asset_type: str,
) -> list[datetime]:
    if asset_type not in {ASSET_TYPE_EQUITY, ASSET_TYPE_INDEX}:
        return []

    start_utc = normalize_timestamp(start)
    end_utc = normalize_timestamp(end)
    current_local_date = start_utc.astimezone(session_rules.tz).date()
    end_local_date = end_utc.astimezone(session_rules.tz).date()

    expected: list[datetime] = []
    while current_local_date <= end_local_date:
        probe = datetime.combine(current_local_date, session_rules.regular_open, tzinfo=session_rules.tz)
        if session_rules.is_trading_session(probe):
            expected.extend(session_slots_for_date(session_rules, current_local_date, interval))
        current_local_date += timedelta(days=1)

    return [ts for ts in expected if start_utc <= ts <= end_utc]

