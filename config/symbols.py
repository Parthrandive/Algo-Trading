from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine


FOREX_SYMBOLS = ["USDINR=X"]
# This list is fixed - forex pairs never change role.
# USDINR is always an external feature, never a target.
# Add other forex pairs here if needed in future.

EQUITY_SYMBOLS: list[str] = []
# This list is intentionally empty in the config file.
# It is populated dynamically at runtime from the data pipeline.
# Whatever equity symbols are available and have sufficient data are included automatically.
# No symbol is ever hardcoded here.

INDEX_SYMBOLS = ["^NSEI"]
SENTINEL_CORE_SYMBOLS = [
    "RELIANCE.NS",
    "TATASTEEL.NS",
]
WATCHLIST_ROTATE_BATCH_SIZE = 8
WATCHLIST_ROTATING_POOL = [
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LARSEN.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "KOTAKBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "WIPRO.NS",
    "ULTRACEMCO.NS",
    "M&M.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "INDUSINDBK.NS",
    "ONGC.NS",
    "BAJAJFINSV.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "GRASIM.NS",
    "TATAMOTORS.NS",
    "APOLLOHOSP.NS",
    "DIVISLAB.NS",
    "CIPLA.NS",
    "EICHERMOT.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "HEROMOTOCO.NS",
    "DRREDDY.NS",
    "UPL.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
]

MIN_ROWS = 300
MIN_TRAIN_ROWS = 200
MIN_VAL_ROWS = 50
MIN_TEST_ROWS = 50
MAX_ZERO_PCT = 0.05
MAX_GAP_COUNT = 100
FX_RESULTS_NOTE = "USDINR=X — external feature only, not a prediction target"


@dataclass(frozen=True)
class SplitCounts:
    train_rows: int
    val_rows: int
    test_rows: int


@dataclass
class SymbolValidationResult:
    symbol: str
    is_active: bool
    reason: str | None = None
    frame: pd.DataFrame | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolDiscoveryResult:
    active_symbols: list[str]
    skipped_symbols: list[str]
    skipped_reasons: dict[str, str]
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_symbols: list[str] = field(default_factory=list)


def dedupe_symbols(symbols: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for symbol in symbols:
        value = str(symbol).strip()
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def is_forex(symbol: str) -> bool:
    return str(symbol).strip() in FOREX_SYMBOLS


def is_equity(symbol: str) -> bool:
    return not is_forex(symbol)


def is_forex_symbol(symbol: str) -> bool:
    return is_forex(symbol)


def format_symbol_list(symbols: Sequence[str]) -> str:
    values = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    return "[" + ", ".join(values) + "]"


def print_fx_results_note(print_fn: Callable[[str], None] = print) -> None:
    print_fn(FX_RESULTS_NOTE)


def print_symbol_selection_summary(
    *,
    active_symbols: Sequence[str],
    skipped_reasons: dict[str, str],
    print_fn: Callable[[str], None] = print,
) -> None:
    print_fn(f"Active equity symbols this run: {format_symbol_list(active_symbols)}")
    skipped_summary = [f"{symbol} ({reason})" for symbol, reason in skipped_reasons.items()]
    print_fn(f"Skipped symbols: {format_symbol_list(skipped_summary)}")
    print_fn(f"Forex (external feature): {format_symbol_list(FOREX_SYMBOLS)}")


def assert_no_forex_targets(training_symbols: Sequence[str]) -> None:
    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )


def _default_database_url(database_url: str | None = None) -> str | None:
    return database_url or os.getenv("DATABASE_URL") or "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"


def _interval_to_timedelta(interval: str) -> pd.Timedelta | None:
    value = str(interval).strip().lower()
    match = re.fullmatch(r"(\d+)([a-z]+)", value)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    mapping = {
        "m": "min",
        "min": "min",
        "mins": "min",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "d": "D",
        "day": "D",
        "days": "D",
        "w": "W",
        "wk": "W",
    }
    normalized = mapping.get(unit)
    if normalized is None:
        return None
    return pd.to_timedelta(amount, unit=normalized)


def _compute_gap_count(timestamps: pd.Series, interval: str) -> int:
    delta = _interval_to_timedelta(interval)
    if delta is None or timestamps.empty:
        return 0
    series = pd.Series(pd.to_datetime(timestamps, utc=True, errors="coerce")).dropna().sort_values().reset_index(drop=True)
    if len(series) < 2:
        return 0

    gap_count = 0
    if delta >= pd.Timedelta(days=1):
        step_days = max(int(delta / pd.Timedelta(days=1)), 1)
        previous = series.iloc[0]
        for current in series.iloc[1:]:
            business_days = max(len(pd.bdate_range(previous.date(), current.date())) - 1, 0)
            missing_steps = max((business_days // step_days) - 1, 0)
            if missing_steps > 0:
                gap_count += 1
            previous = current
        return gap_count

    previous = series.iloc[0]
    for current in series.iloc[1:]:
        if previous.date() != current.date():
            previous = current
            continue
        missing_steps = max(int(round((current - previous) / delta)) - 1, 0)
        if missing_steps > 0:
            gap_count += 1
        previous = current
    return gap_count


def _clean_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(frame)}")
    if "timestamp" not in frame.columns:
        raise ValueError("Frame must include a timestamp column.")
    if "close" not in frame.columns:
        raise ValueError("Frame must include a close column.")

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    cleaned["close"] = pd.to_numeric(cleaned["close"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return cleaned


def validate_equity_symbol(
    *,
    symbol: str,
    frame: pd.DataFrame,
    interval: str,
    split_counts: SplitCounts | None = None,
    required_start: Any | None = None,
    required_end: Any | None = None,
) -> SymbolValidationResult:
    if is_forex(symbol):
        return SymbolValidationResult(symbol=symbol, is_active=False, reason="forex_context_only")

    try:
        cleaned = _clean_symbol_frame(frame)
    except Exception as exc:
        return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_or_clean_failed: {exc}")

    row_count = int(len(cleaned))
    diagnostics: dict[str, Any] = {
        "row_count": row_count,
        "start_timestamp": None if cleaned.empty else cleaned["timestamp"].min().isoformat(),
        "end_timestamp": None if cleaned.empty else cleaned["timestamp"].max().isoformat(),
    }
    if row_count < MIN_ROWS:
        return SymbolValidationResult(
            symbol=symbol,
            is_active=False,
            reason=f"rows={row_count} < MIN_ROWS={MIN_ROWS}",
            diagnostics=diagnostics,
        )

    zero_close_pct = float((cleaned["close"] <= 0).mean()) if row_count else 1.0
    diagnostics["zero_close_pct"] = zero_close_pct
    if zero_close_pct > MAX_ZERO_PCT:
        return SymbolValidationResult(
            symbol=symbol,
            is_active=False,
            reason=f"zero_close_pct={zero_close_pct:.4f} > MAX_ZERO_PCT={MAX_ZERO_PCT:.2f}",
            diagnostics=diagnostics,
        )

    gap_count = _compute_gap_count(cleaned["timestamp"], interval)
    diagnostics["gap_count"] = gap_count
    if gap_count > MAX_GAP_COUNT:
        return SymbolValidationResult(
            symbol=symbol,
            is_active=False,
            reason=f"gap_count={gap_count} > MAX_GAP_COUNT={MAX_GAP_COUNT}",
            diagnostics=diagnostics,
        )

    if required_start is not None:
        start_ts = pd.to_datetime(required_start, utc=True, errors="coerce")
        if pd.isna(start_ts) or cleaned["timestamp"].min() > start_ts:
            return SymbolValidationResult(
                symbol=symbol,
                is_active=False,
                reason=f"training_range_start_missing: need <= {start_ts}",
                diagnostics=diagnostics,
            )

    if required_end is not None:
        end_ts = pd.to_datetime(required_end, utc=True, errors="coerce")
        if pd.isna(end_ts) or cleaned["timestamp"].max() < end_ts:
            return SymbolValidationResult(
                symbol=symbol,
                is_active=False,
                reason=f"training_range_end_missing: need >= {end_ts}",
                diagnostics=diagnostics,
            )

    if split_counts is not None:
        diagnostics["train_rows"] = int(split_counts.train_rows)
        diagnostics["val_rows"] = int(split_counts.val_rows)
        diagnostics["test_rows"] = int(split_counts.test_rows)
        if split_counts.train_rows < MIN_TRAIN_ROWS:
            return SymbolValidationResult(
                symbol=symbol,
                is_active=False,
                reason=f"train_rows={split_counts.train_rows} < MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}",
                diagnostics=diagnostics,
            )
        if split_counts.val_rows < MIN_VAL_ROWS:
            return SymbolValidationResult(
                symbol=symbol,
                is_active=False,
                reason=f"val_rows={split_counts.val_rows} < MIN_VAL_ROWS={MIN_VAL_ROWS}",
                diagnostics=diagnostics,
            )
        if split_counts.test_rows < MIN_TEST_ROWS:
            return SymbolValidationResult(
                symbol=symbol,
                is_active=False,
                reason=f"test_rows={split_counts.test_rows} < MIN_TEST_ROWS={MIN_TEST_ROWS}",
                diagnostics=diagnostics,
            )

    return SymbolValidationResult(
        symbol=symbol,
        is_active=True,
        frame=cleaned,
        diagnostics=diagnostics,
    )


def discover_pipeline_equity_symbols(
    *,
    interval: str,
    requested_symbols: Sequence[str] | None = None,
    database_url: str | None = None,
) -> list[str]:
    engine = get_engine(_default_database_url(database_url))
    requested = dedupe_symbols(requested_symbols or [])
    requested_set = {value.upper() for value in requested}

    query = text(
        """
        SELECT DISTINCT symbol
        FROM market_data_quality
        WHERE dataset_type = 'historical'
          AND interval = :interval
          AND exchange = 'NSE'
          AND asset_type = 'equity'
        ORDER BY symbol ASC
        """
    )
    df = pd.read_sql(query, engine, params={"interval": interval})
    if df.empty:
        fallback_query = text(
            """
            SELECT DISTINCT symbol
            FROM ohlcv_bars
            WHERE interval = :interval
            ORDER BY symbol ASC
            """
        )
        df = pd.read_sql(fallback_query, engine, params={"interval": interval})

    candidates = dedupe_symbols(df.get("symbol", pd.Series(dtype=str)).astype(str).tolist())
    candidates = [symbol for symbol in candidates if symbol and is_equity(symbol)]
    if requested_set:
        candidates = [symbol for symbol in candidates if symbol.upper() in requested_set]
    return candidates


def discover_training_symbols(
    *,
    interval: str,
    validator: Callable[[str], SymbolValidationResult],
    requested_symbols: Sequence[str] | None = None,
    database_url: str | None = None,
    print_fn: Callable[[str], None] = print,
) -> SymbolDiscoveryResult:
    requested = dedupe_symbols(requested_symbols or [])
    pipeline_symbols = discover_pipeline_equity_symbols(
        interval=interval,
        requested_symbols=None,
        database_url=database_url,
    )
    missing_requested = [symbol for symbol in requested if symbol.upper() not in {item.upper() for item in pipeline_symbols}]
    candidate_symbols = requested or pipeline_symbols

    if not candidate_symbols:
        raise ValueError(f"No NSE equity symbols found in the data pipeline for interval={interval}.")

    active_symbols: list[str] = []
    skipped_symbols: list[str] = []
    skipped_reasons: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    print_fn(f"Discovered candidate equity symbols from data pipeline: {format_symbol_list(candidate_symbols)}")

    for symbol in missing_requested:
        skipped_symbols.append(symbol)
        skipped_reasons[symbol] = "not_found_in_data_pipeline"
        print_fn(f"WARNING: [{symbol}] skipped - not found in data pipeline")

    for symbol in candidate_symbols:
        if symbol in missing_requested:
            continue
        outcome = validator(symbol)
        diagnostics[symbol] = dict(outcome.diagnostics)
        if outcome.is_active:
            active_symbols.append(symbol)
            if outcome.frame is not None:
                frames[symbol] = outcome.frame
        else:
            skipped_symbols.append(symbol)
            skipped_reasons[symbol] = outcome.reason or "unknown_reason"
            print_fn(f"WARNING: [{symbol}] skipped - {skipped_reasons[symbol]}")

    print_fn(f"Active equity symbols this run: {format_symbol_list(active_symbols)}")
    skipped_summary = [f"{symbol} ({skipped_reasons[symbol]})" for symbol in skipped_symbols]
    print_fn(f"Skipped symbols: {format_symbol_list(skipped_summary)}")
    print_fn(f"Forex (external feature): {format_symbol_list(FOREX_SYMBOLS)}")

    return SymbolDiscoveryResult(
        active_symbols=active_symbols,
        skipped_symbols=skipped_symbols,
        skipped_reasons=skipped_reasons,
        frames=frames,
        diagnostics=diagnostics,
        candidate_symbols=candidate_symbols,
    )
