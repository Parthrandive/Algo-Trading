import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.history import get_latest_local_timestamp, get_latest_local_timestamp_db, normalize_symbol

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_FATAL = 3
EXIT_INTERRUPTED = 130

INTERVAL_SECONDS = {
    "1h": 3600,
    "1d": 86400,
}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _parse_utc_date(date_str: str, end_of_day: bool) -> datetime:
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date_str}'. Expected YYYY-MM-DD.") from exc

    if end_of_day:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed.replace(tzinfo=timezone.utc)


def build_symbol_candidates(raw_symbol: str) -> list[str]:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        return ["RELIANCE.NS"]
    normalized = normalize_symbol(symbol)
    candidates = [normalized]

    # Include a base NSE alias fallback.
    if normalized.endswith(".NS"):
        candidates.append(normalized[:-3])

    # Include the raw symbol in case user provided a global ticker (e.g. AAPL).
    if symbol != normalized:
        candidates.append(symbol)

    return _dedupe_preserve_order(candidates)


def resolve_historical_dir(symbol_candidates: list[str]) -> tuple[str, Path]:
    root = PROJECT_ROOT / "data" / "silver" / "ohlcv"
    if not root.exists():
        return symbol_candidates[0], root / symbol_candidates[0]

    for candidate in symbol_candidates:
        candidate_dir = root / candidate
        if candidate_dir.exists():
            return candidate, candidate_dir

    available_dirs = {entry.name.upper(): entry for entry in root.iterdir() if entry.is_dir()}
    for candidate in symbol_candidates:
        match = available_dirs.get(candidate.upper())
        if match is not None:
            return match.name, match

    return symbol_candidates[0], root / symbol_candidates[0]


def _build_failover_client(sentinel_config):
    from src.agents.sentinel.broker_client import BrokerAPIClient
    from src.agents.sentinel.failover_client import FailoverSentinelClient
    from src.agents.sentinel.nsepython_client import NSEPythonClient
    from src.agents.sentinel.yfinance_client import YFinanceClient

    primary = YFinanceClient()
    fallbacks = []

    broker_base_url = os.getenv("BROKER_API_BASE_URL")
    if broker_base_url:
        fallbacks.append(
            BrokerAPIClient(
                base_url=broker_base_url,
                api_key=os.getenv("BROKER_API_KEY"),
                access_token=os.getenv("BROKER_ACCESS_TOKEN"),
            )
        )

    fallbacks.append(NSEPythonClient())
    return FailoverSentinelClient(
        primary,
        fallbacks,
        failure_threshold=sentinel_config.failover.failure_threshold,
        cooldown_seconds=sentinel_config.failover.cooldown_seconds,
        recovery_success_threshold=sentinel_config.failover.recovery_success_threshold,
    )


def _build_autofetch_rerun_command(
    symbol: str,
    *,
    interval: str,
    days: int,
    start_date: datetime | None,
    end_date: datetime | None,
) -> str:
    command = ["python3", "scripts/show_latest_data.py", symbol, "--auto-fetch-missing-history"]

    if start_date is not None:
        command.extend(["--from", start_date.strftime("%Y-%m-%d")])
    else:
        command.extend(["--days", str(days)])

    if end_date is not None:
        command.extend(["--to", end_date.strftime("%Y-%m-%d")])

    if interval != "1h":
        command.extend(["--interval", interval])

    return " ".join(shlex.quote(token) for token in command)


def _load_recent_history(base_dir: Path) -> tuple[pd.DataFrame | None, list[str], str | None]:
    if not base_dir.exists():
        return None, [], f"No historical data found in {base_dir}"

    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        return None, [], "No parquet files found."

    latest_files = files[-2:]
    dataframes = []
    for parquet_file in latest_files:
        try:
            dataframes.append(pd.read_parquet(parquet_file))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping unreadable file {parquet_file.name}: {exc}")

    if not dataframes:
        return None, [f.name for f in latest_files], "No readable parquet files available."

    combined = pd.concat(dataframes, ignore_index=True)
    return combined, [f.name for f in latest_files], None


def _count_symbol_rows(base_dir: Path) -> int:
    if not base_dir.exists():
        return 0

    total_rows = 0
    for parquet_file in base_dir.rglob("*.parquet"):
        try:
            total_rows += len(pd.read_parquet(parquet_file))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping unreadable file {parquet_file.name}: {exc}")
    return total_rows


def _get_latest_timestamp(symbol: str):
    return get_latest_local_timestamp_db(symbol) or get_latest_local_timestamp(symbol)


def _build_historical_backfill_service():
    from src.agents.sentinel.bronze_recorder import BronzeRecorder
    from src.agents.sentinel.historical_backfill import HistoricalBackfillService

    return HistoricalBackfillService(
        bronze_recorder=BronzeRecorder(),
    )


def _get_historical_bars(symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
    from src.db.queries import get_bars

    return get_bars(symbol, start_date, end_date, interval=interval)


def _build_live_market_service():
    from src.agents.sentinel.live_market import LiveMarketIngestionService

    return LiveMarketIngestionService()


def show_historical_data(
    symbol_candidates: list[str],
    *,
    auto_fetch: bool,
    interval: str,
    days: int,
    start_date: datetime | None,
    end_date: datetime | None,
) -> dict:
    target_symbol = normalize_symbol(symbol_candidates[0])
    _, base_dir = resolve_historical_dir(symbol_candidates)

    result = {
        "symbol": target_symbol,
        "status": "NO_DATA",
        "fetched_bars": 0,
        "bars_fetched": 0,
        "bars_saved": 0,
        "silver_path": str(base_dir.resolve()),
        "displayed_rows": 0,
        "files": [],
        "error": None,
    }

    print(f"--- 1. Checking Historical Data (Silver Layer) for {target_symbol} ---")

    if auto_fetch:
        now_utc = datetime.now(timezone.utc)
        desired_end = end_date or now_utc
        latest_ts = _get_latest_timestamp(target_symbol)
        if latest_ts is not None and latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)

        if start_date is not None:
            fetch_start = start_date
        elif latest_ts is not None:
            fetch_start = latest_ts + timedelta(seconds=1)
        else:
            fetch_start = desired_end - timedelta(days=days)

        fetch_needed = True
        if latest_ts is not None and start_date is None:
            gap_seconds = (desired_end - latest_ts).total_seconds()
            if gap_seconds < INTERVAL_SECONDS[interval]:
                fetch_needed = False
                print(f"Data is up-to-date (gap {gap_seconds:.0f}s < {INTERVAL_SECONDS[interval]}s). Skipping fetch.")

        if fetch_start >= desired_end:
            fetch_needed = False
            print("Fetch window is empty after gap check. Skipping fetch.")

        if fetch_needed:
            print(f"Fetching missing history from {fetch_start.isoformat()} to {desired_end.isoformat()} ({interval})")
            try:
                service = _build_historical_backfill_service()
                service_result = service.backfill_symbol(
                    symbol=target_symbol,
                    start_date=fetch_start,
                    end_date=desired_end,
                    interval=interval,
                )
                bars_fetched = int(service_result.get("bars_fetched", 0))
                bars_saved = int(service_result.get("bars_added", 0))
                silver_path = str(base_dir.resolve())

                result["fetched_bars"] = bars_fetched
                result["bars_fetched"] = bars_fetched
                result["bars_saved"] = bars_saved
                result["silver_path"] = silver_path

                print("Fetch persistence summary:")
                print(f"  bars_fetched: {bars_fetched}")
                print(f"  bars_saved: {bars_saved}")
                print(f"  silver_path: {silver_path}")
                if service_result.get("quality"):
                    print(
                        "  quality: "
                        f"status={service_result['quality'].get('status')} "
                        f"train_ready={service_result['quality'].get('train_ready')} "
                        f"coverage={service_result['quality'].get('coverage_pct')}"
                    )
            except Exception as exc:
                result["status"] = "ERROR"
                result["error"] = f"Failed to fetch historical data dynamically: {exc}"
                print(result["error"])
    else:
        print("Local-only mode enabled. Reading local data only.")
    
    # Let's request the last 30 days from DB if start_date is not specified
    query_end = end_date or datetime.now(timezone.utc)
    query_start = start_date or (query_end - timedelta(days=days))
    
    df = _get_historical_bars(target_symbol, query_start, query_end, interval)
    
    # We still keep resolve_historical_dir for the "silver_path" output for compatibility
    result["silver_path"] = str(base_dir.resolve())
    result["files"] = ["postgresql_db"]

    if df.empty:
        msg = f"No historical data found in DB for {target_symbol} between {query_start} and {query_end}"
        print(msg)
        if not auto_fetch:
            print(
                "Rerun with auto-fetch to pull and persist history:\n"
                f"  {_build_autofetch_rerun_command(target_symbol, interval=interval, days=days, start_date=start_date, end_date=end_date)}"
            )
        if result["status"] != "ERROR":
            result["error"] = msg
        return result

    assert df is not None
    preferred_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    display_columns = [column for column in preferred_columns if column in df.columns]
    if not display_columns:
        msg = "Historical files loaded, but expected OHLCV columns are missing."
        print(msg)
        print(f"Available columns: {list(df.columns)}")
        result["status"] = "ERROR"
        result["error"] = msg
        return result

    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=True)
        # Convert to IST for easier reading
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                # If naive, localize to UTC first, then convert. If aware, just convert.
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ist)
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert(ist)
            except Exception:
                pass

    print("Reading from: PostgreSQL -> sentinel_db.ohlcv_bars")
    print("\nLast 10 records:")
    print(df.tail(10)[display_columns])

    result["status"] = "SUCCESS"
    result["displayed_rows"] = min(10, len(df))
    return result


def _build_nsepython_client():
    from src.agents.sentinel.nsepython_client import NSEPythonClient

    return NSEPythonClient()


def _build_yfinance_client():
    from src.agents.sentinel.yfinance_client import YFinanceClient

    return YFinanceClient()


def choose_client_order(symbol: str) -> list[tuple[str, callable]]:
    if symbol.endswith(".NS"):
        return [("NSEPython", _build_nsepython_client), ("YFinance", _build_yfinance_client)]
    return [("YFinance", _build_yfinance_client), ("NSEPython", _build_nsepython_client)]


def show_live_quote(symbol_candidates: list[str]) -> dict:
    print("\n--- 2. Checking Real-Time Data (Live Quote) ---")

    result = {
        "status": "ERROR",
        "symbol": None,
        "price": None,
        "volume": None,
        "source": None,
        "timestamp": None,
        "errors": [],
    }

    service = _build_live_market_service()
    for candidate in symbol_candidates:
        print(f"Trying symbol: {candidate}")
        try:
            live_result = service.poll_symbol(candidate)
            if live_result["status"] == "FAILED":
                result["errors"].append(f"{candidate}: {live_result['message']}")
                continue

            print("\nSUCCESS: Received Live Observation")
            print(f"Symbol: {live_result['symbol']}")
            print(f"Price:  {live_result['last_price']}")
            print(f"Volume: {live_result['volume']}")
            print(f"Source: {live_result['source_name']}")
            print(f"Fresh:  {live_result['freshness_status']}")
            print(f"State:  {live_result['source_status']}")
            if live_result.get("timestamp"):
                import pytz

                ist = pytz.timezone("Asia/Kolkata")
                tick_time_ist = datetime.fromisoformat(live_result["timestamp"].replace("Z", "+00:00")).astimezone(ist)
                print(f"Time:   {tick_time_ist}")

            result.update(
                {
                    "status": live_result["status"],
                    "symbol": live_result["symbol"],
                    "price": live_result["last_price"],
                    "volume": live_result["volume"],
                    "source": live_result["source_name"],
                    "timestamp": live_result["timestamp"],
                    "freshness_status": live_result["freshness_status"],
                    "source_status": live_result["source_status"],
                }
            )
            return result
        except Exception as exc:
            result["errors"].append(f"live_service ({candidate}): {exc}")

    print("Failed to fetch live quote for all symbol/provider attempts.")
    for error in result["errors"]:
        print(f"- {error}")

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show latest historical and live data for a symbol.")
    parser.add_argument("symbol", nargs="?", default="RELIANCE.NS", help="Symbol to inspect (e.g., TCS.NS)")

    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument(
        "--auto-fetch-missing-history",
        dest="auto_fetch_missing_history",
        action="store_true",
        default=True,
        help="Fetch missing/gapped history before displaying (default behavior).",
    )
    fetch_group.add_argument(
        "--local-only",
        dest="auto_fetch_missing_history",
        action="store_false",
        help="Read local data only; do not fetch missing/gapped history.",
    )

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--days", type=int, default=30, help="Days to backfill when no local history exists (default: 30)")
    window_group.add_argument("--from", dest="from_date", type=str, help="Backfill start date (UTC) in YYYY-MM-DD")

    parser.add_argument("--to", dest="to_date", type=str, help="Backfill end date (UTC) in YYYY-MM-DD")
    parser.add_argument("--interval", choices=sorted(INTERVAL_SECONDS), default="1h", help="History interval for fetch")
    parser.add_argument("--no-live", action="store_true", help="Skip live quote fetch")
    parser.add_argument("--json", action="store_true", help="Emit a JSON summary")
    return parser


def run(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.days < 1:
        parser.error("--days must be >= 1")

    start_date = _parse_utc_date(args.from_date, end_of_day=False) if args.from_date else None
    end_date = _parse_utc_date(args.to_date, end_of_day=True) if args.to_date else None
    if start_date and end_date and start_date >= end_date:
        parser.error("--from must be earlier than --to")

    symbol_candidates = build_symbol_candidates(args.symbol)
    historical = show_historical_data(
        symbol_candidates,
        auto_fetch=args.auto_fetch_missing_history,
        interval=args.interval,
        days=args.days,
        start_date=start_date,
        end_date=end_date,
    )

    if args.no_live:
        live = {"status": "SKIPPED", "reason": "--no-live set"}
    else:
        live = show_live_quote(symbol_candidates)

    # --- 3. Auto-trigger Preprocessing (Silver -> Gold) ---
    target_symbol = normalize_symbol(symbol_candidates[0])
    _run_preprocessing(target_symbol)

    if args.json:
        payload = {
            "requested_symbol": args.symbol,
            "normalized_candidates": symbol_candidates,
            "historical": historical,
            "live": live,
        }
        print(json.dumps(payload, indent=2, default=str))

    if historical["status"] == "ERROR" and live.get("status") == "ERROR":
        return EXIT_FAILURE
    return EXIT_SUCCESS


def _run_preprocessing(symbol: str) -> None:
    """Auto-trigger preprocessing pipeline for the given symbol after ingestion."""
    print(f"\n--- 3. Auto-Preprocessing (Silver → Gold) for {symbol} ---")
    try:
        from src.agents.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        snapshot_id = f"auto_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            text_source_path="db_virtual",
            snapshot_id=snapshot_id,
            symbol_filter=symbol,
        )
        print(f"Preprocessing complete: {len(output.records)} cleaned rows → gold_features")
    except Exception as exc:
        print(f"Warning: Auto-preprocessing failed for {symbol}: {exc}")


def main() -> None:
    try:
        code = run(sys.argv[1:])
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Fatal error: {exc}")
        code = EXIT_FATAL

    raise SystemExit(code)


if __name__ == "__main__":
    main()
