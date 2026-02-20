import sys
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.utils.history import normalize_symbol, get_latest_local_timestamp

# Imports for fetching
from src.agents.sentinel.broker_client import BrokerAPIClient
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.config import load_default_sentinel_config


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def build_symbol_candidates(raw_symbol: str) -> list[str]:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        return ["RELIANCE.NS"]

    if symbol.endswith(".NS"):
        base_symbol = symbol[:-3]
        return _dedupe_preserve_order([symbol, base_symbol])

    # For plain company symbols (e.g., TCS), try NSE suffix first.
    is_plain_ticker = symbol.replace("-", "").isalnum() and "." not in symbol and "=" not in symbol and "^" not in symbol
    if is_plain_ticker:
        return _dedupe_preserve_order([f"{symbol}.NS", symbol])

    return [symbol]


def resolve_historical_dir(symbol_candidates: list[str]) -> tuple[str, Path]:
    root = Path("data/silver/ohlcv")
    if not root.exists():
        return symbol_candidates[0], root / symbol_candidates[0]

    # Exact match first.
    for candidate in symbol_candidates:
        candidate_dir = root / candidate
        if candidate_dir.exists():
            return candidate, candidate_dir

    # Case-insensitive fallback.
    available_dirs = {entry.name.upper(): entry for entry in root.iterdir() if entry.is_dir()}
    for candidate in symbol_candidates:
        match = available_dirs.get(candidate.upper())
        if match is not None:
            return match.name, match

    return symbol_candidates[0], root / symbol_candidates[0]


def _build_failover_client() -> FailoverSentinelClient:
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
    return FailoverSentinelClient(primary, fallbacks, failure_threshold=2, cooldown_seconds=60, recovery_success_threshold=2)


def show_historical_data(symbol_candidates: list[str], auto_fetch: bool = False) -> None:
    # Pick the most canonical symbol for fetching if we have to fetch
    target_symbol = normalize_symbol(symbol_candidates[0])
    storage_symbol, base_dir = resolve_historical_dir(symbol_candidates)

    print(f"--- 1. Checking Historical Data (Silver Layer) for {target_symbol} ---")

    if auto_fetch:
        print("Auto-fetch flag is SET. Checking gap...")
        now = datetime.now(timezone.utc)
        latest_ts = get_latest_local_timestamp(target_symbol)
        
        start_date = latest_ts if latest_ts is not None else (now - timedelta(days=7))
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)

        gap_seconds = (now - start_date).total_seconds()
        
        # We assume 1h data polling. If gap is larger than 1h (3600s), we fetch.
        if latest_ts is None or gap_seconds >= 3600:
            print(f"Gap detected. Fetching data from {start_date} to {now}...")
            config = load_default_sentinel_config()
            pipeline = SentinelIngestPipeline(
                client=_build_failover_client(),
                silver_recorder=SilverRecorder(),
                bronze_recorder=BronzeRecorder(),
                session_rules=config.session_rules,
            )
            try:
                # We add 1 second to start_date to strictly avoid downloading the exact same bar twice
                fetch_start = start_date + timedelta(seconds=1) if latest_ts else start_date
                bars = pipeline.ingest_historical(target_symbol, fetch_start, now, interval="1h")
                print(f"Fetched and saved {len(bars)} bars.")
                # After fetch, update base_dir to point to canonical name
                storage_symbol, base_dir = resolve_historical_dir([target_symbol])
            except Exception as e:
                print(f"Failed to fetch historical data dynamically: {e}")
        else:
            print(f"Data is up-to-date (gap is < 1 hour: 0 delta). Skipping fetch.")
    else:
        print("Auto-fetch flag is NOT SET. Reading local only.")

    if not base_dir.exists():
        print(f"No historical data found in {base_dir}")
        return

    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        print("No parquet files found.")
        return

    latest_files = files[-2:]
    print(f"Reading files: {[f.name for f in latest_files]}")

    dataframes = []
    for parquet_file in latest_files:
        try:
            dataframes.append(pd.read_parquet(parquet_file))
        except Exception as exc:
            print(f"Skipping unreadable file {parquet_file.name}: {exc}")

    if not dataframes:
        print("No readable parquet files available.")
        return

    df = pd.concat(dataframes, ignore_index=True)
    preferred_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    display_columns = [column for column in preferred_columns if column in df.columns]

    if not display_columns:
        print("Historical files loaded, but expected OHLCV columns are missing.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Sort to ensure tail gives the actual latest
    df = df.sort_values(by="timestamp", ascending=True)

    print(f"\nLast 10 records (spanning {len(latest_files)} days):")
    print(df.tail(10)[display_columns])


def choose_client_order(symbol: str) -> list[tuple[str, object]]:
    if symbol.endswith(".NS"):
        return [("NSEPython", NSEPythonClient()), ("YFinance", YFinanceClient())]
    return [("YFinance", YFinanceClient()), ("NSEPython", NSEPythonClient())]


def show_live_quote(symbol_candidates: list[str]) -> None:
    print("\n--- 2. Checking Real-Time Data (Live Quote) ---")

    errors: list[str] = []
    for candidate in symbol_candidates:
        print(f"Trying symbol: {candidate}")

        for client_name, client in choose_client_order(candidate):
            print(f"Fetching via {client_name}...")
            try:
                tick = client.get_stock_quote(candidate)
                print("\nSUCCESS: Received Live Tick")
                print(f"Symbol: {tick.symbol}")
                print(f"Price:  {tick.price}")
                print(f"Volume: {tick.volume}")
                print(f"Source: {tick.source_type}")
                print(f"Time:   {tick.timestamp}")
                return
            except Exception as exc:
                errors.append(f"{client_name} ({candidate}): {exc}")

    print("Failed to fetch live quote for all symbol/provider attempts.")
    for error in errors:
        print(f"- {error}")


def show_data() -> None:
    parser = argparse.ArgumentParser(description="Show latest data (historical + live) for a stock symbol.")
    parser.add_argument("symbol", nargs="?", default="RELIANCE.NS", help="Stock symbol to view (e.g., TCS.NS)")
    parser.add_argument("--auto-fetch-missing-history", action="store_true", help="Automatically fetch historical data if missing or gapped.")

    args = parser.parse_args()

    symbol_candidates = build_symbol_candidates(args.symbol)

    show_historical_data(symbol_candidates, auto_fetch=args.auto_fetch_missing_history)
    show_live_quote(symbol_candidates)


if __name__ == "__main__":
    show_data()
