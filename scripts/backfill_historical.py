import sys
import argparse
import os
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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

CHECKPOINT_FILE = "data/backfill_checkpoint.json"

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict):
    os.makedirs(os.path.dirname(Path(CHECKPOINT_FILE).absolute()), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

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

def backfill_symbol(symbol: str, days: int) -> dict:
    target_symbol = normalize_symbol(symbol)
    now = datetime.now(timezone.utc)
    latest_ts = get_latest_local_timestamp(target_symbol)
    
    start_date = latest_ts if latest_ts is not None else (now - timedelta(days=days))
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)

    gap_seconds = (now - start_date).total_seconds()
    
    if latest_ts is not None and gap_seconds < 3600:
        return {"symbol": target_symbol, "status": "SUCCESS", "message": "Up to date", "bars_fetched": 0, "timestamp": now.isoformat()}

    # Add jitter to avoid thundering herd on thread start
    time.sleep(random.uniform(0.5, 2.0))
    
    try:
        config = load_default_sentinel_config()
        pipeline = SentinelIngestPipeline(
            client=_build_failover_client(),
            silver_recorder=SilverRecorder(),
            bronze_recorder=BronzeRecorder(),
            session_rules=config.session_rules,
        )
        
        # Add a tiny delta to strictly avoid downloading exactly same bar
        fetch_start = start_date + timedelta(seconds=1) if latest_ts else start_date
        bars = pipeline.ingest_historical(target_symbol, fetch_start, now, interval="1h")
        
        return {"symbol": target_symbol, "status": "SUCCESS", "message": "Fetched data", "bars_fetched": len(bars), "timestamp": now.isoformat()}
    except Exception as e:
        return {"symbol": target_symbol, "status": "FAILED", "message": str(e), "bars_fetched": 0, "timestamp": now.isoformat()}

def run_backfill(symbols: list[str], days: int, max_workers: int, force: bool):
    checkpoint = load_checkpoint() if not force else {}
    
    # Filter out already successful ones if not forcing
    pending_symbols = []
    for sym in symbols:
        sym_norm = normalize_symbol(sym)
        state = checkpoint.get(sym_norm, {})
        if not force and state.get("status") == "SUCCESS":
            print(f"[{sym_norm}] Skipping (marked SUCCESS in checkpoint).")
            continue
        pending_symbols.append(sym_norm)
        
    if not pending_symbols:
        print("No symbols left to process. Everything is up to date according to the checkpoint.")
        return

    print(f"Starting backfill for {len(pending_symbols)} symbols with {max_workers} workers.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backfill_symbol, sym, days): sym for sym in pending_symbols}
        
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                checkpoint[result["symbol"]] = result
                save_checkpoint(checkpoint)
                print(f"[{result['symbol']}] {result['status']}: {result['message']} (Bars: {result['bars_fetched']})")
            except Exception as e:
                print(f"[{sym}] Unexpected thread failure: {e}")
                checkpoint[sym] = {"status": "FAILED", "message": "Thread excepted", "timestamp": datetime.now(timezone.utc).isoformat()}
                save_checkpoint(checkpoint)

def read_universe(universe_file: str) -> list[str]:
    path = Path(universe_file)
    if not path.exists():
        raise FileNotFoundError(f"Universe file {universe_file} not found.")
    
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        # Find column related to symbol
        col = next((c for c in df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), df.columns[0])
        return df[col].dropna().astype(str).tolist()
    else:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Bulk backfill historical data for a universe of symbols.")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (e.g., TCS.NS,INFY.NS)")
    parser.add_argument("--universe", type=str, help="Path to a CSV/TXT universe file containing symbols")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch if no local data exists")
    parser.add_argument("--workers", type=int, default=3, help="Max concurrent downloads")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint successes and force fetch")
    args = parser.parse_args()

    symbols = []
    if args.universe:
        symbols.extend(read_universe(args.universe))
    if args.symbols:
        symbols.extend([s.strip() for s in args.symbols.split(",") if s.strip()])

    symbols = list(set([normalize_symbol(s) for s in symbols]))
    if not symbols:
        print("Please provide --symbols or --universe")
        sys.exit(1)

    run_backfill(symbols, args.days, args.workers, args.force)

if __name__ == "__main__":
    main()
