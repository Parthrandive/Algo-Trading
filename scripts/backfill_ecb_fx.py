#!/usr/bin/env python3
"""
Backfill historical USDINR forex rates using `forex-python`.
Fetches daily ECB reference rates back to 1999 and stores them in the Silver layer.
"""

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from forex_python.converter import CurrencyRates, RatesNotAvailableError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path("data")
BRONZE_DIR = BASE_DIR / "bronze" / "fx_history"
SILVER_DIR = BASE_DIR / "silver" / "ohlcv" / "USDINR=X"  # Store alongside existing yfinance data

def setup_directories() -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

def daterange(start_date: date, end_date: date):
    """Generator for dates between start_date and end_date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_single_date(cr: CurrencyRates, base: str, target: str, single_date: date) -> Dict[str, Any]:
    try:
        rate = cr.get_rate(base, target, single_date)
        return {
            "date": single_date.isoformat(),
            "base": base,
            "target": target,
            "rate": rate,
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
    except RatesNotAvailableError as e:
         logger.debug(f"Rate not available for {single_date}: {e}")
    except Exception as e:
         logger.warning(f"Unexpected error for {single_date}: {e}")
    return None

def fetch_rates(start_date: date, end_date: date, base: str = "USD", target: str = "INR") -> List[Dict[str, Any]]:
    """Fetch daily rates from forex-python for the given date range concurrently."""
    cr = CurrencyRates()
    results = []
    
    logger.info(f"Fetching {base}/{target} rates from {start_date} to {end_date} concurrently...")
    
    dates_to_fetch = [d for d in daterange(start_date, end_date) if d.weekday() < 5]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_date = {executor.submit(fetch_single_date, cr, base, target, d): d for d in dates_to_fetch}
        
        for idx, future in enumerate(as_completed(future_to_date)):
            res = future.result()
            if res:
                results.append(res)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dates_to_fetch)} days")
             
    logger.info(f"Successfully fetched {len(results)} daily rates.")
    
    # Sort results by date so they are in order
    results.sort(key=lambda x: x["date"])
    return results

def save_to_bronze(data: List[Dict[str, Any]], start_date: date, end_date: date) -> Path:
    """Save raw fetched data to Bronze JSONL."""
    filename = f"usdinr_ecb_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.jsonl"
    filepath = BRONZE_DIR / filename
    
    with filepath.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    logger.info(f"Stored {len(data)} raw records in Bronze: {filepath}")
    return filepath

def process_to_silver(bronze_file: Path) -> Path:
    """Process raw JSONL into the Silver Parquet schema (OHLCV-like for compatibility)."""
    records = []
    with bronze_file.open("r") as f:
        for line in f:
             records.append(json.loads(line))
             
    if not records:
         logger.warning("No records to process to Silver.")
         return None
         
    df = pd.DataFrame(records)
    
    # Convert string dates to datetime (UTC midnight)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    
    # Map to Silver Bar schema
    # Since we only have a daily midpoint rate, we use it for O/H/L/C
    df_silver = pd.DataFrame({
        "symbol": "USDINR=X",
        "timestamp": df["date"],
        "source_type": "official_api", # ECB via forex-python
        "interval": "1d",
        "open": df["rate"],
        "high": df["rate"],
        "low": df["rate"],
        "close": df["rate"],
        "volume": 0, # No volume data
        "quality_status": "PASS"
    })
    
    # Save as parquet (monthly partitions is ideal, but for a single bulk fetch we save as one file for now, 
    # or append to existing if we were merging. Let's save as a single historical file)
    start_str = df_silver["timestamp"].min().strftime('%Y%m%d')
    end_str = df_silver["timestamp"].max().strftime('%Y%m%d')
    
    filename = f"history_ecb_{start_str}_{end_str}.parquet"
    filepath = SILVER_DIR / filename
    
    df_silver.to_parquet(filepath, index=False)
    logger.info(f"Stored {len(df_silver)} normalized records in Silver: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Backfill historical USDINR rates using forex-python")
    parser.add_argument("--start", type=str, default="1999-01-04", help="Start date (YYYY-MM-DD), default 1999-01-04")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    setup_directories()
    
    raw_data = fetch_rates(start_date, end_date)
    if raw_data:
        bronze_file = save_to_bronze(raw_data, start_date, end_date)
        process_to_silver(bronze_file)
        
        # Verify
        df_silver = pd.read_parquet(SILVER_DIR) # Reads all parquet files in dir
        logger.info(f"Total Silver records for USDINR=X now: {len(df_silver)}")
        logger.info(f"Date range in Silver: {df_silver['timestamp'].min()} to {df_silver['timestamp'].max()}")

if __name__ == "__main__":
    main()
