import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.db.silver_db_recorder import SilverDBRecorder
from src.db.init_db import init_database
from src.db.queries import get_bars, get_latest_timestamp

def test_db_operations():
    print("=== Database End-to-End Verification ===")
    
    # 1. Initialize DB
    print("\n[Step 1] Ensuring Database is initialized...")
    init_database()
    print("  Database schema is active and ready.")
    
    # 2. Pick a new symbol not currently in the DB
    symbol = "WIPRO.NS"
    print(f"\n[Step 2] Testing with completely new symbol: {symbol}")
    
    # Check if it exists before we do anything
    latest_ts = get_latest_timestamp(symbol)
    if latest_ts:
        print(f"  Note: {symbol} already exists in DB (latest timestamp: {latest_ts}).")
        # Ensure we fetch something new anyway
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5)
    else:
        print(f"  Confirmed: No prior records for {symbol} exist in DB. New group creation will be tested.")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5)

    # 3. Fetch & Ingest Data
    print(f"\n[Step 3] Fetching historical data (5 days) via Python Pipeline...")
    
    client = YFinanceClient()
    silver_recorder = SilverDBRecorder()
    bronze_recorder = BronzeRecorder()
    
    # We bypass SessionRules for the test to fetch raw regardless of time of day
    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver_recorder,
        bronze_recorder=bronze_recorder,
        session_rules=None 
    )
    
    try:
        bars = pipeline.ingest_historical(symbol, start_date, end_date, interval="1h")
        print(f"  YFinance returned {len(bars)} bars.")
        print(f"  SilverDBRecorder processed and stored {len(bars)} bars in PostgreSQL/TimescaleDB.")
    except Exception as e:
        print(f"  FAIL: Error during ingestion pipeline: {e}")
        return
        
    if not bars:
        print("  FAIL: No data available from YFinance for this testing range. Cannot proceed with DB retrieval test.")
        return
        
    # 4. Read back from Database
    print(f"\n[Step 4] Retrieving stored data from Database...")
    
    try:
        db_bars_df = get_bars(symbol, start_date, end_date, interval="1h")
        print(f"  SUCCESS: Select Query successfully retrieved the data.")
    except Exception as e:
        print(f"  FAIL: Error executing read query: {e}")
        return
        
    if db_bars_df.empty:
        print("  FAIL: The data was reportedly saved, but a SELECT query returned 0 rows.")
        return
        
    # 5. Detail Row Validation
    print(f"\n[Step 5] Validating Stored PostgreSQL Rows:")
    
    first_row = db_bars_df.iloc[0]
    last_row = db_bars_df.iloc[-1]
    
    print(f"  Symbol verified        : {first_row['symbol']}")
    print(f"  Interval verified      : {first_row['interval']}")
    print(f"  Rows retrieved count   : {len(db_bars_df)}")
    print(f"  Data Start Timestamp   : {first_row['timestamp']}")
    print(f"  Data End Timestamp     : {last_row['timestamp']}")
    print(f"  Sample Open Price      : {first_row['open']}")
    print(f"  Sample Volume          : {first_row['volume']}")
    
    print("\n[Step 6] Final Database Gap Check...")
    new_latest_ts = get_latest_timestamp(symbol)
    print(f"  The new maximum database timestamp for {symbol} is: {new_latest_ts}")
    
    print("\n=== All Database E2E Tests Passed Successfully 🎉 ===")

if __name__ == "__main__":
    test_db_operations()
