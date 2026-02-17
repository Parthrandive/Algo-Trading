
import sys
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.schemas.market_data import Tick

def show_data():
    print("--- 1. Checking Historical Data (Silver Layer) ---")
    # Find the latest parquet file we created
    base_dir = Path("data/silver/ohlcv/RELIANCE.NS")
    if not base_dir.exists():
        print("No historical data found in data/silver/ohlcv/RELIANCE.NS")
    else:
        # data/silver/ohlcv/RELIANCE.NS/{year}/{month}/{date}.parquet
        # Let's just find any parquet file
        files = list(base_dir.rglob("*.parquet"))
        if files:
            latest_file = sorted(files)[-1]
            print(f"Reading latest file: {latest_file}")
            df = pd.read_parquet(latest_file)
            print("\nLast 5 records:")
            print(df.tail()[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        else:
            print("No parquet files found.")

    print("\n--- 2. Checking Real-Time Data (NSEPython) ---")
    client = NSEPythonClient()
    symbol = "RELIANCE.NS"
    print(f"Fetching live quote for {symbol}...")
    try:
        tick = client.get_stock_quote(symbol)
        print(f"\nSUCCESS: Received Live Tick:")
        print(f"Symbol: {tick.symbol}")
        print(f"Price:  {tick.price}")
        print(f"Volume: {tick.volume}")
        print(f"Source: {tick.source_type}")
        print(f"Time:   {tick.timestamp}")
    except Exception as e:
        print(f"Failed to fetch live quote: {e}")

if __name__ == "__main__":
    show_data()
