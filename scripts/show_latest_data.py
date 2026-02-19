
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
    # Get symbol from command line args or default to RELIANCE.NS
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = "RELIANCE.NS"

    print(f"--- 1. Checking Historical Data (Silver Layer) for {symbol} ---")
    # Find the latest parquet file we created
    base_dir = Path(f"data/silver/ohlcv/{symbol}")
    if not base_dir.exists():
        print(f"No historical data found in data/silver/ohlcv/{symbol}")
    else:
        # data/silver/ohlcv/{symbol}/{year}/{month}/{date}.parquet
        # Find all parquet files recursively
        files = sorted(list(base_dir.rglob("*.parquet")))
        
        if files:
            # Take last 2 files (or just 1 if only 1 exists)
            latest_files = files[-2:]
            print(f"Reading files: {[f.name for f in latest_files]}")
            
            dfs = []
            for f in latest_files:
                dfs.append(pd.read_parquet(f))
            
            if dfs:
                df = pd.concat(dfs)
                print(f"\nLast 10 records (spanning {len(latest_files)} days):")
                print(df.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        else:
            print("No parquet files found.")

    if symbol.endswith(".NS"):
        print(f"\n--- 2. Checking Real-Time Data (NSEPython) for {symbol} ---")
        client = NSEPythonClient()
    else:
        print(f"\n--- 2. Checking Real-Time Data (YFinance) for {symbol} ---")
        from src.agents.sentinel.yfinance_client import YFinanceClient
        client = YFinanceClient()
    
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
