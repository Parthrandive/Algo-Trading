
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.recorder import SilverRecorder
from src.schemas.market_data import Bar

def test_ingest():
    print("Initializing YFinanceClient...")
    client = YFinanceClient()
    
    symbol = "RELIANCE.NS"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    try:
        bars = client.get_historical_data(symbol, start_date, end_date)
        print(f"Fetched {len(bars)} bars.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if not bars:
        print("No bars fetched. Exiting.")
        return

    print("Initializing SilverRecorder...")
    recorder = SilverRecorder()
    
    print("Saving bars to Silver Layer...")
    try:
        recorder.save_bars(bars)
        print("Bars saved successfully.")
    except Exception as e:
        print(f"Error saving bars: {e}")
        return

    # Verification
    # Check if file exists
    # We need to predict the path based on the first bar's date
    first_bar_date = bars[0].timestamp
    year = str(first_bar_date.year)
    month = f"{first_bar_date.month:02d}"
    date_str = first_bar_date.strftime('%Y-%m-%d')
    file_path = Path(f"data/silver/ohlcv/{symbol}/{year}/{month}/{date_str}.parquet")
    
    if file_path.exists():
        print(f"SUCCESS: File created at {file_path}")
    else:
        # It might be in a different file if we fetch 30 days. Let's check the directory.
        dir_path = Path(f"data/silver/ohlcv/{symbol}/{year}/{month}")
        if dir_path.exists():
             files = list(dir_path.glob("*.parquet"))
             if files:
                 print(f"SUCCESS: Files created in {dir_path}: {len(files)} files found.")
             else:
                 print(f"FAILURE: Directory exists but no parquet files found in {dir_path}")
        else:
             print(f"FAILURE: Directory {dir_path} not found.")

if __name__ == "__main__":
    test_ingest()
