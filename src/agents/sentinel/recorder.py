import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from src.schemas.market_data import Bar

class SilverRecorder:
    """
    Recorder for the Silver Layer.
    Persists validated Bar objects to Parquet files organized by symbol/year/month.
    """
    def __init__(self, base_dir: str = "data/silver"):
        self.base_dir = Path(base_dir)
        self.ohlcv_dir = self.base_dir / "ohlcv"
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)

    def save_bars(self, bars: List[Bar]):
        """
        Save a list of Bar objects to Parquet.
        Partitions by Symbol -> Year -> Month.
        """
        if not bars:
            return

        # Convert to DataFrame
        data = [bar.model_dump() for bar in bars]
        df = pd.DataFrame(data)

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # We process by symbol to ensure correct partitioning
        for symbol, group in df.groupby('symbol'):
            self._save_symbol_group(symbol, group)

    def _save_symbol_group(self, symbol: str, df: pd.DataFrame):
        """
        Save dataframe for a specific symbol.
        """
        # Add partition columns if needed, but directory structure handles year/month
        # We'll use the first timestamp to determine the partition, or iterate if multiple months
        # For simplicity, let's just append to a day-level file or month-level file.
        # The plan says: {symbol}/{year}/{month}/{date}.parquet
        
        # Group by date to handle multiple days in one batch
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        for date_str, day_group in df.groupby('date_str'):
            ts = pd.to_datetime(date_str)
            year = str(ts.year)
            month = f"{ts.month:02d}"
            
            target_dir = self.ohlcv_dir / symbol / year / month
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = target_dir / f"{date_str}.parquet"
            
            # If file exists, we might need to append or overwrite. 
            # For data idempotency, let's overwrite for now or read-append-dedup
            # Simple approach: Overwrite (Last Writer Wins for the day)
            # Or better: Read existing, append new, drop duplicates, write back.
            
            if file_path.exists():
                try:
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, day_group])
                    # Dedup based on timestamp and symbol
                    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
                except Exception as e:
                    # Log error, maybe backup corrupt file?
                    print(f"Error reading existing parquet {file_path}: {e}")
                    combined_df = day_group
            else:
                combined_df = day_group

            # Write back
            # Ensure proper types for parquet
            # Drop the temp column
            if 'date_str' in combined_df.columns:
                combined_df = combined_df.drop(columns=['date_str'])
            
            combined_df.to_parquet(file_path, index=False)
