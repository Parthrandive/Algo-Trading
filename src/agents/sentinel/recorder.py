import pandas as pd
from pathlib import Path
from typing import List
import logging

from src.schemas.market_data import Bar, CorporateAction
from src.utils.validation import StreamMonotonicityChecker

logger = logging.getLogger(__name__)

class SilverRecorder:
    """
    Recorder for the Silver Layer.
    Persists validated Bar objects to Parquet files organized by symbol/year/month.
    Handles monotonicity checks and quarantines out-of-order data.
    """
    def __init__(self, base_dir: str = "data/silver", quarantine_dir: str = "data/quarantine"):
        self.base_dir = Path(base_dir)
        self.ohlcv_dir = self.base_dir / "ohlcv"
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        
        self.corp_action_dir = self.base_dir / "corporate_actions"
        self.corp_action_dir.mkdir(parents=True, exist_ok=True)
        
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.monotonicity_checker = StreamMonotonicityChecker()

    def save_bars(self, bars: List[Bar]):
        """
        Save a list of Bar objects to Parquet.
        Partitions by Symbol -> Year -> Month.
        """
        if not bars:
            return

        valid_bars = []
        quarantined_bars = []

        # Sort bars by timestamp to ensure we process in order within the batch
        # But we still check against state for inter-batch monotonicity
        sorted_bars = sorted(bars, key=lambda b: b.timestamp)

        for bar in sorted_bars:
            if self.monotonicity_checker.check(bar.symbol, bar.timestamp):
                valid_bars.append(bar)
            else:
                logger.warning(f"Quarantining out-of-order bar for {bar.symbol} at {bar.timestamp}")
                quarantined_bars.append(bar)

        self._persist_valid_bars(valid_bars)
        self._persist_quarantined_bars(quarantined_bars)

    def save_corporate_actions(self, actions: List[CorporateAction]):
        """
        Save a list of CorporateAction objects to Parquet.
        Partitions by Symbol -> Year -> Month based on ex_date.
        """
        if not actions:
            return

        # Convert to DataFrame
        data = [action.model_dump() for action in actions]
        df = pd.DataFrame(data)

        # We'll use ex_date for partitioning corporate actions instead of timestamp
        if 'ex_date' in df.columns:
            df['ex_date'] = pd.to_datetime(df['ex_date'])
        
        # We process by symbol to ensure correct partitioning
        for symbol, group in df.groupby('symbol'):
            self._save_symbol_group_corp(symbol, group, self.corp_action_dir)

    def _persist_valid_bars(self, bars: List[Bar]):
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
            self._save_symbol_group(symbol, group, self.ohlcv_dir)

    def _persist_quarantined_bars(self, bars: List[Bar]):
        if not bars:
            return
            
        # Convert to DataFrame
        data = [bar.model_dump() for bar in bars]
        df = pd.DataFrame(data)

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        for symbol, group in df.groupby('symbol'):
            self._save_symbol_group(symbol, group, self.quarantine_dir)

    def _save_symbol_group(self, symbol: str, df: pd.DataFrame, base_output_dir: Path):
        """
        Save dataframe for a specific symbol to the target directory.
        """
        if df.empty:
            return
            
        # Group by date to handle multiple days in one batch
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        for date_str, day_group in df.groupby('date_str'):
            ts = pd.to_datetime(date_str)
            year = str(ts.year)
            month = f"{ts.month:02d}"
            
            target_dir = base_output_dir / symbol / year / month
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = target_dir / f"{date_str}.parquet"
            
            if file_path.exists():
                try:
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, day_group])
                    # Dedup based on timestamp and symbol
                    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
                except Exception as e:
                    logger.error(f"Error reading existing parquet {file_path}: {e}")
                    combined_df = day_group
            else:
                combined_df = day_group

            # Write back
            if 'date_str' in combined_df.columns:
                combined_df = combined_df.drop(columns=['date_str'])
            
            combined_df.to_parquet(file_path, index=False)
            
    def _save_symbol_group_corp(self, symbol: str, df: pd.DataFrame, base_output_dir: Path):
        """
        Save dataframe for corporate actions based on ex_date.
        """
        if df.empty:
            return
            
        # Group by date to handle multiple days in one batch
        df['date_str'] = df['ex_date'].dt.strftime('%Y-%m-%d')
        
        for date_str, day_group in df.groupby('date_str'):
            ts = pd.to_datetime(date_str)
            year = str(ts.year)
            month = f"{ts.month:02d}"
            
            target_dir = base_output_dir / symbol / year / month
            target_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = target_dir / f"{date_str}.parquet"
            
            if file_path.exists():
                try:
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, day_group])
                    # Dedup based on date and type
                    combined_df = combined_df.drop_duplicates(subset=['ex_date', 'action_type', 'symbol'], keep='last')
                except Exception as e:
                    logger.error(f"Error reading existing parquet {file_path}: {e}")
                    combined_df = day_group
            else:
                combined_df = day_group

            # Write back
            if 'date_str' in combined_df.columns:
                combined_df = combined_df.drop(columns=['date_str'])
            
            combined_df.to_parquet(file_path, index=False)
