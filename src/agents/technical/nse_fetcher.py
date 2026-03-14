import os
import time
import logging
import pandas as pd
from datetime import datetime, timezone
from nselib import capital_market

logger = logging.getLogger(__name__)

class NseFetcher:
    """
    Fetches raw historical market data directly from the NSE website using nselib.
    """
    
    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes nselib column names to fit our db schema."""
        # The columns often have whitespace, quotes or BOM artifacts from the NSE CSVs
        # e.g., 'ï»¿"Symbol"'
        
        # 1. Strip all funky characters to standardized names
        clean_map = {}
        for col in df.columns:
            c = col.replace('ï»¿', '').replace('"', '').strip()
            if c == 'Date':
                clean_map[col] = 'timestamp'
            elif c == 'OpenPrice':
                clean_map[col] = 'open'
            elif c == 'HighPrice':
                clean_map[col] = 'high'
            elif c == 'LowPrice':
                clean_map[col] = 'low'
            elif c == 'ClosePrice':
                clean_map[col] = 'close'
            elif c in ['TotalTradedQuantity', 'Volume']:
                clean_map[col] = 'volume'
            elif c == 'Symbol':
                clean_map[col] = 'symbol'
        
        df = df.rename(columns=clean_map)
        return df

    @staticmethod
    def _strip_ns(symbol: str) -> str:
        """Removes Yahoo Finance / DB specific suffixes."""
        if symbol.endswith('.NS'):
            return symbol[:-3]
        return symbol

    @classmethod
    def fetch_historical(cls, symbol: str, from_date: str, to_date: str, retries: int = 3) -> pd.DataFrame:
        """
        Fetch historical data from NSE and format it to our standard OHLCV.
        Dates must be in format 'DD-MM-YYYY'.
        """
        nse_symbol = cls._strip_ns(symbol)
        
        for attempt in range(retries):
            try:
                logger.info(f"NSE Fetch [{attempt+1}/{retries}] for {nse_symbol}: {from_date} to {to_date}...")
                df = capital_market.price_volume_and_deliverable_position_data(
                    symbol=nse_symbol,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if df is None or df.empty:
                    logger.warning(f"NSE returned empty response for {nse_symbol} from {from_date} to {to_date}")
                    return pd.DataFrame()
                
                df = cls._clean_columns(df)
                
                # Keep only what we need and convert types
                cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols_to_keep if c in df.columns]
                df = df[available_cols].copy()
                
                # Convert timestamps and make timezone-aware (NSE operates in IST, but we standardise on UTC)
                # First parse as naive datetime, then localize to Asia/Kolkata, then convert to UTC
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%b-%Y', cache=True)
                df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata').dt.tz_convert('UTC')
                
                # Convert price/volume columns to float/int
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        # nselib sometimes returns strings with commas like '1,234.56'
                        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                
                if 'volume' in df.columns:
                    df['volume'] = df['volume'].astype(str).str.replace(',', '').astype(float).astype(int)
                
                # Standardize symbol back to original requested format (with .NS)
                df.insert(1, 'symbol', symbol)
                
                # Ensure sorted chronologically
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Successfully fetched {len(df)} rows for {symbol} from NSE.")
                
                return df

            except Exception as e:
                logger.error(f"Error fetching NSE data for {symbol}: {e}")
                if attempt < retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch data from NSE for {symbol} after {retries} attempts.")
                    return pd.DataFrame()
