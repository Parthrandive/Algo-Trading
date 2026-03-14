import logging
import pandas as pd
from datetime import datetime
from nsemine import historical

logger = logging.getLogger(__name__)

class NseMineFetcher:
    """
    Wrapper around the nsemine Python package to fetch historical
    stock and index data efficiently.
    """

    # Map our interval strings to nsemine's API expectations
    INTERVAL_MAP = {
        "1m": 1, 
        "3m": 3, 
        "5m": 5, 
        "10m": 10,
        "15m": 15, 
        "30m": 30, 
        "1h": 60,
        "1d": "D", 
        "1w": "W", 
        "1M": "M"
    }

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """Strip '.NS' or other suffixes for NSE API."""
        if symbol.endswith(".NS"):
            return symbol[:-3]
        return symbol

    @staticmethod
    def fetch_historical(
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical bars directly from NSE.
        
        Args:
            symbol (str): Stock symbol (e.g. 'INFY.NS' or 'TCS')
            start_date (str): 'DD-MM-YYYY' format
            end_date (str): 'DD-MM-YYYY' format
            interval (str): '1m', '5m', '15m', '1h', '1d', '1w', '1M'
            
        Returns:
            pd.DataFrame: OHLCV DataFrame matching our internal schema,
                          or empty DataFrame if fetch fails.
        """
        clean_sym = NseMineFetcher._clean_symbol(symbol)
        
        try:
            start_dt = datetime.strptime(start_date, "%d-%m-%Y")
        except ValueError:
            try:
                # Fallback for YYYY-MM-DD
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid start_date format: {start_date}. Use DD-MM-YYYY or YYYY-MM-DD")
                return pd.DataFrame()
                
        try:
            end_dt = datetime.strptime(end_date, "%d-%m-%Y")
        except ValueError:
            try:
                # Fallback for YYYY-MM-DD
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid end_date format: {end_date}. Use DD-MM-YYYY or YYYY-MM-DD")
                return pd.DataFrame()

        nse_interval = NseMineFetcher.INTERVAL_MAP.get(interval, "D")
        
        try:
            # Note: nsemine returns data in a DataFrame with generic names
            df = historical.get_stock_historical_data(
                clean_sym, 
                start_dt, 
                end_dt, 
                interval=nse_interval
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned by nsemine for {clean_sym} from {start_date} to {end_date}")
                return pd.DataFrame()
                
            # Normalize column names to our schema
            schema_map = {
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            }
            
            # Map columns
            df = df.rename(columns=schema_map)
            
            # Ensure timestamp is a true datetime object and tz-aware
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # nsemine may return naive datetimes for IST, so we localize then convert to UTC
            if df['timestamp'].dt.tz is None:
                # Standard NSE time is IST
                df['timestamp'] = df['timestamp'].dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
                
            # Add symbol literal back
            df['symbol'] = symbol
            
            # Keep only the required columns
            required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in df.columns]
            df = df[available_cols]
            
            # Sort chronologically
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"nsemine fetch exception for {clean_sym}: {e}")
            return pd.DataFrame()
