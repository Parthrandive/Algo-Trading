import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
from src.agents.technical.nsemine_fetcher import NseMineFetcher

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads historical Silver/Gold tier OHLCV bars for the Technical Agent.
    """
    
    def __init__(self, db_url: str):
        """
        Initialize the DataLoader with a database connection.
        
        Args:
            db_url (str): SQLAlchemy database connection string.
        """
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def load_from_nse(self, symbol: str, from_date: str, to_date: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch historical blocks purely from NSE via nsemine."""
        return NseMineFetcher.fetch_historical(symbol, from_date, to_date, interval)

    def load_historical_bars(self, symbol: str, limit: Optional[int] = None, use_nse_fallback: bool = False, min_fallback_rows: int = 100, interval: str = "1d") -> pd.DataFrame:
        """
        Load historical OHLCV data for a specific symbol from the sentinel_db.ohlcv_bars table.
        If use_nse_fallback is True and the DB returns fewer than min_fallback_rows,
        it will attempt to fetch the last 1 year of data from the NSE natively.
        
        Args:
            symbol (str): The stock symbol (e.g., 'TATASTEEL.NS')
            limit (int, optional): Maximum number of rows to retrieve.
            use_nse_fallback (bool, optional): Allow hitting NSE if local data is sparse.
            min_fallback_rows (int, optional): Threshold for triggering NSE fallback.
            interval (str, optional): The candle interval, like '1d', '1h'. Defaults to '1d'.
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV features.
        """
        if not symbol:
            raise ValueError("`symbol` must be a non-empty string.")

        query = """
            SELECT *
            FROM ohlcv_bars
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY timestamp ASC
        """
        params = {"symbol": symbol, "interval": interval}

        if limit is not None:
            if limit <= 0:
                raise ValueError("`limit` must be a positive integer when provided.")
            query += " LIMIT :limit"
            params["limit"] = int(limit)

        try:
            df = pd.read_sql(text(query), self.engine, params=params)
            
            # Check for fallback
            if use_nse_fallback and len(df) < min_fallback_rows:
                logger.info(f"DB returned {len(df)} rows, less than {min_fallback_rows}. Trying NSE fallback...")
                import datetime as dt
                to_dt = dt.datetime.now()
                from_dt = to_dt - dt.timedelta(days=365)
                nse_df = self.load_from_nse(
                    symbol=symbol, 
                    from_date=from_dt.strftime("%d-%m-%Y"), 
                    to_date=to_dt.strftime("%d-%m-%Y"),
                    interval=interval
                )
                if not nse_df.empty:
                    if limit is not None:
                        return nse_df.tail(limit).copy()
                    return nse_df
            
            return df
        except Exception as e:
            logger.error(f"DB Load failed: {e}. Trying NSE if permitted.")
            if use_nse_fallback:
                import datetime as dt
                to_dt = dt.datetime.now()
                from_dt = to_dt - dt.timedelta(days=365)
                return self.load_from_nse(
                    symbol=symbol, 
                    from_date=from_dt.strftime("%d-%m-%Y"), 
                    to_date=to_dt.strftime("%d-%m-%Y"),
                    interval=interval
                )
            raise RuntimeError(f"Failed to load historical bars: {str(e)}")
