import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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

    def load_historical_bars(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load historical OHLCV data for a specific symbol from the sentinel_db.ohlcv_bars table.
        
        Args:
            symbol (str): The stock symbol (e.g., 'TATASTEEL.NS')
            limit (int, optional): Maximum number of rows to retrieve.
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV features.
        """
        if not symbol:
            raise ValueError("`symbol` must be a non-empty string.")

        query = """
            SELECT *
            FROM sentinel_db.ohlcv_bars
            WHERE symbol = :symbol
            ORDER BY timestamp ASC
        """
        params = {"symbol": symbol}

        if limit is not None:
            if limit <= 0:
                raise ValueError("`limit` must be a positive integer when provided.")
            query += " LIMIT :limit"
            params["limit"] = int(limit)

        try:
            # Using pandas read_sql to execute query and return DataFrame directly
            df = pd.read_sql(text(query), self.engine, params=params)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load historical bars: {str(e)}")
