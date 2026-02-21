from typing import List, Optional
from datetime import datetime
import pandas as pd
from sqlalchemy import select, func, and_

from src.db.connection import get_engine
from src.db.models import OHLCVBar, CorporateActionDB

def get_latest_timestamp(symbol: str) -> Optional[datetime]:
    """
    Find the maximum timestamp for a symbol currently stored in the database.
    Useful for gap detection before fetching new data.
    Provides O(1) loop-up via the B-tree index, replacing the O(N) Parquet scan.
    """
    engine = get_engine()
    
    # Needs a session-less execute
    with engine.connect() as conn:
        stmt = select(func.max(OHLCVBar.timestamp)).where(OHLCVBar.symbol == symbol)
        result = conn.execute(stmt).scalar_one_or_none()
        return result

def get_bars(symbol: str, start: datetime, end: datetime, interval: str = "1h") -> pd.DataFrame:
    """
    Fetches bars for a specific symbol, interval, and time range.
    Returns a pandas DataFrame sorted by timestamp.
    """
    engine = get_engine()
    
    stmt = select(OHLCVBar).where(
        and_(
            OHLCVBar.symbol == symbol,
            OHLCVBar.interval == interval,
            OHLCVBar.timestamp >= start,
            OHLCVBar.timestamp <= end
        )
    ).order_by(OHLCVBar.timestamp.asc())
    
    # Read sql directly into pandas dataframe
    df = pd.read_sql(stmt, engine)
    
    # Ensure correct types
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df

def get_corporate_actions(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetches corporate actions for a specific symbol within an ex_date range.
    """
    engine = get_engine()
    
    stmt = select(CorporateActionDB).where(
        and_(
            CorporateActionDB.symbol == symbol,
            CorporateActionDB.ex_date >= start,
            CorporateActionDB.ex_date <= end
        )
    ).order_by(CorporateActionDB.ex_date.asc())
    
    df = pd.read_sql(stmt, engine)
    if not df.empty and 'ex_date' in df.columns:
        df['ex_date'] = pd.to_datetime(df['ex_date'])
        
    return df
