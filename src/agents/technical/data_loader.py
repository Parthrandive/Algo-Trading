import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
from src.agents.technical.nsemine_fetcher import NseMineFetcher

logger = logging.getLogger(__name__)

MACRO_COLUMNS = [
    "CPI",
    "WPI",
    "IIP",
    "FII_FLOW",
    "DII_FLOW",
    "FX_RESERVES",
    "INDIA_US_10Y_SPREAD",
    "RBI_BULLETIN",
    "REPO_RATE",
    "US_10Y",
]

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

    def _is_postgres_engine(self) -> bool:
        url = str(self.engine.url).lower()
        return url.startswith("postgresql")

    def _augment_with_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        As-of join macro indicators onto market bars.
        Keeps existing columns and appends/updates known macro columns.
        """
        if df.empty or not self._is_postgres_engine():
            return df
        if "timestamp" not in df.columns:
            return df

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if out.empty:
            return out

        min_ts = out["timestamp"].min()
        max_ts = out["timestamp"].max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            return out

        try:
            macro_query = text(
                """
                SELECT indicator_name, timestamp, value
                FROM macro_indicators
                WHERE timestamp <= :max_ts
                  AND timestamp >= :min_ts
                ORDER BY timestamp ASC
                """
            )
            # Pull a wider window so slower macro series still align for recent bars.
            macro_df = pd.read_sql(
                macro_query,
                self.engine,
                params={
                    "min_ts": (min_ts - pd.Timedelta(days=400)).to_pydatetime(),
                    "max_ts": max_ts.to_pydatetime(),
                },
            )
        except Exception as exc:
            logger.warning("Macro augmentation query failed: %s", exc)
            return out

        if macro_df.empty:
            return out

        macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"], utc=True, errors="coerce")
        macro_df["value"] = pd.to_numeric(macro_df["value"], errors="coerce")
        macro_df = macro_df.dropna(subset=["indicator_name", "timestamp", "value"])
        if macro_df.empty:
            return out

        pivot = (
            macro_df.pivot_table(
                index="timestamp",
                columns="indicator_name",
                values="value",
                aggfunc="last",
            )
            .sort_index()
            .reset_index()
        )
        if pivot.empty:
            return out

        merged = pd.merge_asof(
            out.sort_values("timestamp"),
            pivot.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            suffixes=("", "_macro"),
        )

        for col in MACRO_COLUMNS:
            source_col = f"{col}_macro" if f"{col}_macro" in merged.columns else col
            if source_col not in merged.columns:
                continue
            if col not in out.columns:
                out[col] = pd.NA
            out[col] = out[col].where(pd.notna(out[col]), merged[source_col])

        return out.sort_values("timestamp").reset_index(drop=True)

    def load_historical_bars(
        self,
        symbol: str,
        limit: Optional[int] = None,
        use_nse_fallback: bool = False,
        min_fallback_rows: int = 100,
        interval: str = "1d",
        include_macro: bool = True,
    ) -> pd.DataFrame:
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
            if include_macro:
                df = self._augment_with_macro(df)
            
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
                    if include_macro:
                        nse_df = self._augment_with_macro(nse_df)
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
                nse_df = self.load_from_nse(
                    symbol=symbol, 
                    from_date=from_dt.strftime("%d-%m-%Y"), 
                    to_date=to_dt.strftime("%d-%m-%Y"),
                    interval=interval
                )
                if include_macro:
                    nse_df = self._augment_with_macro(nse_df)
                return nse_df
            raise RuntimeError(f"Failed to load historical bars: {str(e)}")
