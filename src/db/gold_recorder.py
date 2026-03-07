import logging
from typing import List, Dict, Any
import pandas as pd

from src.db.connection import get_engine
from src.schemas.preprocessing_data import TransformOutput

logger = logging.getLogger(__name__)

class GoldRecorder:
    """
    Handles persisting processed "Gold" tier features to the database.
    Because Gold features have dynamic columns based on configuration, 
    we write them to a dedicated table using pandas to_sql.
    """
    def __init__(self, table_name: str = "gold_features"):
        self.engine = get_engine()
        self.table_name = table_name

    def save_features(self, df: pd.DataFrame, snapshot_id: str) -> None:
        """
        Appends the processed features to the gold_features table.
        """
        if df.empty:
            logger.info("No gold features to save.")
            return

        # Add metadata column for lineage
        df_to_save = df.copy()
        df_to_save['snapshot_id'] = snapshot_id

        try:
            from sqlalchemy import inspect, text
            
            # Check if table exists and handle dynamic schema evolution
            inspector = inspect(self.engine)
            if inspector.has_table(self.table_name):
                existing_columns = [col['name'] for col in inspector.get_columns(self.table_name)]
                missing_columns = [c for c in df_to_save.columns if c not in existing_columns]
                
                if missing_columns:
                    logger.info(f"Adding new dynamic columns to {self.table_name}: {missing_columns}")
                    with self.engine.begin() as conn:
                        for col in missing_columns:
                            # Map basic pandas types to postgres types
                            pg_type = "TEXT"
                            if pd.api.types.is_float_dtype(df_to_save[col]):
                                pg_type = "DOUBLE PRECISION"
                            elif pd.api.types.is_integer_dtype(df_to_save[col]):
                                pg_type = "BIGINT"
                            elif pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
                                pg_type = "TIMESTAMP WITH TIME ZONE"
                                
                            conn.execute(text(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col}" {pg_type}'))
            
            # We use 'append' to add to the existing table safely now that schema is evolved.
            df_to_save.to_sql(
                self.table_name,
                self.engine,
                if_exists="append",
                index=False
            )
            logger.info(f"Saved {len(df_to_save)} rows to {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to save gold features to DB: {e}")
            raise

    def save_features_on_demand(self, df: pd.DataFrame, snapshot_id: str, symbol: str) -> None:
        """
        Saves on-demand features for a specific symbol by replacing existing rows for the same 
        symbol and timestamps to prevent duplicates.
        """
        if df.empty:
            logger.info(f"No gold features to save for {symbol}.")
            return

        # Add metadata column for lineage
        df_to_save = df.copy()
        df_to_save['snapshot_id'] = snapshot_id

        try:
            from sqlalchemy import inspect, text
            
            # 1. Check if table exists and handle dynamic schema evolution
            inspector = inspect(self.engine)
            if inspector.has_table(self.table_name):
                existing_columns = [col['name'] for col in inspector.get_columns(self.table_name)]
                missing_columns = [c for c in df_to_save.columns if c not in existing_columns]
                
                if missing_columns:
                    logger.info(f"Adding new dynamic columns to {self.table_name}: {missing_columns}")
                    with self.engine.begin() as conn:
                        for col in missing_columns:
                            # Map basic pandas types to postgres types
                            pg_type = "TEXT"
                            if pd.api.types.is_float_dtype(df_to_save[col]):
                                pg_type = "DOUBLE PRECISION"
                            elif pd.api.types.is_integer_dtype(df_to_save[col]):
                                pg_type = "BIGINT"
                            elif pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
                                pg_type = "TIMESTAMP WITH TIME ZONE"
                                
                            conn.execute(text(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col}" {pg_type}'))
                            
                # 2. Delete existing rows for this symbol and these timestamps to perform an "upsert"
                if 'timestamp' in df_to_save.columns:
                    min_ts = df_to_save['timestamp'].min()
                    max_ts = df_to_save['timestamp'].max()
                    
                    with self.engine.begin() as conn:
                        query = text(f"DELETE FROM {self.table_name} WHERE symbol = :sym AND timestamp >= :min_ts AND timestamp <= :max_ts")
                        conn.execute(query, {"sym": symbol, "min_ts": min_ts, "max_ts": max_ts})
            
            # 3. We use 'append' to add the fresh rows safely now that overlapping old rows are deleted
            df_to_save.to_sql(
                self.table_name,
                self.engine,
                if_exists="append",
                index=False
            )
            logger.info(f"Upserted {len(df_to_save)} rows for {symbol} to {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to upsert gold features to DB for {symbol}: {e}")
            raise
