import json
import re
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.schemas.macro_data import MacroIndicator
from src.schemas.market_data import Bar, Tick
from src.schemas.preprocessing_data import PreprocessingContract

class SchemaVersionError(Exception):
    pass

class PreprocessingLoader:
    def __init__(self, contract_path: str = "configs/preprocessing_contract_v1.json"):
        with open(contract_path, "r") as f:
            self.contract = PreprocessingContract.model_validate_json(f.read())
            
        self._accepted_schemas = self.contract.accepted_input_schemas
        self._accepted_schemas_canonical = {
            self._canonical_schema_id(schema_id) for schema_id in self._accepted_schemas
        }

    @staticmethod
    def _canonical_schema_id(schema_id: str) -> str:
        """
        Normalize schema IDs to a canonical form where the terminal version segment
        does not require a leading 'v' (e.g., macro.indicator.v1.1 == macro.indicator.1.1).
        """
        return re.sub(r"\.v(?=\d)", ".", schema_id)

    def _is_schema_allowed(self, prefix: str, schema_version: str) -> bool:
        schema_id = self._canonical_schema_id(f"{prefix}.{schema_version}")
        return schema_id in self._accepted_schemas_canonical

class MacroLoader(PreprocessingLoader):
    @staticmethod
    def _read_macro_json(file_path: Path) -> pd.DataFrame:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            # Support either a single record or an envelope with a records key.
            if "records" in payload and isinstance(payload["records"], list):
                return pd.DataFrame(payload["records"])
            return pd.DataFrame([payload])
        raise ValueError(f"Unsupported JSON payload type in {file_path}: {type(payload).__name__}")

    def load(self, source_path: str, snapshot_id: str) -> pd.DataFrame:
        """
        Loads MacroIndicator data directly from the PostgreSQL database,
        validates against schema v1.1, and returns a DataFrame.
        """
        from src.db.connection import get_engine
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM macro_indicators", engine)
        
        if df.empty:
            return df
            
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        if 'schema_version' in df.columns:
            for version in df['schema_version'].unique():
                if not self._is_schema_allowed("macro.indicator", version):
                    raise SchemaVersionError(
                        f"Schema version {version} not accepted by contract. "
                        f"Allowed: {self._accepted_schemas}"
                    )
                    
        df["dataset_snapshot_id"] = snapshot_id
        return df

class TextLoader(PreprocessingLoader):
    def load(self, source_path: str, snapshot_id: str, symbol: str = None) -> pd.DataFrame:
        """
        Loads Textual data (news, social sentiment) directly from the PostgreSQL database.
        Returns a DataFrame aggregated by day and symbol containing sentiment features.
        """
        from src.db.connection import get_engine
        engine = get_engine()
        
        # Load all textual items that successfully passed NLP/NER (sentiment attached)
        query = "SELECT * FROM text_items WHERE sentiment_score IS NOT NULL"
        if symbol:
            query += f" AND symbol = '{symbol}'"
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return pd.DataFrame()
            
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        # Group by symbol and day to provide daily sentiment features
        if 'symbol' in df.columns:
            # Drop items not bound to a specific symbol for Phase 1 stock modeling
            df = df.dropna(subset=['symbol'])
            df['date'] = df['timestamp'].dt.floor('D')
            
            agg_df = df.groupby(['symbol', 'date']).agg(
                sentiment_score=('sentiment_score', 'mean'),
                text_items_count=('source_id', 'count')
            ).reset_index()
            
            # Rename date back to timestamp for alignment merging
            agg_df = agg_df.rename(columns={'date': 'timestamp'})
            agg_df["dataset_snapshot_id"] = snapshot_id
            return agg_df
            
        return pd.DataFrame()

class MarketLoader(PreprocessingLoader):
    @staticmethod
    def _tick_to_bar(tick: Tick) -> Bar:
        """
        Convert a real-time Tick into a synthetic Bar row so existing
        preprocessing transforms can consume live data without schema drift.
        """
        return Bar(
            symbol=tick.symbol,
            exchange=tick.exchange,
            timestamp=tick.timestamp,
            source_type=tick.source_type,
            ingestion_timestamp_utc=tick.ingestion_timestamp_utc,
            ingestion_timestamp_ist=tick.ingestion_timestamp_ist,
            schema_version=tick.schema_version,
            quality_status=tick.quality_status,
            interval="tick",
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.volume,
        )

    @staticmethod
    def _select_market_files(base_path: Path) -> list[Path]:
        parquet_files = list(base_path.glob("**/*.parquet"))
        if not parquet_files:
            return []

        if base_path.name in {"ohlcv", "ticks"}:
            return parquet_files

        market_only = [p for p in parquet_files if ("ohlcv" in p.parts or "ticks" in p.parts)]
        return market_only or parquet_files

    def load(self, source_path: str, snapshot_id: str, symbol: str = None) -> pd.DataFrame:
        """
        Loads market data (Bar and Tick) directly from PostgreSQL, validates against schema,
        and returns a unified DataFrame.
        """
        from src.db.connection import get_engine
        engine = get_engine()
        
        where_clause = f" WHERE symbol = '{symbol}'" if symbol else ""
        bars_df = pd.read_sql(f"SELECT * FROM ohlcv_bars{where_clause}", engine)
        ticks_df = pd.read_sql(f"SELECT * FROM ticks{where_clause}", engine)
        
        if not ticks_df.empty:
            ticks_df['interval'] = 'tick'
            ticks_df['open'] = ticks_df['price']
            ticks_df['high'] = ticks_df['price']
            ticks_df['low'] = ticks_df['price']
            ticks_df['close'] = ticks_df['price']
            # Align columns
            valid_cols = [c for c in ticks_df.columns if c in bars_df.columns or c == 'interval']
            ticks_df = ticks_df[valid_cols]
            
        if bars_df.empty and ticks_df.empty:
            return pd.DataFrame()
            
        df = pd.concat([bars_df, ticks_df], ignore_index=True) if not ticks_df.empty else bars_df
            
        # Deduplicate to prevent realtime tick noise from duplicating historical bars
        if 'symbol' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['symbol', 'timestamp', 'interval']).drop_duplicates(
                subset=['symbol', 'timestamp'], keep='last'
            )
            
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        if 'schema_version' in df.columns:
            for version in df['schema_version'].unique():
                if not (self._is_schema_allowed("market.bar", version) or self._is_schema_allowed("market.tick", version)):
                    raise SchemaVersionError(f"Schema version {version} not accepted by contract.")
                    
        df["dataset_snapshot_id"] = snapshot_id
        return df
