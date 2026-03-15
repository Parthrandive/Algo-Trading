import json
import logging
import re
from pathlib import Path

import pandas as pd

from src.schemas.market_data import Bar, Tick
from src.schemas.preprocessing_data import PreprocessingContract

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _clean_macro_frame(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        required_columns = {"indicator_name", "timestamp", "value"}
        missing = [column for column in required_columns if column not in cleaned.columns]
        if missing:
            raise ValueError(f"Macro payload is missing required columns: {sorted(missing)}")

        cleaned["indicator_name"] = (
            cleaned["indicator_name"]
            .astype(str)
            .str.strip()
            .str.replace("MacroIndicatorType.", "", regex=False)
            .str.upper()
            .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
        )
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
        cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
        cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)

        before = len(cleaned)
        cleaned = cleaned.dropna(subset=["indicator_name", "timestamp", "value"])

        sort_columns = ["indicator_name", "timestamp"]
        if "ingestion_timestamp_utc" in cleaned.columns:
            cleaned["ingestion_timestamp_utc"] = pd.to_datetime(
                cleaned["ingestion_timestamp_utc"], utc=True, errors="coerce"
            )
            sort_columns.append("ingestion_timestamp_utc")

        cleaned = cleaned.sort_values(sort_columns).drop_duplicates(
            subset=["indicator_name", "timestamp"],
            keep="last",
        )
        cleaned = cleaned.reset_index(drop=True)

        dropped = before - len(cleaned)
        if dropped:
            logger.warning("MacroLoader dropped %d malformed/duplicate row(s) during cleaning.", dropped)
        return cleaned

    def load(self, source_path: str, snapshot_id: str) -> pd.DataFrame:
        """
        Loads MacroIndicator data. When source_path is 'db_virtual', reads from PostgreSQL.
        Otherwise, reads from the given JSON file path (for tests and legacy usage).
        """
        if source_path != "db_virtual":
            # File-based loading (used by tests and legacy scripts)
            source = Path(source_path)
            if source.is_file():
                df = self._read_macro_json(source)
            elif source.is_dir():
                frames = [self._read_macro_json(f) for f in sorted(source.glob("**/*.json"))]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            # DB-based loading (production)
            from src.db.connection import get_engine
            engine = get_engine()
            df = pd.read_sql("SELECT * FROM macro_indicators", engine)
        
        if df.empty:
            return df

        df = self._clean_macro_frame(df)

        if "schema_version" in df.columns:
            for version in df["schema_version"].dropna().unique():
                if not self._is_schema_allowed("macro.indicator", version):
                    raise SchemaVersionError(
                        f"Schema version {version} not accepted by contract. "
                        f"Allowed: {self._accepted_schemas}"
                    )

        if "dataset_snapshot_id" not in df.columns:
            df["dataset_snapshot_id"] = snapshot_id
        return df

class TextLoader(PreprocessingLoader):
    def load(self, source_path: str, snapshot_id: str, symbol: str = None) -> pd.DataFrame:
        """
        Loads Textual data. When source_path is 'db_virtual', reads from PostgreSQL.
        Otherwise returns empty DataFrame (tests don't use textual data by default).
        """
        if source_path != "db_virtual":
            return pd.DataFrame()
            
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
        Loads market data. When source_path is 'db_virtual', reads from PostgreSQL.
        Otherwise, reads from the given Parquet file/directory path (for tests and legacy usage).
        """
        if source_path != "db_virtual":
            # File-based loading (used by tests and legacy scripts)
            source = Path(source_path)
            if source.is_file():
                df = pd.read_parquet(source)
            elif source.is_dir():
                files = self._select_market_files(source)
                frames = [pd.read_parquet(f) for f in files]
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df = pd.DataFrame()
            
            # Handle tick files: if 'price' column exists but no 'close', it's a tick file
            if not df.empty and 'price' in df.columns and 'close' not in df.columns:
                df['interval'] = 'tick'
                df['open'] = df['price']
                df['high'] = df['price']
                df['low'] = df['price']
                df['close'] = df['price']
            
            if not df.empty:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                if 'schema_version' in df.columns:
                    for version in df['schema_version'].unique():
                        if not (self._is_schema_allowed("market.bar", version) or self._is_schema_allowed("market.tick", version)):
                            raise SchemaVersionError(f"Schema version {version} not accepted by contract.")
                if 'dataset_snapshot_id' not in df.columns:
                    df["dataset_snapshot_id"] = snapshot_id
            return df
        
        # DB-based loading (production)
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
