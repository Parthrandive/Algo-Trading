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
        Loads MacroIndicator data, validates against schema v1.1, and returns a DataFrame.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        records: List[MacroIndicator] = []
        macro_adapter = TypeAdapter(list[MacroIndicator])
        
        # Macro inputs can be provided as parquet (silver outputs) or json fixtures.
        if path.is_dir():
            files = sorted([*path.glob("**/*.parquet"), *path.glob("**/*.json")])
        else:
            files = [path]

        for file_path in files:
            try:
                if file_path.suffix.lower() == ".parquet":
                    df_part = pd.read_parquet(file_path)
                elif file_path.suffix.lower() == ".json":
                    df_part = self._read_macro_json(file_path)
                else:
                    continue

                if not df_part.empty:
                    # Convert to records for schema validation
                    payload = df_part.to_dict(orient="records")
                    validated_batch = macro_adapter.validate_python(payload)
                    for record in validated_batch:
                        if not self._is_schema_allowed("macro.indicator", record.schema_version):
                            raise SchemaVersionError(
                                f"Schema version {record.schema_version} not accepted by contract. "
                                f"Allowed: {self._accepted_schemas}"
                            )
                        records.append(record)
                        
            except ValidationError as e:
                raise ValueError(f"Validation failed for {file_path}: {e}")
            except Exception as e:
                raise IOError(f"Failed to read {file_path}: {e}")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame([r.model_dump() for r in records])
        # Force snapshot ID per Section 5.5
        df["dataset_snapshot_id"] = snapshot_id
        return df

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

    def load(self, source_path: str, snapshot_id: str) -> pd.DataFrame:
        """
        Loads market data (historical Bar and real-time Tick), validates against schema,
        and returns a unified DataFrame.

        Tick rows are converted to synthetic bars (interval="tick") so downstream
        transforms can use the same interface (`close`, `volume`, timestamps).
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
            
        try:
            # Silver recorder outputs parquet
            if path.is_dir():
                # Read market parquet files in the directory tree.
                parquet_files = self._select_market_files(path)
                if not parquet_files:
                    return pd.DataFrame()
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            else:
                df = pd.read_parquet(path)
        except Exception as e:
            raise IOError(f"Failed to load Market data from {source_path}: {e}")
            
        if df.empty:
            return df
            
        records: List[Bar] = []

        dict_records = df.to_dict(orient="records")
        invalid_messages: List[str] = []

        # Per-row dual-parse strategy:
        # 1) Try canonical Bar
        # 2) Fallback to Tick and convert to synthetic Bar
        for idx, row in enumerate(dict_records):
            try:
                bar = Bar.model_validate(row)
                if not self._is_schema_allowed("market.bar", bar.schema_version):
                    raise SchemaVersionError(
                        f"Schema version {bar.schema_version} not accepted for Bar. "
                        f"Allowed: {self._accepted_schemas}"
                    )
                records.append(bar)
                continue
            except ValidationError as bar_error:
                first_error = bar_error

            try:
                tick = Tick.model_validate(row)
                if not self._is_schema_allowed("market.tick", tick.schema_version):
                    raise SchemaVersionError(
                        f"Schema version {tick.schema_version} not accepted for Tick. "
                        f"Allowed: {self._accepted_schemas}"
                    )
                records.append(self._tick_to_bar(tick))
            except ValidationError as tick_error:
                invalid_messages.append(
                    f"row={idx} could not be parsed as Bar or Tick: "
                    f"bar_error={first_error.errors()} tick_error={tick_error.errors()}"
                )

        if invalid_messages:
            sample = " | ".join(invalid_messages[:3])
            raise ValueError(f"Market Validation failed for {len(invalid_messages)} row(s): {sample}")

        validated_df = pd.DataFrame([r.model_dump() for r in records])
        validated_df["dataset_snapshot_id"] = snapshot_id
        
        return validated_df
