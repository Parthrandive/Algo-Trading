import json
from pathlib import Path
from typing import List, Generator

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.schemas.macro_data import MacroIndicator
from src.schemas.market_data import Bar
from src.schemas.preprocessing_data import PreprocessingContract

class SchemaVersionError(Exception):
    pass

class PreprocessingLoader:
    def __init__(self, contract_path: str = "configs/preprocessing_contract_v1.json"):
        with open(contract_path, "r") as f:
            self.contract = PreprocessingContract.model_validate_json(f.read())
            
        self._accepted_schemas = self.contract.accepted_input_schemas

class MacroLoader(PreprocessingLoader):
    def load(self, source_path: str, snapshot_id: str) -> pd.DataFrame:
        """
        Loads MacroIndicator data, validates against schema v1.1, and returns a DataFrame.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        records: List[MacroIndicator] = []
        macro_adapter = TypeAdapter(list[MacroIndicator])
        
        # Load logic defaults to json arrays matching Phase 1 sync gate specifications
        if path.is_dir():
            files = list(path.glob("**/*.json"))
        else:
            files = [path]

        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    payload = json.load(f)
                    
                    if not isinstance(payload, list):
                        payload = [payload]
                        
                    validated_batch = macro_adapter.validate_python(payload)
                    for record in validated_batch:
                        if "macro.indicator." + record.schema_version not in self._accepted_schemas:
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
    def load(self, source_path: str, snapshot_id: str) -> pd.DataFrame:
        """
        Loads Bar data (OHLCV), validates against schema v1.0, and returns a DataFrame.
        Supports reading raw parquet datasets from the Silver layer.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
            
        try:
            # Silver recorder outputs parquet
            if path.is_dir():
                # Read all parquet files in the directory tree (e.g. symbol/year/month/*.parquet)
                parquet_files = list(path.glob("**/*.parquet"))
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
        bar_adapter = TypeAdapter(list[Bar])
        
        # Convert df back to dict records to run through pydantic validation
        # Data loss or corruption at Silver level must trigger fail-fast
        try:
            # Convert timestamps correctly for pydantic
            dict_records = df.to_dict(orient="records")
            validated_batch = bar_adapter.validate_python(dict_records)
            
            for record in validated_batch:
                if "market.bar." + record.schema_version not in self._accepted_schemas:
                    raise SchemaVersionError(
                        f"Schema version {record.schema_version} not accepted. "
                        f"Allowed: {self._accepted_schemas}"
                    )
                records.append(record)
        except ValidationError as e:
            raise ValueError(f"Market Validation failed: {e}")
            
        validated_df = pd.DataFrame([r.model_dump() for r in records])
        validated_df["dataset_snapshot_id"] = snapshot_id
        
        return validated_df
