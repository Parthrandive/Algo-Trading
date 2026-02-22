"""
Day 13 End-to-End Completeness Validation
Checks Bronze and Silver folders for OHLCV and corporate-action outputs.
"""

import json
from pathlib import Path
import sys
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_CORP_ACTION_TYPES = {"dividend", "split", "bonus", "rights"}


def _read_parquet_rows(files: list[Path]) -> int:
    total_rows = 0
    for pq_file in files:
        df = pd.read_parquet(pq_file)
        total_rows += len(df)
    return total_rows


def main():
    print("Running Completeness Checks...")

    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    bronze_dir = data_dir / "bronze"
    silver_dir = data_dir / "silver"
    ohlcv_dir = silver_dir / "ohlcv"
    corp_dir = silver_dir / "corporate_actions"

    if not bronze_dir.exists() or not silver_dir.exists():
        print("ERROR: E2E test data directories not found. Run test_day13_e2e.py first.")
        sys.exit(1)

    # Check Bronze
    bronze_events = 0
    bronze_schemas = Counter()
    symbols_found = set()
    for jsonl_file in bronze_dir.rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                symbols_found.add(event.get("symbol"))
                schema_id = event.get("schema_id")
                if schema_id:
                    bronze_schemas[schema_id] += 1
                bronze_events += 1

    print(f"Bronze: Found {bronze_events} events.")
    print(f"Bronze: Found symbols: {symbols_found}")
    print(f"Bronze: Schema counts: {dict(bronze_schemas)}")
    assert len(symbols_found) > 0, "No symbols found in bronze data."
    assert bronze_schemas.get("market.bar.v1", 0) > 0, "No Bronze market.bar.v1 events found."
    assert bronze_schemas.get("market.corporate_action.v1", 0) > 0, "No Bronze market.corporate_action.v1 events found."

    # Check Silver OHLCV
    ohlcv_files = list(ohlcv_dir.rglob("*.parquet"))
    ohlcv_rows = 0
    for pq_file in ohlcv_files:
        df = pd.read_parquet(pq_file)
        ohlcv_rows += len(df)
        assert "symbol" in df.columns
        assert "timestamp" in df.columns
        assert "source_type" in df.columns
        assert "ingestion_timestamp_utc" in df.columns
        assert "open" in df.columns

    print(f"Silver OHLCV: Found {len(ohlcv_files)} parquet files with {ohlcv_rows} rows.")
    assert ohlcv_rows > 0, "No rows found in Silver OHLCV data."

    # Check Silver Corporate Actions
    corp_files = list(corp_dir.rglob("*.parquet"))
    corp_rows = 0
    corp_type_counter = Counter()
    for pq_file in corp_files:
        df = pd.read_parquet(pq_file)
        corp_rows += len(df)
        assert "symbol" in df.columns
        assert "action_type" in df.columns
        assert "ex_date" in df.columns
        assert "source_type" in df.columns
        assert "ingestion_timestamp_utc" in df.columns
        corp_type_counter.update(str(action_type).lower() for action_type in df["action_type"].dropna().tolist())

    print(f"Silver Corporate Actions: Found {len(corp_files)} parquet files with {corp_rows} rows.")
    print(f"Silver Corporate Actions: Type counts {dict(corp_type_counter)}")
    assert corp_rows > 0, "No rows found in Silver corporate actions data."

    missing_types = REQUIRED_CORP_ACTION_TYPES - set(corp_type_counter)
    assert not missing_types, f"Missing corporate action types in Silver data: {sorted(missing_types)}"

    print("Completeness checks passed.")


if __name__ == "__main__":
    main()
