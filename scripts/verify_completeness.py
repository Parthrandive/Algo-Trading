"""
Day 13 End-to-End Completeness Validation
Checks bronze and silver folders for generated data.
"""

import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("Running Completeness Checks...")
    
    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    bronze_dir = data_dir / "bronze"
    silver_dir = data_dir / "silver"
    
    if not bronze_dir.exists() or not silver_dir.exists():
        print("ERROR: E2E test data directories not found. Run test_day13_e2e.py first.")
        sys.exit(1)
        
    # Check Bronze
    bronze_events = 0
    symbols_found = set()
    for jsonl_file in bronze_dir.rglob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                event = json.loads(line)
                symbols_found.add(event.get("symbol"))
                bronze_events += 1
                
    print(f"Bronze: Found {bronze_events} events.")
    print(f"Bronze: Found symbols: {symbols_found}")
    assert len(symbols_found) > 0, "No symbols found in bronze data."
    
    # Check Silver
    silver_rows = 0
    silver_files = list(silver_dir.rglob("*.parquet"))
    for pq_file in silver_files:
        df = pd.read_parquet(pq_file)
        silver_rows += len(df)
        
        # Schema checks
        assert "symbol" in df.columns
        assert "timestamp" in df.columns
        assert "source_type" in df.columns
        assert "ingestion_timestamp_utc" in df.columns
        assert "open" in df.columns
        
    print(f"Silver: Found {len(silver_files)} parquet files with {silver_rows} rows.")
    assert silver_rows > 0, "No rows found in silver data."
    
    print("Completeness checks passed.")

if __name__ == "__main__":
    main()
