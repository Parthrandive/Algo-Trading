import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import json

from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.agents.preprocessing.reproducibility import ReproducibilityHasher
from src.schemas.preprocessing_data import TransformOutput

def test_reproducibility_hasher_stability():
    """Verify that identical input data and config produce strictly identical hashes."""
    
    data = {
        "timestamp": [
            datetime(2023, 1, 1, 9, 15, tzinfo=timezone.utc),
            datetime(2023, 1, 1, 9, 15, tzinfo=timezone.utc),
        ],
        "symbol": ["RELIANCE", "TCS"],
        "close": [100.0, 50.0],
        "CPI": [10.0, 10.0]
    }
    df1 = pd.DataFrame(data)
    
    # Shuffle rows to verify sorting logic works
    df2 = df1.sample(frac=1).reset_index(drop=True)
    
    out1 = ReproducibilityHasher.build_deterministic_output(df1, snapshot_id="snap1", config_version="v1.0")
    out2 = ReproducibilityHasher.build_deterministic_output(df2, snapshot_id="snap1", config_version="v1.0")

    assert out1.output_hash == out2.output_hash
    assert out1.input_snapshot_id == "snap1"
    
def test_full_pipeline_reproducibility(tmp_path):
    """Verify pipeline output hash stability across identical runs (CP2 criteria)."""
    
    # Set up mock data
    market_file = tmp_path / "silver_market.parquet"
    macro_file = tmp_path / "silver_macro.parquet"
    
    market_data = {
        "timestamp": [
            datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc),
        ],
        "symbol": ["RELIANCE", "RELIANCE"],
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [98.0, 101.0],
        "close": [102.0, 104.0],
        "volume": [1000, 1200],
        "source_type": ["official_api", "official_api"],
        "interval": ["1d", "1d"],
        "ingestion_timestamp_utc": [
             datetime(2023, 1, 1, 5, 0, tzinfo=timezone.utc),
             datetime(2023, 1, 2, 5, 0, tzinfo=timezone.utc),
        ],
        "ingestion_timestamp_ist": [
             datetime(2023, 1, 1, 10, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
             datetime(2023, 1, 2, 10, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        ]
    }
    pd.DataFrame(market_data).to_parquet(market_file)
    
    macro_file = tmp_path / "silver_macro.json"
    
    macro_records = [{
         "timestamp": "2022-12-01T00:00:00Z",
         "indicator_name": "CPI",
         "value": 10.0,
         "unit": "index",
         "period": "Monthly",
         "schema_version": "v1.1",
         "source_type": "official_api",
         "ingestion_timestamp_utc": "2022-12-01T00:00:00Z",
         "ingestion_timestamp_ist": "2022-12-01T05:30:00+05:30"
    }]
    
    with open(macro_file, "w") as f:
         json.dump(macro_records, f)

    # Initialize pipeline
    pipeline = PreprocessingPipeline("configs/transform_config_v1.json")
    
    # Run 3 times
    hashes = set()
    for _ in range(3):
        out = pipeline.process_snapshot(
            str(market_file),
            str(macro_file),
            snapshot_id="test_snap_001"
        )
        hashes.add(out.output_hash)
        
    # Verify hash stability across 3 runs
    assert len(hashes) == 1, "Pipeline produced different hashes for the same input!"
    
    # Verify records exist and schema version is correct
    assert out.records
    assert out.transform_config_version == "v1.0"
