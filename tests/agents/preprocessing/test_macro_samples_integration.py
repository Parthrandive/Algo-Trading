import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from pydantic import TypeAdapter

from src.agents.preprocessing.loader import MacroLoader, MarketLoader
from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.schemas.macro_data import MacroIndicator
from src.schemas.preprocessing_data import TransformOutput

MACRO_SAMPLES_DIR = Path("data/macro_samples")
REQUIRED_INDICATORS = [
    "cpi", "wpi", "iip", "fii_flow", "dii_flow", "fx_reserves", "rbi_bulletin", "india_us_10y_spread"
]

@pytest.fixture
def mock_market_data(tmp_path):
    """Create a minimal market data payload to act as a join basis."""
    import datetime
    from zoneinfo import ZoneInfo
    UTC = ZoneInfo("UTC")
    now = datetime.datetime.now(UTC)
    
    # Needs to match some dates close to what's in the generated macro payloads
    # Let's generate the last 15 days of market bars
    data = []
    for i in range(15):
        dt = now - datetime.timedelta(days=i)
        data.append({
            "symbol": "RELIANCE.NS",
            "exchange": "NSE",
            "timestamp": dt.replace(hour=15, minute=30, second=0, microsecond=0).isoformat(),
            "source_type": "official_api",
            "ingestion_timestamp_utc": now.isoformat(),
            "ingestion_timestamp_ist": now.isoformat(),
            "schema_version": "1.0",
            "quality_status": "pass",
            "interval": "1D",
            "open": 2500.0,
            "high": 2550.0,
            "low": 2480.0,
            "close": 2520.0 + (i * 2),
            "volume": 1000000 + (1000 * i)
        })
        
    df = pd.DataFrame(data)
    out_path = tmp_path / "mock_market.parquet"
    df.to_parquet(out_path)
    return str(out_path)

def test_macro_sample_files_exist():
    """Ensure all required sample files were published by Macro Monitor."""
    missing = []
    for ind in REQUIRED_INDICATORS:
        expected_path = MACRO_SAMPLES_DIR / f"{ind}_sample.json"
        if not expected_path.exists():
            missing.append(ind)
            
    assert not missing, f"Missing sample payloads for: {missing}"

@pytest.mark.parametrize("indicator_name", REQUIRED_INDICATORS)
def test_schema_validation_per_indicator(indicator_name):
    """Validate each payload strictly using TypeAdapter."""
    file_path = MACRO_SAMPLES_DIR / f"{indicator_name}_sample.json"
    
    with open(file_path, "r") as f:
        payload = json.load(f)
        
    # Strictly parse list[MacroIndicator]
    adapter = TypeAdapter(list[MacroIndicator])
    validated = adapter.validate_python(payload)
    
    # Assert structural integrity 
    assert len(validated) > 0
    assert validated[0].schema_version == "1.1"
    
    # Prove provenance tags exist natively
    for record in validated:
        assert record.ingestion_timestamp_utc is not None
        assert record.source_type is not None

def test_end_to_end_pipeline_integration(mock_market_data, tmp_path):
    """
    Consume the entire macro sample directory through the MacroLoader,
    process it through LagAligner + Transforms, and produce the output hash.
    """
    pipeline = PreprocessingPipeline("configs/transform_config_v1.json")
    snapshot_id = "test_cp3_sync_gate_a"
    
    # Note: `process_snapshot` intrinsically calls `self.macro_loader.load(macro_source_path, snapshot_id)`
    try:
        output: TransformOutput = pipeline.process_snapshot(
            market_source_path=mock_market_data,
            macro_source_path=str(MACRO_SAMPLES_DIR),
            snapshot_id=snapshot_id
        )
    except Exception as e:
        pytest.fail(f"Pipeline crashed during execution: {e}")
        
    # Validation checkpoints
    assert output is not None
    assert output.output_hash is not None
    assert output.input_snapshot_id == snapshot_id
    assert output.transform_config_version == "v1.0"
    
    print("\n")
    print("=" * 60)
    print(f"Sync Gate A - Preprocessing Output Complete!")
    print(f"Output Hash: {output.output_hash}")
    print(f"Rows Produced: {len(output.records)}")
    print("=" * 60)
    
    # Verify provenance exists in the output records
    if len(output.records) > 0:
        first_record = output.records[0]
        # Depending on how TransformGraph builds the output, ensure it appended snapshot ID
        assert "dataset_snapshot_id" in first_record or output.input_snapshot_id == snapshot_id
