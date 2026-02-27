import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from src.agents.preprocessing.pipeline import PreprocessingPipeline

@pytest.fixture
def mock_pipeline_data(tmp_path):
    # Setup mock config
    config = {
        "transforms": [
            {
                "transform_name": "LogReturnNormalizer",
                "version": "1.0",
                "input_schema_version": "market.bar.1.0",
                "output_schema_version": "preprocessing.features.v1.0",
                "parameters": {"target_column": "close", "output_column": "returns"}
            }
        ]
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    # Setup mock market data
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    market_df = pd.DataFrame({
        "symbol": ["RELIANCE", "RELIANCE"],
        "exchange": ["NSE", "NSE"],
        "timestamp": [base_time, base_time + timedelta(hours=1)],
        "source_type": ["official_api", "official_api"],
        "ingestion_timestamp_utc": [base_time, base_time],
        "ingestion_timestamp_ist": [base_time, base_time],
        "schema_version": ["1.0", "1.0"],
        "interval": ["1h", "1h"],
        "open": [100.0, 110.0],
        "high": [105.0, 115.0],
        "low": [95.0, 105.0],
        "close": [100.0, 110.0],
        "volume": [1000, 1000],
        "quality_status": ["pass", "pass"]
    })
    market_path = tmp_path / "mock_market.parquet"
    market_df.to_parquet(market_path)
    
    # Setup mock macro data
    macro_data = [
        {
            "indicator_name": "CPI",
            "value": 5.0,
            "unit": "%",
            "period": "Monthly",
            "timestamp": (base_time - timedelta(days=20)).isoformat(), # Published 6 days ago (20-14 lag)
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": base_time.isoformat(),
            "ingestion_timestamp_ist": base_time.isoformat(),
            "quality_status": "pass",
            "region": "India"
        }
    ]
    macro_path = tmp_path / "mock_macro.json"
    with open(macro_path, "w") as f:
        json.dump(macro_data, f)
        
    return config_path, market_path, macro_path

def test_pipeline_end_to_end(mock_pipeline_data):
    config_path, market_path, macro_path = mock_pipeline_data
    
    pipeline = PreprocessingPipeline(config_path=str(config_path))
    output = pipeline.process_snapshot(
        market_source_path=str(market_path),
        macro_source_path=str(macro_path),
        snapshot_id="snapshot_123"
    )
    
    assert output.input_snapshot_id == "snapshot_123"
    assert len(output.records) == 2
    
    # Check that transform executed (LogReturns should be present)
    assert "returns" in output.records[0]
    
    # Check that CPI was aligned
    assert output.records[0]["CPI"] == 5.0
    
def test_pipeline_determinism(mock_pipeline_data):
    config_path, market_path, macro_path = mock_pipeline_data
    pipeline = PreprocessingPipeline(config_path=str(config_path))
    
    outputs = []
    for _ in range(3):
        out = pipeline.process_snapshot(str(market_path), str(macro_path), "snap_1")
        outputs.append(out.output_hash)
        
    # Same snapshot ID and same data = perfectly deterministic hash
    assert len(set(outputs)) == 1

def test_pipeline_replay_mode(mock_pipeline_data):
    config_path, market_path, macro_path = mock_pipeline_data
    pipeline = PreprocessingPipeline(config_path=str(config_path))
    
    # Cutoff right after the first bar
    cutoff = datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc).isoformat()
    
    output = pipeline.replay_snapshot(
        market_source_path=str(market_path),
        macro_source_path=str(macro_path),
        snapshot_id="snap_replay",
        cutoff_date=cutoff
    )
    
    # Only 1 record should survive the cutoff
    assert len(output.records) == 1
    assert output.records[0]["close"] == 100.0


def test_pipeline_accepts_realtime_ticks(tmp_path):
    config = {
        "transforms": [
            {
                "transform_name": "LogReturnNormalizer",
                "version": "1.0",
                "input_schema_version": "market.bar.1.0",
                "output_schema_version": "preprocessing.features.v1.0",
                "parameters": {"target_column": "close", "output_column": "returns"},
            }
        ]
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tick_df = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "RELIANCE"],
            "exchange": ["NSE", "NSE"],
            "timestamp": [base_time, base_time + timedelta(minutes=1)],
            "source_type": ["official_api", "official_api"],
            "ingestion_timestamp_utc": [base_time, base_time],
            "ingestion_timestamp_ist": [base_time, base_time],
            "schema_version": ["1.0", "1.0"],
            "price": [100.0, 101.0],
            "volume": [1000, 1100],
            "quality_status": ["pass", "pass"],
        }
    )
    tick_path = tmp_path / "mock_tick.parquet"
    tick_df.to_parquet(tick_path)

    macro_data = [
        {
            "indicator_name": "CPI",
            "value": 5.0,
            "unit": "%",
            "period": "Monthly",
            "timestamp": (base_time - timedelta(days=20)).isoformat(),
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": base_time.isoformat(),
            "ingestion_timestamp_ist": base_time.isoformat(),
            "quality_status": "pass",
            "region": "India",
        }
    ]
    macro_path = tmp_path / "mock_macro.json"
    with open(macro_path, "w") as f:
        json.dump(macro_data, f)

    pipeline = PreprocessingPipeline(config_path=str(config_path))
    output = pipeline.process_snapshot(
        market_source_path=str(tick_path),
        macro_source_path=str(macro_path),
        snapshot_id="snapshot_ticks_01",
    )

    assert len(output.records) == 2
    assert "returns" in output.records[0]
    assert output.records[0]["interval"] == "tick"
