import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.agents.preprocessing.pipeline import PreprocessingPipeline

@pytest.fixture
def nse_pipeline_setup(tmp_path):
    # Setup mock config
    config = {
        "transforms": [
            {
                "transform_name": "LogReturnNormalizer",
                "version": "1.0",
                "input_schema_version": "market.bar.1.0",
                "output_schema_version": "preprocessing.features.v1.0",
                "parameters": {"target_column": "close", "output_column": "returns"}
            },
            {
                "transform_name": "ZScoreNormalizer",
                "version": "1.0",
                "input_schema_version": "preprocessing.features.v1.0",
                "output_schema_version": "preprocessing.features.v1.0",
                "parameters": {"target_column": "returns", "output_column": "z_returns", "window": 10}
            }
        ]
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    # Setup mock macro data
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    macro_data = []
    
    # Generate some mock CPI and FII_FLOW data to join against the real NSE bars
    for i in range(30):
        # CPI published every 14 days
        if i % 14 == 0:
            macro_data.append({
                "indicator_name": "CPI",
                "value": 5.0 + (i * 0.1),
                "unit": "%",
                "period": "Monthly",
                "timestamp": (base_time + timedelta(days=i)).isoformat(), 
                "source_type": "official_api",
                "schema_version": "v1.1",
                "ingestion_timestamp_utc": base_time.isoformat(),
                "ingestion_timestamp_ist": base_time.isoformat(),
                "quality_status": "pass",
                "region": "India"
            })
            
        # FII_FLOW published daily
        macro_data.append({
            "indicator_name": "FII_FLOW",
            "value": 1000.0 * (i % 5),
            "unit": "Cr",
            "period": "Daily",
            "timestamp": (base_time + timedelta(days=i)).isoformat(), 
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": base_time.isoformat(),
            "ingestion_timestamp_ist": base_time.isoformat(),
            "quality_status": "pass",
            "region": "India"
        })
            
    macro_path = tmp_path / "mock_macro.json"
    with open(macro_path, "w") as f:
        json.dump(macro_data, f)
        
    # The real NSE data path mapping
    real_market_path = Path("data/silver/ohlcv")
    
    return config_path, str(real_market_path), str(macro_path)

@pytest.mark.skipif(not Path("data/silver/ohlcv/RELIANCE.NS").exists(), reason="Real NSE data missing")
def test_real_nse_data_pipeline(nse_pipeline_setup):
    """
    Integration test utilizing the fully populated data/silver/ohlcv directory.
    Tests reading all 11 symbols and processing them through the pipeline.
    """
    config_path, real_market_path, macro_path = nse_pipeline_setup
    
    pipeline = PreprocessingPipeline(config_path=str(config_path))
    output = pipeline.process_snapshot(
        market_source_path=real_market_path,
        macro_source_path=macro_path,
        snapshot_id="snapshot_real_nse_01"
    )
    
    # Asserts for base pipeline success
    assert output.input_snapshot_id == "snapshot_real_nse_01"
    assert len(output.records) > 1000  # 474 files * ~7 hours = ~3000 records
    
    # Check shape/features
    first_record = output.records[0]
    
    assert "symbol" in first_record
    assert "close" in first_record
    assert "returns" in first_record
    assert "z_returns" in first_record
    
    # FII_FLOW and CPI might be 0.0 or NaN filled depending on alignment, but they should be columns
    assert "FII_FLOW" in first_record or output.records[-1].get("FII_FLOW") is not None

    # 1. Reproducibility Check (Section 5.3 + 5.5)
    output_run_2 = pipeline.process_snapshot(
        market_source_path=real_market_path,
        macro_source_path=macro_path,
        snapshot_id="snapshot_real_nse_01"
    )
    assert output.output_hash == output_run_2.output_hash, "NSE Pipeline output is not reproducible!"

    # 2. Leakage Check (Section 5.3)
    from src.agents.preprocessing.leakage_test import LeakageTestHarness
    harness = LeakageTestHarness()
    
    market_df = pipeline.market_loader.load(real_market_path, "snapshot_real_nse_01")
    macro_df = pipeline.macro_loader.load(macro_path, "snapshot_real_nse_01")
    
    assert harness.verify_time_alignment(output), "Time alignment test failed for real NSE data!"
    assert harness.verify_no_lookahead(market_df, macro_df, output), "Lookahead leakage detected in real NSE data!"
