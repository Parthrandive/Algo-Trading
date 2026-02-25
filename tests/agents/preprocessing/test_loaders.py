import json
import pandas as pd
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from src.agents.preprocessing.loader import MacroLoader, MarketLoader, SchemaVersionError

IST = ZoneInfo("Asia/Kolkata")

def test_macro_loader_success(tmp_path):
    loader = MacroLoader()
    
    mock_macro = {
        "indicator_name": "CPI",
        "value": 5.5,
        "unit": "%",
        "period": "Monthly",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": "official_api",
        "schema_version": "v1.1" 
    }

    test_file = tmp_path / "mock_macro.json"
    with open(test_file, "w") as f:
        json.dump([mock_macro], f)

    snapshot_id = "test_snapshot_macro_01"
    df = loader.load(str(tmp_path), snapshot_id)

    assert not df.empty, "DataFrame should not be empty"
    assert "dataset_snapshot_id" in df.columns
    assert df["dataset_snapshot_id"].iloc[0] == snapshot_id
    assert df["indicator_name"].iloc[0] == "CPI"

def test_market_loader_success(tmp_path):
    loader = MarketLoader()
    
    now_utc = datetime.now(timezone.utc)
    mock_bar = {
        "symbol": "RELIANCE",
        "timestamp": now_utc,
        "source_type": "broker_api",
        "ingestion_timestamp_utc": now_utc,
        "ingestion_timestamp_ist": now_utc.astimezone(IST),
        "schema_version": "1.0", 
        "interval": "1d",
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1000
    }
    
    df_in = pd.DataFrame([mock_bar])
    
    test_file = tmp_path / "mock_market.parquet"
    df_in.to_parquet(test_file)

    snapshot_id = "test_snapshot_market_01"
    
    df = loader.load(str(test_file), snapshot_id)
    assert not df.empty
    assert "dataset_snapshot_id" in df.columns
    assert df["dataset_snapshot_id"].iloc[0] == snapshot_id
