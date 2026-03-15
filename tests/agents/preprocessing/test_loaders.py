import json
import pandas as pd
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from src.agents.preprocessing.loader import MacroLoader, MarketLoader

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


def test_macro_loader_preserves_existing_dataset_snapshot_id(tmp_path):
    loader = MacroLoader()

    upstream_snapshot_id = "upstream_macro_snapshot_01"
    mock_macro = {
        "indicator_name": "CPI",
        "value": 5.5,
        "unit": "%",
        "period": "Monthly",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": "official_api",
        "schema_version": "v1.1",
        "dataset_snapshot_id": upstream_snapshot_id,
    }

    test_file = tmp_path / "mock_macro_with_snapshot.json"
    with open(test_file, "w") as f:
        json.dump([mock_macro], f)

    df = loader.load(str(test_file), "fallback_snapshot_id")

    assert not df.empty
    assert "dataset_snapshot_id" in df.columns
    assert df["dataset_snapshot_id"].iloc[0] == upstream_snapshot_id


def test_macro_loader_cleans_invalid_rows_and_dedupes(tmp_path):
    loader = MacroLoader()

    payload = [
        {
            "indicator_name": "CPI",
            "value": "5.5",
            "unit": "%",
            "period": "Monthly",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-02T00:00:00+00:00",
        },
        {
            "indicator_name": "CPI",
            "value": "5.8",
            "unit": "%",
            "period": "Monthly",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-03T00:00:00+00:00",
        },
        {
            "indicator_name": "WPI",
            "value": "2.4",
            "unit": "%",
            "period": "Monthly",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-03T00:00:00+00:00",
        },
        {
            "indicator_name": "IIP",
            "value": "not-a-number",
            "unit": "%",
            "period": "Monthly",
            "timestamp": "2026-01-03T00:00:00+00:00",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-03T00:00:00+00:00",
        },
        {
            "indicator_name": None,
            "value": "1.0",
            "unit": "%",
            "period": "Monthly",
            "timestamp": "2026-01-03T00:00:00+00:00",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-03T00:00:00+00:00",
        },
        {
            "indicator_name": "DII_FLOW",
            "value": "123.5",
            "unit": "INR_Cr",
            "period": "Daily",
            "timestamp": "not-a-date",
            "source_type": "official_api",
            "schema_version": "v1.1",
            "ingestion_timestamp_utc": "2026-01-03T00:00:00+00:00",
        },
    ]

    test_file = tmp_path / "macro_dirty.json"
    with open(test_file, "w") as f:
        json.dump(payload, f)

    df = loader.load(str(test_file), "snapshot_clean_macro")

    assert len(df) == 2
    assert set(df["indicator_name"].tolist()) == {"CPI", "WPI"}
    cpi_value = df.loc[df["indicator_name"] == "CPI", "value"].iloc[0]
    assert cpi_value == pytest.approx(5.8)
    assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)


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


def test_market_loader_preserves_existing_dataset_snapshot_id(tmp_path):
    loader = MarketLoader()

    now_utc = datetime.now(timezone.utc)
    upstream_snapshot_id = "upstream_market_snapshot_01"
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
        "volume": 1000,
        "dataset_snapshot_id": upstream_snapshot_id,
    }

    test_file = tmp_path / "mock_market_with_snapshot.parquet"
    pd.DataFrame([mock_bar]).to_parquet(test_file)

    df = loader.load(str(test_file), "fallback_snapshot_id")

    assert not df.empty
    assert "dataset_snapshot_id" in df.columns
    assert df["dataset_snapshot_id"].iloc[0] == upstream_snapshot_id


def test_market_loader_tick_success(tmp_path):
    loader = MarketLoader()

    now_utc = datetime.now(timezone.utc)
    mock_tick = {
        "symbol": "RELIANCE",
        "timestamp": now_utc,
        "source_type": "official_api",
        "ingestion_timestamp_utc": now_utc,
        "ingestion_timestamp_ist": now_utc.astimezone(IST),
        "schema_version": "1.0",
        "price": 105.5,
        "volume": 1200,
    }

    test_file = tmp_path / "mock_tick.parquet"
    pd.DataFrame([mock_tick]).to_parquet(test_file)

    snapshot_id = "test_snapshot_tick_01"
    df = loader.load(str(test_file), snapshot_id)

    assert not df.empty
    assert "dataset_snapshot_id" in df.columns
    assert df["dataset_snapshot_id"].iloc[0] == snapshot_id
    assert df["interval"].iloc[0] == "tick"
    assert df["close"].iloc[0] == pytest.approx(105.5)
    assert df["open"].iloc[0] == pytest.approx(105.5)
    assert df["high"].iloc[0] == pytest.approx(105.5)
    assert df["low"].iloc[0] == pytest.approx(105.5)
