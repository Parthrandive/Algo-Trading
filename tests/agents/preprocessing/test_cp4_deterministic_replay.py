"""
Day 7 Deterministic Replay Test for CP4 Week 3 Acceptance Gate.

Validates that evaluating a frozen set of inputs (mock market data + macro canonical samples)
through the PreprocessingPipeline multiple times yields exactly the same Output Hash.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.schemas.preprocessing_data import TransformOutput

MACRO_SAMPLES_DIR = Path("data/macro_samples")

@pytest.fixture
def mock_market_data(tmp_path):
    """Fixture to create standardized mock OHLCV input data to combine with macro samples."""
    import datetime
    from zoneinfo import ZoneInfo
    UTC = ZoneInfo("UTC")
    now = datetime.datetime.now(UTC)
    
    data = []
    for i in range(15):
        dt = now - datetime.timedelta(days=i)
        data.append({
            "symbol": "NIFTY50",
            "exchange": "NSE",
            "timestamp": dt.replace(hour=15, minute=30, second=0, microsecond=0).isoformat(),
            "source_type": "official_api",
            "ingestion_timestamp_utc": now.isoformat(),
            "ingestion_timestamp_ist": now.isoformat(),
            "schema_version": "1.0",
            "quality_status": "pass",
            "interval": "1D",
            "open": 22000.0,
            "high": 22100.0,
            "low": 21900.0,
            "close": 22050.0 + (i * 2),
            "volume": 1000000 + (1000 * i)
        })
        
    df = pd.DataFrame(data)
    
    # Store temporary parquet to mock reading from the Bronze/Silver storage later
    output_path = tmp_path / "mock_market.parquet"
    df.to_parquet(output_path)
    return str(output_path)

def test_deterministic_replay_hash_stability(mock_market_data, tmp_path):
    """
    Simulates a locked dataset (Snapshot ID) and ensures that running the entire
    preprocessing pipeline 3 times chronologically guarantees exact parity 
    on the mathematical Reproducibility Hash.
    """
    
    # 1. Initialize Pipeline with frozen config
    pipeline = PreprocessingPipeline("configs/transform_config_v1.json")
    
    # 2. Assign standard Snapshot ID
    snapshot_id = "test_cp4_deterministic_baseline"
    
    hashes = []
    
    # 3. Simulate Replaying identical physical data multiple times and capturing its exact artifact hashes
    for idx in range(3):
        # Process snapshot is a completely independent orchestration run mimicking a full historical replay
        output: TransformOutput = pipeline.process_snapshot(
            market_source_path=mock_market_data,
            macro_source_path=str(MACRO_SAMPLES_DIR),
            snapshot_id=snapshot_id
        )
        hashes.append(output.output_hash)
        
    print("\n" + "="*80)
    print("CP4 DETERMINISTIC REPLAY TEST RESULTS")
    print("="*80)
    print(f"Run 1 Hash: {hashes[0]}")
    print(f"Run 2 Hash: {hashes[1]}")
    print(f"Run 3 Hash: {hashes[2]}")
    
    # 4. Strict determinism check - all strings MUST be physically identical down to the bit mapping
    assert hashes[0] == hashes[1], "Run 1 deviates from Run 2"
    assert hashes[1] == hashes[2], "Run 2 deviates from Run 3"
    
    print("\n✅ Verification COMPLETE: Mathematical Determinism Guaranteed (Phase 1 5.5 rules satisfied).")
    print("="*80)
