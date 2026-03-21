import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.agents.preprocessing.lag_alignment import LagAligner
from src.agents.preprocessing.leakage_test import LeakageTestHarness, LeakageError
from src.schemas.preprocessing_data import TransformOutput

@pytest.fixture
def base_harness():
    return LeakageTestHarness()

@pytest.fixture
def mock_market_df():
    data = {
        "timestamp": [
            datetime(2023, 1, 1, 9, 15, tzinfo=timezone.utc),
            datetime(2023, 1, 2, 9, 15, tzinfo=timezone.utc),
            datetime(2023, 1, 3, 9, 15, tzinfo=timezone.utc),
            datetime(2023, 1, 4, 9, 15, tzinfo=timezone.utc),
            datetime(2023, 1, 5, 9, 15, tzinfo=timezone.utc)
        ],
        "symbol": ["RELIANCE"] * 5,
        "close": [100.0, 101.0, 102.0, 103.0, 104.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_macro_df():
    # CPI has a 14 day delay.
    data = {
        "timestamp": [
            datetime(2022, 12, 18, 0, 0, tzinfo=timezone.utc), # Effective Jan 1
            datetime(2022, 12, 20, 0, 0, tzinfo=timezone.utc), # Effective Jan 3
        ],
        "indicator_name": ["CPI", "CPI"],
        "value": [10.0, 11.0]
    }
    return pd.DataFrame(data)

def test_no_leakage_on_lagged_data(base_harness, mock_market_df, mock_macro_df):
    """Verify that correctly lagged data passes the leakage test."""
    
    # Simulate a correctly aligned output df where the Jan 3 CPI reading (effective Jan 3) 
    # only appears on Jan 3, and Jan 1/2 gets the old Jan 1 effective reading.
    output_records = [
        {"timestamp": mock_market_df.iloc[0]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 10.0}, # Jan 1
        {"timestamp": mock_market_df.iloc[1]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 10.0}, # Jan 2
        {"timestamp": mock_market_df.iloc[2]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 11.0}, # Jan 3
        {"timestamp": mock_market_df.iloc[3]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 11.0}, # Jan 4
        {"timestamp": mock_market_df.iloc[4]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 11.0}, # Jan 5
    ]
    
    output = TransformOutput(
        output_hash="mock",
        input_snapshot_id="mock",
        transform_config_version="v1.0",
        records=output_records
    )
    
    assert base_harness.verify_no_lookahead(mock_market_df, mock_macro_df, output)
    assert base_harness.verify_time_alignment(output)

def test_leakage_detected(base_harness, mock_market_df, mock_macro_df):
    """Verify that a planted future leak raises LeakageError."""
    
    # Plant a leak: On Jan 2, we "leak" the CPI value of 11.0 that shouldn't be effective until Jan 3
    output_records = [
        {"timestamp": mock_market_df.iloc[0]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 10.0}, # Jan 1
        {"timestamp": mock_market_df.iloc[1]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 11.0}, # Jan 2: LEAK!
        {"timestamp": mock_market_df.iloc[2]["timestamp"].isoformat(), "symbol": "RELIANCE", "CPI": 11.0}, # Jan 3
    ]
    
    output = TransformOutput(
        output_hash="mock",
        input_snapshot_id="mock",
        transform_config_version="v1.0",
        records=output_records
    )
    
    with pytest.raises(LeakageError, match="Future Leak for CPI"):
         base_harness.verify_no_lookahead(mock_market_df, mock_macro_df, output)


def test_lag_aligner_preserves_macro_values_with_unsorted_market_index():
    aligner = LagAligner(publication_delays={"CPI": timedelta(0)})
    market_df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 15, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 9, 15, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 9, 15, tzinfo=timezone.utc),
            ],
            "symbol": ["RELIANCE", "RELIANCE", "INFY"],
            "close": [100.0, 103.0, 102.0],
        },
        index=[100, 300, 200],
    )
    macro_df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 0, 0, tzinfo=timezone.utc),
            ],
            "indicator_name": ["CPI", "CPI"],
            "value": [10.0, 11.0],
        }
    )

    aligned = aligner.align(market_df, macro_df)
    observed = {
        (row.symbol, row.timestamp.isoformat()): row.CPI
        for row in aligned.itertuples(index=False)
    }

    assert observed[("INFY", datetime(2023, 1, 2, 9, 15, tzinfo=timezone.utc).isoformat())] == 10.0
    assert observed[("RELIANCE", datetime(2023, 1, 3, 9, 15, tzinfo=timezone.utc).isoformat())] == 11.0

def test_time_alignment_validation(base_harness):
    """Verify that non-monotonic timestamps raise LeakageError."""
    
    output_records = [
        {"timestamp": datetime(2023, 1, 2, tzinfo=timezone.utc).isoformat(), "symbol": "RELIANCE"}, # Out of order
        {"timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat(), "symbol": "RELIANCE"}, 
    ]
    
    output = TransformOutput(
        output_hash="mock",
        input_snapshot_id="mock",
        transform_config_version="v1.0",
        records=output_records
    )
    
    with pytest.raises(LeakageError, match="not strictly monotonic increasing"):
        base_harness.verify_time_alignment(output)
