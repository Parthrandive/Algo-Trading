from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from src.agents.preprocessing.lag_alignment import LagAligner, CorporateActionValidator

def test_lag_aligner_no_lookahead():
    """
    Verifies that macro data published AFTER the market timestamp is not visible.
    """
    # 1. Market data at T=10 and T=20
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    market_df = pd.DataFrame({
        "timestamp": [base_time, base_time + timedelta(days=10)],
        "symbol": ["RELIANCE", "RELIANCE"],
        "close": [100.0, 110.0]
    })
    
    # 2. Macro data at T-20, T-5, T+5 days
    # Let's say CPI has a 14-day publication delay.
    macro_df = pd.DataFrame({
        "timestamp": [
            base_time - timedelta(days=20), # Published at T-6
            base_time - timedelta(days=5),  # Published at T+9
            base_time + timedelta(days=5)   # Published at T+19
        ],
        "indicator_name": ["CPI", "CPI", "CPI"],
        "value": [5.0, 5.1, 5.2]
    })

    aligner = LagAligner(publication_delays={"CPI": timedelta(days=14)})
    aligned_df = aligner.align(market_df, macro_df)

    # At Market T=0 (Jan 1), CPI from T-20 (published T-6) SHOULD be visible.
    assert aligned_df.iloc[0]["CPI"] == 5.0

    # At Market T+10 (Jan 11), CPI from T-5 (published T+9) SHOULD be visible.
    # The record from T+5 (published T+19) should NOT be visible.
    assert aligned_df.iloc[1]["CPI"] == 5.1

def test_lag_aligner_missing_macros():
    """
    Missing macros shouldn't crash the aligner.
    """
    market_df = pd.DataFrame({"timestamp": [datetime.now(timezone.utc)], "close": [100.0]})
    macro_df = pd.DataFrame(columns=["timestamp", "indicator_name", "value"])

    aligner = LagAligner()
    aligned_df = aligner.align(market_df, macro_df)
    
    assert len(aligned_df) == 1
    assert "close" in aligned_df.columns

def test_corporate_action_adjustments():
    """
    Verifies that corporate actions correctly adjust historical prices and volume recursively.
    """
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    # Prices: 100, 110 (split 1:2), 55, 60
    # True historical prices are ~50, 55 before split
    market_df = pd.DataFrame({
        "symbol": ["TEST"] * 4,
        "timestamp": [
            base_time,
            base_time + timedelta(days=1),
            base_time + timedelta(days=2),
            base_time + timedelta(days=3)
        ],
        "open": [100.0, 110.0, 55.0, 60.0],
        "close": [100.0, 110.0, 55.0, 60.0],
        "volume": [1000, 1000, 2000, 2000]
    })
    
    ca_df = pd.DataFrame({
        "symbol": ["TEST"],
        "ex_date": [base_time + timedelta(days=2)], # Split happens on day 2
        "action_type": ["split"],
        "ratio": ["1:2"] # 1 share becomes 2 -> price ratio 0.5
    })
    
    validator = CorporateActionValidator()
    adjusted_df = validator.apply_adjustments(market_df, ca_df)
    
    assert adjusted_df.iloc[0]["close"] == 50.0 # 100 * 0.5
    assert adjusted_df.iloc[1]["close"] == 55.0 # 110 * 0.5
    assert adjusted_df.iloc[2]["close"] == 55.0 # Unchanged (on ex_date)
    assert adjusted_df.iloc[3]["close"] == 60.0 # Unchanged (after ex_date)
    
    assert adjusted_df.iloc[0]["volume"] == 2000 # 1000 / 0.5
    assert adjusted_df.iloc[1]["volume"] == 2000 # 1000 / 0.5
