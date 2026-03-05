import sys
import logging
import pandas as pd
from datetime import datetime, timezone

from src.agents.preprocessing.transform_graph import TransformGraph
from src.agents.preprocessing.normalizers import LogReturnNormalizer, ZScoreNormalizer, MinMaxNormalizer, DirectionalChangeDetector

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_drill():
    logger.info("--- Starting Feed Freeze Simulation Drill ---")
    
    # We will pass data through the TransformGraph to see how it handles it
    graph = TransformGraph("configs/transform_config_v1.json")
    graph.register(LogReturnNormalizer)
    graph.register(ZScoreNormalizer)
    graph.register(MinMaxNormalizer)
    graph.register(DirectionalChangeDetector)
    graph.build()
    
    # Base timestamps
    t1 = datetime(2026, 3, 5, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 5, 10, 15, tzinfo=timezone.utc) # The window in question
    t3 = datetime(2026, 3, 5, 10, 30, tzinfo=timezone.utc)
    
    # 1. Zero Volume Scenario
    # The feed is active, but there are no trades (volume=0). Row is emitted.
    zero_volume_df = pd.DataFrame([
        {"symbol": "TCS.NS", "timestamp": t1, "close": 100.0, "volume": 5000},
        {"symbol": "TCS.NS", "timestamp": t2, "close": 100.0, "volume": 0},    # Zero volume
        {"symbol": "TCS.NS", "timestamp": t3, "close": 105.0, "volume": 2000},
    ])
    
    logger.info("\n1. Processing Zero Volume Scenario (Feed Online, No Trades):")
    out_zero = graph.execute(zero_volume_df)
    logger.info(out_zero[["timestamp", "close", "volume", "close_log_return"]])
    assert len(out_zero) == 3, "Expected 3 rows in zero volume scenario"
    assert out_zero.iloc[1]["volume"] == 0, "Expected volume to be exactly 0"
    
    # 2. Feed Freeze (Missing Data) Scenario
    # The feed went down. The interval t2 is completely missing.
    missing_data_df = pd.DataFrame([
        {"symbol": "TCS.NS", "timestamp": t1, "close": 100.0, "volume": 5000},
        # t2 is missing!
        {"symbol": "TCS.NS", "timestamp": t3, "close": 105.0, "volume": 2000},
    ])
    
    logger.info("\n2. Processing Feed Freeze Scenario (Feed Offline, Missing Tick/Bar):")
    out_missing = graph.execute(missing_data_df)
    logger.info(out_missing[["timestamp", "close", "volume", "close_log_return"]])
    assert len(out_missing) == 2, "Expected 2 rows in feed freeze scenario"
    
    logger.info("\n3. Verification:")
    logger.info("Engine correctly distinguishes between zero volume (row exists, math applies identically) and missing data (row absent, gap in series).")
    logger.info("Downstream models must handle the timestamp gaps (e.g. via resample/ffill in feature alignment before model scoring).")
    
    logger.info("\n--- Feed Freeze Simulation Drill Complete ---")

if __name__ == "__main__":
    run_drill()
