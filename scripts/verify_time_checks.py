import sys
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.time_sync import get_clock_drift, is_clock_synced, validate_utc_ist_consistency
from src.utils.validation import StreamMonotonicityChecker
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.config import load_default_sentinel_config
from src.schemas.market_data import Bar, SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clock_drift():
    logger.info("--- Testing Clock Drift ---")
    config = load_default_sentinel_config()
    threshold_seconds = config.clock_drift_threshold_seconds
    drift = get_clock_drift()
    if drift is None:
        logger.warning("Clock drift unavailable (NTP unreachable).")
    else:
        logger.info(f"Clock Drift: {drift:.6f} seconds")
    is_synced = is_clock_synced(threshold_seconds=threshold_seconds)
    logger.info(f"Is Synced ({threshold_seconds}s threshold): {is_synced}")
    if not is_synced:
        logger.warning("Clock drift verification failed.")
    else:
        logger.info("Clock drift verification passed.")

def test_monotonicity_logic():
    logger.info("\n--- Testing Monotonicity Logic ---")
    checker = StreamMonotonicityChecker()
    symbol = "TEST_SYM"
    
    t1 = datetime.now(timezone.utc)
    t2 = t1 + timedelta(minutes=1)
    t3 = t1 - timedelta(minutes=1) # Out of order
    
    logger.info(f"Check t1 ({t1}): {checker.check(symbol, t1, interval='1h')} (Expected: True)")
    logger.info(f"Check t2 ({t2}): {checker.check(symbol, t2, interval='1h')} (Expected: True)")
    logger.info(f"Check t3 ({t3}): {checker.check(symbol, t3, interval='1h')} (Expected: False)")
    logger.info(f"Check t2 again ({t2}): {checker.check(symbol, t2, interval='1h')} (Expected: False - duplicate/old)")
    
    t4 = t2 + timedelta(minutes=1)
    logger.info(f"Check t4 ({t4}): {checker.check(symbol, t4, interval='1h')} (Expected: True)")
    logger.info(f"Check t1 on different interval: {checker.check(symbol, t1, interval='1d')} (Expected: True)")

def test_utc_ist_consistency():
    logger.info("\n--- Testing UTC/IST Consistency ---")
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now.astimezone(timezone(timedelta(hours=5, minutes=30)))
    inconsistent_ist = (utc_now + timedelta(minutes=2)).astimezone(timezone(timedelta(hours=5, minutes=30)))

    logger.info(f"Consistent pair check: {validate_utc_ist_consistency(utc_now, ist_now)} (Expected: True)")
    logger.info(f"Inconsistent pair check: {validate_utc_ist_consistency(utc_now, inconsistent_ist)} (Expected: False)")

def test_recorder_quarantine():
    logger.info("\n--- Testing Recorder Quarantine ---")
    
    # Setup paths
    test_base_dir = "data/test_silver"
    test_quarantine_dir = "data/test_quarantine"
    
    # Clean up previous runs
    if os.path.exists(test_base_dir): shutil.rmtree(test_base_dir)
    if os.path.exists(test_quarantine_dir): shutil.rmtree(test_quarantine_dir)
    
    recorder = SilverRecorder(base_dir=test_base_dir, quarantine_dir=test_quarantine_dir)
    
    symbol = "QUARANTINE_TEST"
    base_time = datetime.now(timezone.utc)
    
    # Create bars: 
    # Batch 1: Establish high watermark at base_time
    batch1 = [
        Bar(symbol=symbol, timestamp=base_time, open=100, high=105, low=95, close=102, volume=1000, source_type=SourceType.MANUAL_OVERRIDE, interval="1m")
    ]
    
    # Batch 2: Late arrival (5 mins ago) + New data (1 min after base)
    batch2 = [
        Bar(symbol=symbol, timestamp=base_time - timedelta(minutes=5), open=100, high=105, low=95, close=102, volume=1000, source_type=SourceType.MANUAL_OVERRIDE, interval="1m"), # Should be quarantined
        Bar(symbol=symbol, timestamp=base_time + timedelta(minutes=1), open=100, high=105, low=95, close=102, volume=1000, source_type=SourceType.MANUAL_OVERRIDE, interval="1m") # Valid
    ]
    
    logger.info("Saving Batch 1 (Baseline)...")
    recorder.save_bars(batch1)
    
    logger.info("Saving Batch 2 (Contains late data)...")
    recorder.save_bars(batch2)
    
    # Verify files
    # Valid bars should be in test_base_dir
    valid_files = list(Path(test_base_dir).rglob("*.parquet"))
    logger.info(f"Valid files found: {len(valid_files)} (Expected files for t0 and t+1)")
    
    # Quarantine bars should be in test_quarantine_dir
    quarantine_files = list(Path(test_quarantine_dir).rglob("*.parquet"))
    logger.info(f"Quarantine files found: {len(quarantine_files)} (Expected file for t-5)")
    
    if len(valid_files) >= 1 and len(quarantine_files) >= 1:
        logger.info("Quarantine verification passed!")

    else:
        logger.error("Quarantine verification failed. Check directories.")

if __name__ == "__main__":
    test_clock_drift()
    test_monotonicity_logic()
    test_utc_ist_consistency()
    test_recorder_quarantine()
