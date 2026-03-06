import time
import logging
from datetime import datetime, timezone
from src.agents.sentinel.failover_client import FailoverSentinelClient, DegradationState
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.nsepython_client import NSEPythonClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockFailingPrimary(YFinanceClient):
    def __init__(self):
        super().__init__()
        self.should_fail = False

    def get_stock_quote(self, symbol: str):
        if self.should_fail:
            raise ConnectionError("Simulated primary NSE API outage")
        return super().get_stock_quote(symbol)

def run_drill():
    logger.info("--- Starting Sentinel Failover Drill ---")
    
    primary = MockFailingPrimary()
    fallback = NSEPythonClient()
    
    # Configure low cooldown and threshold for fast testing
    client = FailoverSentinelClient(
        primary_client=primary,
        fallback_clients=[fallback],
        failure_threshold=2,
        cooldown_seconds=3,
        recovery_success_threshold=2
    )

    symbol = "TATASTEEL.NS"
    
    logger.info("\n1. Normal Operation")
    tick = client.get_stock_quote(symbol)
    logger.info(f"State: {client.degradation_state}, Tick Source: {tick.source_type}, Flag: {tick.quality_status}")
    assert client.degradation_state == DegradationState.NORMAL
    
    logger.info("\n2. Simulating Primary Outage...")
    primary.should_fail = True
    outage_start_time = time.time()
    
    # Trigger first failure
    tick = client.get_stock_quote(symbol)
    logger.info(f"State: {client.degradation_state}, Tick Source: {tick.source_type}, Flag: {tick.quality_status}")
    
    # Trigger second failure (crosses threshold of 2)
    tick = client.get_stock_quote(symbol)
    logger.info(f"State: {client.degradation_state}, Tick Source: {tick.source_type}, Flag: {tick.quality_status}")
    assert client.degradation_state == DegradationState.REDUCE_ONLY
    logger.info("Verified: reduce-only / close-only advisory flag is raised.")
    
    logger.info("\n3. Waiting for cooldown period (3 seconds)...")
    time.sleep(3.5)
    
    logger.info("\n4. Simulating Primary Recovery...")
    primary.should_fail = False
    
    # First success
    tick = client.get_stock_quote(symbol)
    logger.info(f"State: {client.degradation_state}, Tick Source: {tick.source_type}, Flag: {tick.quality_status}")
    
    # Second success (crosses recovery threshold of 2)
    tick = client.get_stock_quote(symbol)
    recovery_time = time.time()
    logger.info(f"State: {client.degradation_state}, Tick Source: {tick.source_type}, Flag: {tick.quality_status}")
    
    assert client.degradation_state == DegradationState.NORMAL
    logger.info("Verified: automatic recovery when primary feed returns.")
    
    mttr = recovery_time - outage_start_time
    logger.info(f"\nMTTR (Mean Time To Recovery): {mttr:.2f} seconds")
    logger.info("--- Sentinel Failover Drill Complete ---")

if __name__ == "__main__":
    run_drill()
