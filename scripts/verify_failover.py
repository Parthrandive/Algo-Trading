
import logging
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timezone
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.client import NSEClientInterface
from src.schemas.market_data import Tick, SourceType, QualityFlag

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_failover")

class BrokenClient(NSEClientInterface):
    def get_stock_quote(self, symbol: str) -> Tick:
        raise ConnectionError("Simulated connection failure")

    def get_historical_data(self, symbol, start, end):
        raise ConnectionError("Simulated connection failure")

class WorkingClient(NSEClientInterface):
    def get_stock_quote(self, symbol: str) -> Tick:
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            source_type=SourceType.MANUAL_OVERRIDE, # Just for test
            price=123.45,
            volume=1000,
            quality_status=QualityFlag.PASS
        )

    def get_historical_data(self, symbol, start, end):
        return []

def main():
    primary = BrokenClient()
    fallback = WorkingClient()
    
    # Threshold 1 for immediate failover
    client = FailoverSentinelClient(primary, [fallback], failure_threshold=1)
    
    symbol = "RELIANCE"
    logger.info(f"Fetching quote for {symbol} with Broken Primary...")
    
    try:
        quote = client.get_stock_quote(symbol)
        logger.info(f"Success! Quote price: {quote.price} (from {quote.source_type})")
        logger.info(f"Primary failures recorded: {client._primary_failures}")
        logger.info(f"Primary healthy status: {client._is_primary_healthy}")
        
        # Verify it was fallback
        if quote.source_type == SourceType.MANUAL_OVERRIDE:
             logger.info("Verified: Data came from fallback client.")
        else:
             logger.error("Error: Data did NOT come from fallback!")
             
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
