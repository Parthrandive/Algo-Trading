import sys
import os
from datetime import datetime

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.logger import setup_logger
    from src.schemas.market_data import Tick, SourceType as MarketSourceType
except ImportError as e:
    print(f"CRITICAL: Failed to import project modules: {e}")
    sys.exit(1)

logger = setup_logger("smoke_test")

def run_smoke_test():
    logger.info("Starting smoke test...")
    
    # 1. Test Key Imports & Logging
    logger.info("Logger initialized successfully.")
    
    # 2. Test Market Data Schema
    try:
        tick = Tick(
            symbol="SMOKE_TEST",
            timestamp=datetime.now(),
            source_type=MarketSourceType.OFFICIAL_API,
            price=100.0,
            volume=10
        )
        logger.info(f"Schema validation passed for Tick: {tick.symbol} at {tick.price}")
    except Exception as e:
        logger.error(f"Schema validation FAILED: {e}")
        sys.exit(1)

    # 3. Environment Check (Optional: check for .env)
    if os.path.exists(".env"):
        logger.info("Found .env file.")
    else:
        logger.warning(".env file NOT found (this might be expected in CI/clean envs).")

    logger.info("Smoke test PASSED!")

if __name__ == "__main__":
    run_smoke_test()
