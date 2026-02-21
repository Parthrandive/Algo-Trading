"""
Day 13 End-to-End Validation Script
Runs the SentinelIngestPipeline with FailoverSentinelClient.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

# Ensure src in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.schemas.market_data import SourceType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("e2e_test")

def main():
    logger.info("Starting Day 13 E2E Validation...")
    
    # 1. Initialize Clients
    primary = NSEPythonClient()
    fallback = YFinanceClient()
    
    # 2. Setup Failover Client
    client = FailoverSentinelClient(
        primary_client=primary,
        fallback_clients=[fallback],
        fallback_source_type=SourceType.FALLBACK_SCRAPER
    )
    
    # 3. Setup Recorders
    data_dir = PROJECT_ROOT / "data" / "e2e_test"
    bronze_dir = data_dir / "bronze"
    silver_dir = data_dir / "silver"
    quarantine_dir = data_dir / "quarantine"
    
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    bronze = BronzeRecorder(base_dir=str(bronze_dir))
    silver = SilverRecorder(base_dir=str(silver_dir), quarantine_dir=str(quarantine_dir))
    
    # 4. Load Config
    config = load_default_sentinel_config()
    
    # 5. Initialize Pipeline
    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver,
        bronze_recorder=bronze,
        session_rules=config.session_rules
    )
    
    # 6. Test Data Ingestion
    symbols = ["RELIANCE.NS", "TCS.NS"]
    
    end_date = datetime.now(ZoneInfo("UTC"))
    start_date = end_date - timedelta(days=3)
    
    for symbol in symbols:
        logger.info(f"Ingesting historical data for {symbol}")
        try:
            persisted = pipeline.ingest_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1h"
            )
            logger.info(f"Persisted {len(persisted)} bars for {symbol}")
        except Exception as e:
            logger.error(f"Failed to ingest historical data for {symbol}: {e}")
            
        logger.info(f"Ingesting real-time quote for {symbol}")
        try:
            tick = pipeline.ingest_quote(symbol=symbol)
            logger.info(f"Ingested quote for {symbol}: {tick.price}")
        except Exception as e:
            logger.error(f"Failed to ingest quote for {symbol}: {e}")

    logger.info("Day 13 E2E test completed. Output generated in data/e2e_test.")

if __name__ == "__main__":
    main()
