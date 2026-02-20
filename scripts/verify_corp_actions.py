import sys
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.recorder import SilverRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_corporate_actions_ingest():
    logger.info("--- Testing Corporate Actions Ingestion ---")
    
    # Setup paths
    test_base_dir = "data/test_silver_corp"
    
    # Clean up previous runs
    if os.path.exists(test_base_dir): shutil.rmtree(test_base_dir)
    
    client = YFinanceClient()
    recorder = SilverRecorder(base_dir=test_base_dir)
    
    # Test with RELIANCE.NS, fetching last 5 years to ensure we hit dividends/splits
    symbol = "RELIANCE.NS"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * 5)
    
    logger.info(f"Fetching corporate actions for {symbol} from {start_date.date()} to {end_date.date()}")
    
    actions = client.get_corporate_actions(symbol, start_date, end_date)
    
    logger.info(f"Retrieved {len(actions)} corporate actions.")
    for a in actions:
        logger.info(f"  - {a.action_type.value}: ex_date={a.ex_date.date()}, value={a.value}, ratio={a.ratio}")
        
    if not actions:
        logger.error("No actions retrieved. This might be a yfinance issue or network problem.")
        return
        
    logger.info("Saving to Silver layer...")
    recorder.save_corporate_actions(actions)
    
    # Verify files
    corp_dir = Path(test_base_dir) / "corporate_actions" / symbol
    
    if corp_dir.exists():
        files = list(corp_dir.rglob("*.parquet"))
        logger.info(f"Created {len(files)} parquet partitions.")
        if len(files) > 0:
            logger.info("Corporate actions verification passed!")
        else:
            logger.error("Directory created but no parquet files found.")
    else:
        logger.error(f"Corporate actions directory {corp_dir} not created.")

if __name__ == "__main__":
    test_corporate_actions_ingest()
