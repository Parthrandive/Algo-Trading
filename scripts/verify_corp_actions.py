import sys
import os
import shutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.sentinel.yfinance_client import YFinanceClient
from src.agents.sentinel.recorder import SilverRecorder
from src.utils.history import normalize_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'nse_sentinel_runtime_v1.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def test_corporate_actions_ingest():
    logger.info("--- Testing Corporate Actions Ingestion ---")
    
    # Setup paths
    test_base_dir = "data/test_silver_corp"
    
    # Clean up previous runs
    if os.path.exists(test_base_dir): shutil.rmtree(test_base_dir)
    
    client = YFinanceClient()
    recorder = SilverRecorder(base_dir=test_base_dir)
    
    config = load_config()
    if len(sys.argv) > 1:
        symbols = [normalize_symbol(s) for s in sys.argv[1:]]
    else:
        symbols = config.get("symbol_universe", {}).get("core_symbols", ["RELIANCE.NS"])
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365 * 5)
    
    all_actions = []
    
    for symbol in symbols:
        logger.info(f"\nFetching corporate actions for {symbol} from {start_date.date()} to {end_date.date()}")
        
        actions = client.get_corporate_actions(symbol, start_date, end_date)
        
        logger.info(f"Retrieved {len(actions)} corporate actions for {symbol}.")
        for a in actions:
            logger.info(f"  - {a.action_type.value}: ex_date={a.ex_date.date()}, value={a.value}, ratio={a.ratio}")
            
        all_actions.extend(actions)
        
    if not all_actions:
        logger.error("No actions retrieved across all symbols. This might be a yfinance issue or network problem.")
        return
        
    logger.info("\nSaving to Silver layer...")
    recorder.save_corporate_actions(all_actions)
    
    # Verify files
    files = list(Path(test_base_dir).rglob("*.parquet"))
    logger.info(f"Created {len(files)} parquet partitions across {len(symbols)} symbols.")
    
    if len(files) > 0:
        logger.info("Corporate actions verification passed!")
    else:
        logger.error("Directory created but no parquet files found.")

if __name__ == "__main__":
    test_corporate_actions_ingest()
