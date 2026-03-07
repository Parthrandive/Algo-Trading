import sys
import logging
from datetime import datetime
from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.db.connection import get_engine
import pandas as pd

# Configure logging to see info output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # 1. Check existing rows in Gold for INFY.NS
        engine = get_engine()
        initial_count = pd.read_sql("SELECT count(*) FROM gold_features WHERE symbol='INFY.NS'", engine).iloc[0, 0]
        logger.info(f"Initial Gold features count for INFY.NS: {initial_count}")

        # 2. Run the Preprocessing Pipeline
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        snapshot_id = f"realtime_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Triggering Preprocessing Pipeline: {snapshot_id}")
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            snapshot_id=snapshot_id
        )
        
        # 3. Check new rows in Gold for INFY.NS
        new_count = pd.read_sql("SELECT count(*) FROM gold_features WHERE symbol='INFY.NS'", engine).iloc[0, 0]
        logger.info(f"New Gold features count for INFY.NS: {new_count}")
        logger.info(f"Successfully processed {new_count - initial_count} real-time frames into Gold!")
        
        # 4. Show a sample of the cleaned data
        sample = pd.read_sql("SELECT * FROM gold_features WHERE symbol='INFY.NS' ORDER BY timestamp DESC LIMIT 2", engine)
        logger.info(f"Cleaned Gold Data Sample:\n{sample[['timestamp', 'symbol', 'interval', 'close', 'close_log_return', 'close_log_return_zscore']]}")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
