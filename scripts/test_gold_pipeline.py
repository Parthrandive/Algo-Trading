import sys
import logging
from datetime import datetime
from src.agents.preprocessing.pipeline import PreprocessingPipeline

# Configure logging to see info output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        snapshot_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Triggering Preprocessing Pipeline for snapshot: {snapshot_id}")
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            snapshot_id=snapshot_id
        )
        logger.info(f"Pipeline executed successfully!")
        logger.info(f"Output Hash: {output.output_hash}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
