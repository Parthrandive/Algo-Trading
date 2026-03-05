import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_drill():
    logger.info("--- Starting Preprocessing Idempotency Drill ---")
    
    pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
    
    market_path = str(PROJECT_ROOT / "data" / "silver" / "ohlcv")
    macro_path = str(PROJECT_ROOT / "data" / "silver" / "macro")
    
    snapshot_id = "idempotency_test_snapshot"
    cutoff = datetime.now(timezone.utc).isoformat()
    
    logger.info("\n1. Running Pipeline - Run 1...")
    output_1 = pipeline.replay_snapshot(
        market_source_path=market_path,
        macro_source_path=macro_path,
        snapshot_id=snapshot_id,
        cutoff_date=cutoff,
        corporate_action_path=None
    )
    hash_1 = output_1.output_hash
    logger.info(f"Run 1 Output Hash: {hash_1}")
    
    logger.info("\n2. Re-running Pipeline exactly the same way - Run 2...")
    output_2 = pipeline.replay_snapshot(
        market_source_path=market_path,
        macro_source_path=macro_path,
        snapshot_id=snapshot_id,
        cutoff_date=cutoff,
        corporate_action_path=None
    )
    hash_2 = output_2.output_hash
    logger.info(f"Run 2 Output Hash: {hash_2}")
    
    logger.info("\n3. Comparing Hashes...")
    if hash_1 == hash_2:
        logger.info("SUCCESS: Output hashes match. Pipeline is idempotent/reproducible.")
    else:
        logger.error("FAILURE: Hashes diverge. Idempotency broken!")
        sys.exit(1)
        
    logger.info("\n--- Preprocessing Idempotency Drill Complete ---")

if __name__ == "__main__":
    run_drill()
