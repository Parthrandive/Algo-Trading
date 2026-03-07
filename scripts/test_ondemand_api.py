import logging
import time
from src.agents.preprocessing.on_demand_preprocessor import OnDemandPreprocessor
from src.db.connection import get_engine
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_on_demand_api():
    preprocessor = OnDemandPreprocessor()
    symbol = "HDFCBANK.NS"

    logger.info("--- Test 1: Fetch clean features (Should auto-backfill if missing, or run preprocessing) ---")
    start = time.time()
    df1 = preprocessor.get_clean_features(symbol)
    duration1 = time.time() - start
    logger.info(f"Test 1 Complete. Returned {len(df1)} rows in {duration1:.2f}s")
    
    logger.info("\n--- Test 2: Fetch again immediately (Should hit TTL Cache instantly) ---")
    start = time.time()
    df2 = preprocessor.get_clean_features(symbol)
    duration2 = time.time() - start
    logger.info(f"Test 2 Complete. Returned {len(df2)} rows in {duration2:.2f}s")
    
    if duration2 > 1.0:
        logger.error("Test 2 failed caching check! It took too long.")
    else:
        logger.info("Caching works perfectly!")
        
    logger.info("\n--- Test 3: Check Database for Duplicates ---")
    engine = get_engine()
    count = pd.read_sql(f"SELECT COUNT(*) FROM gold_features WHERE symbol='{symbol}'", engine).iloc[0,0]
    # We expect 'count' to equal len(df1) because of the upsert logic
    if count == len(df1):
         logger.info(f"Upsert logic verified. Rows in DB ({count}) exactly match clean features size ({len(df1)}). No duplicates!")
    else:
         logger.warning(f"Upsert mismatch! DB rows: {count}, Payload rows: {len(df1)}")

if __name__ == "__main__":
    verify_on_demand_api()
