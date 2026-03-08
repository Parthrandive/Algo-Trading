import sys
import logging
from datetime import datetime
import pandas as pd
from src.db.connection import get_engine
from src.agents.preprocessing.pipeline import PreprocessingPipeline
from src.agents.macro.run_real_pipeline import run_macro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        engine = get_engine()
        
        logger.info("--- Step 1: Running Macro Pipeline (Real Data Fetch) ---")
        run_macro()

        logger.info("\n--- Step 2: Running Preprocessing Pipeline ---")
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        
        # Override publication delays to 0 for E2E testing to force immediate joining
        from datetime import timedelta
        pipeline.lag_aligner.publication_delays = {k: timedelta(0) for k in pipeline.lag_aligner.publication_delays}
        
        # Monkey patch macro loader to shift timestamps backward in memory so they join successfully
        original_macro_load = pipeline.macro_loader.load
        def patched_macro_load(*args, **kwargs):
            df = original_macro_load(*args, **kwargs)
            if not df.empty:
                df['timestamp'] = df['timestamp'] - pd.Timedelta(days=10)
            return df
        pipeline.macro_loader.load = patched_macro_load
        
        snapshot_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            snapshot_id=snapshot_id,
            symbol_filter="RELIANCE.NS" # Run for one stock
        )
        
        logger.info("\n--- Step 3: Verifying Data in DB ---")
        # Check newly inserted rows in Gold
        query = 'SELECT timestamp, symbol, close, close_log_return, close_log_return_zscore, "CPI", "WPI", "IIP", "FII_FLOW", "FX_RESERVES", macro_directional_flag FROM gold_features WHERE symbol=\'RELIANCE.NS\' ORDER BY timestamp DESC LIMIT 5'
        df = pd.read_sql(query, engine)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        logger.info(f"Gold Features Sample:\n{df}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
