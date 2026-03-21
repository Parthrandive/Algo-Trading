import os
import sys
import time
import logging
import schedule
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Setup simple stdout logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("DB_Scheduler")

from scripts.backfill_historical import run as run_backfill

import json

from config.symbols import (
    FOREX_SYMBOLS,
    INDEX_SYMBOLS,
    SENTINEL_CORE_SYMBOLS,
    WATCHLIST_ROTATE_BATCH_SIZE,
    WATCHLIST_ROTATING_POOL,
    dedupe_symbols,
)

# Configuration 
WATCHLIST_PATH = PROJECT_ROOT / "configs" / "watchlist.json"
RUN_TIME_IST_MORNING = os.environ.get("SCHEDULE_TIME_MORNING_IST", "10:20")
RUN_TIME_IST_EVENING = os.environ.get("SCHEDULE_TIME_EVENING_IST", "16:30")
DAYS_TO_FETCH = int(os.environ.get("SCHEDULE_DAYS", "3"))

def get_todays_symbols() -> list[str]:
    """
    Builds today's watchlist from the central symbol config and the optional
    rotate_batch_size override in watchlist.json.
    """
    config: dict[str, object] = {}
    if not WATCHLIST_PATH.exists():
        logger.warning(f"Watchlist not found at {WATCHLIST_PATH}, using central symbol defaults.")
    else:
        with open(WATCHLIST_PATH, "r") as f:
            config = json.load(f)

    core = list(config.get("core_symbols", SENTINEL_CORE_SYMBOLS))
    index_symbols = list(config.get("index_symbols", INDEX_SYMBOLS))
    fx_symbols = list(config.get("fx_symbols", FOREX_SYMBOLS))
    pool = list(config.get("rotating_pool", WATCHLIST_ROTATING_POOL))
    batch_size = int(config.get("rotate_batch_size", WATCHLIST_ROTATE_BATCH_SIZE))
    
    if not pool:
        return dedupe_symbols([*core, *index_symbols, *fx_symbols])
        
    # Simple deterministic rotation based on day of year
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    
    # Calculate starting index for today's batch
    batch_index = (day_of_year % ((len(pool) + batch_size - 1) // batch_size))
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(pool))
    
    todays_rotating_batch = pool[start_idx:end_idx]
    
    logger.info(f"Rotation: Day {day_of_year}, picking batch {batch_index+1} (indices {start_idx}-{end_idx})")
    
    # Combine and deduplicate
    all_symbols = [*core, *index_symbols, *fx_symbols, *todays_rotating_batch]
    return dedupe_symbols(all_symbols)

def ingestion_job():
    logger.info("="*50)
    logger.info(f"Starting daily DB ingestion job at {datetime.now(timezone.utc)}")
    
    todays_symbols = get_todays_symbols()
    symbols_arg = ",".join(todays_symbols)
    logger.info(f"Target symbols for this run: {symbols_arg}")
    
    # We call the backfill script's run() method directly
    # It will automatically write to SilverDBRecorder (PostgreSQL) now
    args = [
        "--symbols", symbols_arg,
        "--days", str(DAYS_TO_FETCH),
        "--interval", "1h",
        "--workers", "3",
        "--skip-recent-hours", "1", # Force fetch newly closed day (min 1hr allowed)
    ]
    
    try:
        logger.info(f"Running backfill with args: {args}")
        # Overwrite sys.argv so argparse in backfill_historical works smoothly
        old_argv = sys.argv
        sys.argv = ["backfill_historical.py"] + args
        
        exit_code = run_backfill(sys.argv[1:])
        sys.argv = old_argv
        
        if exit_code == 0:
            logger.info("Daily market ingestion completed successfully.")
        else:
            logger.warning(f"Daily market ingestion finished with non-zero exit code: {exit_code}")
            
    except Exception as e:
        logger.error(f"Daily market ingestion failed with error: {e}")
        
    try:
        from src.agents.macro.run_real_pipeline import run_macro
        logger.info("Running Daily Macro Agent Ingestion...")
        run_macro()
        logger.info("Macro Agent ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Daily macro ingestion failed with error: {e}")
        
    try:
        from src.agents.textual.textual_data_agent import TextualDataAgent
        from src.db.silver_db_recorder import SilverDBRecorder
        logger.info("Running Daily Textual Agent Ingestion...")
        # Point to dynamic PGSQL DB
        text_agent = TextualDataAgent.from_default_components(recorder=SilverDBRecorder())
        text_agent.run_once()
        logger.info("Textual Agent ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Daily textual ingestion failed with error: {e}")
        
    try:
        from src.agents.preprocessing.pipeline import PreprocessingPipeline
        logger.info("Running Daily Preprocessing Agent (Silver -> Gold)...")
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        snapshot_id = f"daily_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            text_source_path="db_virtual",
            snapshot_id=snapshot_id
        )
        logger.info(f"Preprocessing Agent completed successfully. Cleaned {len(output.records)} rows into Gold.")
    except Exception as e:
        logger.error(f"Daily preprocessing failed with error: {e}")

    try:
        from src.agents.regime.regime_agent import RegimeAgent

        logger.info("Running Daily Regime Agent Inference...")
        regime_agent = RegimeAgent()
        regime_payloads = []
        for symbol in todays_symbols:
            try:
                pred = regime_agent.detect_regime(symbol=symbol, limit=800)
                regime_payloads.append(
                    {
                        "symbol": pred.symbol,
                        "regime_state": pred.regime_state.value,
                        "risk_level": pred.risk_level.value,
                        "confidence": pred.confidence,
                    }
                )
            except Exception as inner_exc:
                logger.warning(f"Regime inference failed for {symbol}: {inner_exc}")
        logger.info(f"Regime Agent completed. Generated {len(regime_payloads)} prediction(s).")
        if regime_payloads:
            logger.info(f"Regime sample: {regime_payloads[:3]}")
    except Exception as e:
        logger.error(f"Daily regime inference failed with error: {e}")
        
    finally:
        logger.info(f"Finished job at {datetime.now(timezone.utc)}")
        logger.info("="*50)

def main():
    logger.info(f"Automated DB Ingestion Scheduler starting...")
    logger.info(f"Scheduled times (IST): Morning at {RUN_TIME_IST_MORNING}, Evening at {RUN_TIME_IST_EVENING} every day")
    logger.info(f"Lookback window: {DAYS_TO_FETCH} days")
    
    # Check if this is a one-off run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true", help="Run the ingestion immediately and exit")
    args_p, unknown = parser.parse_known_args()
    
    if args_p.run_once:
        logger.info("Running single ingestion pass due to --run-once flag.")
        ingestion_job()
        sys.exit(0)
    
    # schedule library works on local system time by default
    schedule.every().day.at(RUN_TIME_IST_MORNING).do(ingestion_job)
    schedule.every().day.at(RUN_TIME_IST_EVENING).do(ingestion_job)
    
    # Optional: Run once immediately on startup for testing/bootstrap
    if os.environ.get("RUN_IMMEDIATELY", "false").lower() == "true":
        logger.info("RUN_IMMEDIATELY flag set, running first job now...")
        ingestion_job()
        
    logger.info("Scheduler is now running. Press Ctrl+C to exit.")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60) # Wake up every minute to check schedule
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break
        except Exception as e:
            logger.error(f"Scheduler loop encountered error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
