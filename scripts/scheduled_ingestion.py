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

# Configuration 
DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC"]
RUN_TIME_IST = os.environ.get("SCHEDULE_TIME_IST", "18:00")  # 6:00 PM IST is 12:30 PM UTC
DAYS_TO_FETCH = int(os.environ.get("SCHEDULE_DAYS", "3"))

def ingestion_job():
    logger.info("="*50)
    logger.info(f"Starting daily DB ingestion job at {datetime.now(timezone.utc)}")
    
    symbols_arg = ",".join(DEFAULT_SYMBOLS)
    
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
            logger.info("Daily ingestion completed successfully.")
        else:
            logger.warning(f"Daily ingestion finished with non-zero exit code: {exit_code}")
            
    except Exception as e:
        logger.error(f"Daily ingestion failed with error: {e}")
    finally:
        logger.info(f"Finished job at {datetime.now(timezone.utc)}")
        logger.info("="*50)

def main():
    logger.info(f"Automated DB Ingestion Scheduler starting...")
    logger.info(f"Target symbols: {DEFAULT_SYMBOLS}")
    logger.info(f"Scheduled time (IST): {RUN_TIME_IST} every day")
    logger.info(f"Lookback window: {DAYS_TO_FETCH} days")
    
    # schedule library works on local system time by default
    # If container timezone is UTC, "18:00" might need to be "12:30"
    # The .env file currently has TIMEZONE="Asia/Kolkata" but we'll 
    # just rely on the OS time configured in the container.
    
    schedule.every().day.at(RUN_TIME_IST).do(ingestion_job)
    
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
