import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.technical.nsemine_fetcher import NseMineFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Bulk fetch historical data from NSE and optionally store to DB.")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g., INFY.NS)")
    parser.add_argument("--from-date", required=True, help="Start date in DD-MM-YYYY format")
    parser.add_argument("--to-date", required=True, help="End date in DD-MM-YYYY format")
    parser.add_argument("--interval", default="1d", help="Resolution: 1m, 5m, 15m, 1h, 1d, 1w... default='1d'")
    parser.add_argument("--store-db", action="store_true", help="Store fetched data in the local PostgreSQL DB")
    args = parser.parse_args()

    # 1. Fetch from NSE via nsemine
    try:
        df = NseMineFetcher.fetch_historical(args.symbol, args.from_date, args.to_date, args.interval)
    except Exception as e:
        logger.error(f"Failed to fetch data from NSE: {e}")
        return

    if df.empty:
        logger.error(f"No data retrieved for {args.symbol} from {args.from_date} to {args.to_date}.")
        return

    logger.info(f"Successfully retrieved {len(df)} rows for {args.symbol} at {args.interval} interval.")
    
    # 2. Optionally store in DB
    if args.store_db:
        db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
        try:
            from sqlalchemy import text
            from sqlalchemy.dialects.postgresql import insert
            engine = create_engine(db_url)
            
            # Fill in all NOT NULL columns required by ohlcv_bars table
            now_utc = datetime.now()
            df["interval"] = args.interval
            df["exchange"] = "NSE"
            df["source_type"] = "nsemine_historical"
            df["quality_status"] = "pass"
            df["schema_version"] = "1.0"
            df["ingestion_timestamp_utc"] = now_utc
            # Calculate naive IST equivalent for matching schema requirements
            df["ingestion_timestamp_ist"] = pd.Timestamp(now_utc).tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)

            # Convert to list of dicts for bulk insert
            records = df.to_dict(orient="records")
            
            # Use proper Postgres UPSERT (ON CONFLICT DO NOTHING)
            from sqlalchemy import Table, MetaData
            metadata = MetaData()
            ohlcv_table = Table("ohlcv_bars", metadata, autoload_with=engine)
            
            stmt = insert(ohlcv_table).values(records)
            # primary keys: timestamp, symbol, interval
            stmt = stmt.on_conflict_do_nothing(index_elements=["timestamp", "symbol", "interval"])
            
            with engine.begin() as conn:
                result = conn.execute(stmt)
                logger.info(f"Inserted/Updated rows for {args.symbol}. (Total in batch: {len(records)})")
                
        except Exception as e:
            logger.error(f"Failed to store in DB: {e}")
            logger.info("Ensure timescaledb is running and schema matches.")
    else:
        logger.info("Fetched preview (pass --store-db to save to database):")
        print(df.head())
        print("...")
        print(df.tail())

if __name__ == "__main__":
    main()
