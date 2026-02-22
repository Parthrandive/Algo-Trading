import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text

from src.db.connection import get_engine
from src.db.models import OHLCVBar, CorporateActionDB, QuarantineBar
from src.db.init_db import init_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def migrate_ohlcv(data_dir: Path, engine, dry_run: bool = False):
    logger.info(f"Looking for OHLCV parquet files in {data_dir}")
    if not data_dir.exists():
        logger.warning(f"Directory {data_dir} does not exist. Skipping.")
        return 0
        
    files = list(data_dir.rglob("*.parquet"))
    logger.info(f"Found {len(files)} parquet files mapping to OHLCV data")
    
    total_inserted = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
                
            # Rename columns if necessary or ensure they match the DB schema
            # 'date_str' should have been removed, but drop if present
            if 'date_str' in df.columns:
                df = df.drop(columns=['date_str'])
                
            # Rename timestamp to UTC timezone awareness
            if 'timestamp' in df.columns and df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
            records = df.to_dict(orient='records')
            
            if dry_run:
                total_inserted += len(records)
                logger.info(f"[DRY RUN] Would insert {len(records)} bars from {f}")
                continue
                
            # Insert using PostgreSQL UPSERT
            with engine.connect() as conn:
                with conn.begin():
                    stmt = insert(OHLCVBar).values(records)
                    update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['timestamp', 'symbol', 'interval'],
                        set_=update_dict
                    )
                    conn.execute(stmt)
            
            total_inserted += len(records)
            logger.info(f"Migrated {len(records)} bars from {f}")
            
        except Exception as e:
            logger.error(f"Failed to migrate {f}: {e}")
            
    return total_inserted

def migrate_corporate_actions(data_dir: Path, engine, dry_run: bool = False):
    logger.info(f"Looking for Corporate Action parquet files in {data_dir}")
    if not data_dir.exists():
        logger.warning(f"Directory {data_dir} does not exist. Skipping.")
        return 0
        
    files = list(data_dir.rglob("*.parquet"))
    logger.info(f"Found {len(files)} parquet files mapping to corporate actions")
    
    total_inserted = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
                
            if 'date_str' in df.columns:
                df = df.drop(columns=['date_str'])
                
            if 'ex_date' in df.columns and df['ex_date'].dt.tz is None:
                df['ex_date'] = df['ex_date'].dt.tz_localize('UTC')
            if 'record_date' in df.columns and df['record_date'].dt.tz is None:
                df['record_date'] = df['record_date'].dt.tz_localize('UTC')
            if 'timestamp' in df.columns and df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
            records = df.to_dict(orient='records')
            
            if dry_run:
                total_inserted += len(records)
                logger.info(f"[DRY RUN] Would insert {len(records)} corporate actions from {f}")
                continue
                
            with engine.connect() as conn:
                with conn.begin():
                    stmt = insert(CorporateActionDB).values(records)
                    update_dict = {c.name: c for c in stmt.excluded if not c.primary_key}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['ex_date', 'symbol', 'action_type'],
                        set_=update_dict
                    )
                    conn.execute(stmt)
            
            total_inserted += len(records)
            logger.info(f"Migrated {len(records)} corporate actions from {f}")
            
        except Exception as e:
            logger.error(f"Failed to migrate corp action {f}: {e}")
            
    return total_inserted

def migrate_quarantine(data_dir: Path, engine, dry_run: bool = False):
    logger.info(f"Looking for Quarantine parquet files in {data_dir}")
    if not data_dir.exists():
        logger.warning(f"Directory {data_dir} does not exist. Skipping.")
        return 0
        
    files = list(data_dir.rglob("*.parquet"))
    logger.info(f"Found {len(files)} parquet files mapping to Quarantine")
    
    total_inserted = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
                
            if 'date_str' in df.columns:
                df = df.drop(columns=['date_str'])
            if 'timestamp' in df.columns and df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
            # Add required db fields that might not be in parquet
            if 'reason' not in df.columns:
                df['reason'] = 'monotonicity_violation'
            if 'quarantined_at' not in df.columns:
                df['quarantined_at'] = pd.Timestamp.now(tz='UTC')
                
            records = df.to_dict(orient='records')
            
            if dry_run:
                total_inserted += len(records)
                logger.info(f"[DRY RUN] Would insert {len(records)} quarantined bars from {f}")
                continue
                
            with engine.connect() as conn:
                with conn.begin():
                    # Just skip duplicates since we don't upsert quarantine
                    stmt = insert(QuarantineBar).values(records).on_conflict_do_nothing()
                    conn.execute(stmt)
            
            total_inserted += len(records)
            logger.info(f"Migrated {len(records)} quarantined bars from {f}")
            
        except Exception as e:
            logger.error(f"Failed to migrate quarantine {f}: {e}")
            
    return total_inserted

def main():
    parser = argparse.ArgumentParser(description="Migrate Sentinel Parquet data to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Do not insert rows into DB, just print count")
    parser.add_argument("--base-dir", default="data", help="Data base directory")
    args = parser.parse_args()
    
    logger.info(f"Starting migration from base dir: {args.base_dir}")
    if args.dry_run:
        logger.info("Executing DRY RUN. No data will be written to DB.")

    base_path = Path(args.base_dir)
    silver_ohlcv_dir = base_path / "silver" / "ohlcv"
    silver_corp_dir = base_path / "silver" / "corporate_actions"
    quarantine_dir = base_path / "quarantine"
    
    engine = get_engine()
    
    logger.info("Ensuring database schema is initialized...")
    if not args.dry_run:
        init_database()
    
    total_ohlcv = migrate_ohlcv(silver_ohlcv_dir, engine, args.dry_run)
    total_corp = migrate_corporate_actions(silver_corp_dir, engine, args.dry_run)
    total_quarantine = migrate_quarantine(quarantine_dir, engine, args.dry_run)
    
    logger.info("="*50)
    logger.info("MIGRATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Dry Run Mode:    {args.dry_run}")
    logger.info(f"OHLCV Bars:      {total_ohlcv}")
    logger.info(f"Corp Actions:    {total_corp}")
    logger.info(f"Quarantined:     {total_quarantine}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
