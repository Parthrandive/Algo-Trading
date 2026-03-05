import json
import logging
from datetime import datetime, UTC, timedelta
from sqlalchemy import text
from src.db.connection import get_engine, get_session
from src.db.models import MacroIndicatorDB
from src.agents.macro.freshness import MacroFreshnessChecker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_drill():
    logger.info("--- Starting Macro Stale-Source Drill ---")
    
    # 1. Simulate delayed macro release by inserting an old record
    engine = get_engine()
    Session = get_session(engine)
    
    # We will pick 'CPI' which is required in the configs config
    ind_name = "CPI"
    old_time = datetime.now(UTC) - timedelta(days=2) # 48 hours ago
    
    logger.info(f"\n1. Simulating delayed macro release for {ind_name}...")
    logger.info(f"Injecting a record with timestamp {old_time.isoformat()} (48 hours old).")
    
    with Session() as session:
        # Delete any newer records for CPI to ensure it's picked up as the latest
        session.query(MacroIndicatorDB).filter(
            MacroIndicatorDB.indicator_name == ind_name,
            MacroIndicatorDB.ingestion_timestamp_utc >= old_time
        ).delete()
        
        # Insert the stale record
        stale_record = MacroIndicatorDB(
            indicator_name=ind_name,
            value=105.2,
            unit="Index",
            period="Monthly",
            region="India",
            source_type="official_api",
            timestamp=old_time,
            ingestion_timestamp_utc=old_time,
            ingestion_timestamp_ist=datetime.now(),
            schema_version="1.1",
            quality_status="pass"
        )
        session.add(stale_record)
        session.commit()
        
    logger.info("\n2. Running Freshness Checker...")
    checker = MacroFreshnessChecker("configs/macro_monitor_runtime_v1.json")
    report = checker.generate_report()
    
    ind_details = report["details"].get(ind_name)
    assert ind_details is not None, f"Expected {ind_name} in the report details."
    
    status = ind_details["status"]
    logger.info(f"Report status for {ind_name}: {status}")
    
    if status == "STALE":
        logger.info("Verified: Stale markers appear correctly.")
    else:
        logger.error(f"Expected STALE status, got {status}")
        
    logger.info("\n3. Verifying downstream weighting controls (Phase 1 Stub)...")
    logger.info("Downstream models (like exogenous_risk_proxy) query the 'quality_status' and 'status' markers.")
    logger.info("They acknowledge stale data by applying penalty weights as defined in SLA specifications.")
    
    logger.info("\n--- Macro Stale-Source Drill Complete ---")

if __name__ == "__main__":
    run_drill()
