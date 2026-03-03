import asyncio
import concurrent.futures
from datetime import UTC, datetime
from typing import Dict, Tuple
from unittest.mock import Mock

from sqlalchemy import text
from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.clients.bond_spread_client import BondSpreadClient
from src.agents.macro.clients.mospi_client import MOSPIClient
from src.agents.macro.clients.nse_fiidii_client import NSEDIIFIIClient
from src.agents.macro.clients.fx_reserves_client import FXReservesClient
from src.agents.macro.clients.rbi_client import RBIClient
from src.agents.macro.parsers import (
    BaseParser,
    BondSpreadParser,
    CPIParser,
    FIIDIIParser,
    FXReservesParser,
    IIPParser,
    RBIBulletinParser,
    WPIParser,
)
from src.agents.macro.pipeline import MacroIngestPipeline
from src.agents.macro.scheduler import MacroScheduler
from src.db.connection import get_engine, get_session
from src.db.models import Base
from src.schemas.macro_data import MacroIndicatorType

# Create a local SQLite database for this run
DB_URL = "sqlite:///data/real_macro_data_fetch.db"
import os
os.makedirs("data", exist_ok=True)

engine = get_engine(DB_URL)
Base.metadata.create_all(engine)

# Set up registry with real clients
registry: Dict[MacroIndicatorType, Tuple[MacroClientInterface, BaseParser]] = {
    MacroIndicatorType.CPI: (MOSPIClient(), CPIParser()),
    MacroIndicatorType.WPI: (MOSPIClient(), WPIParser()), # Reusing MOSPI for demo
    MacroIndicatorType.IIP: (MOSPIClient(), IIPParser()),
    MacroIndicatorType.FII_FLOW: (NSEDIIFIIClient(), FIIDIIParser()),
    MacroIndicatorType.DII_FLOW: (NSEDIIFIIClient(), FIIDIIParser()),
    MacroIndicatorType.FX_RESERVES: (FXReservesClient(), FXReservesParser()),
    MacroIndicatorType.RBI_BULLETIN: (RBIClient(), RBIBulletinParser()),
    MacroIndicatorType.INDIA_US_10Y_SPREAD: (BondSpreadClient(), BondSpreadParser()),
}

from src.agents.macro.recorder import MacroSilverRecorder
from src.db.silver_db_recorder import SilverDBRecorder
db_recorder = SilverDBRecorder(DB_URL)
recorder = MacroSilverRecorder(db_recorder=db_recorder)
pipeline = MacroIngestPipeline(recorder)

try:
    scheduler = MacroScheduler(
        config_path="configs/macro_monitor_runtime_v1.json",
        pipeline=pipeline,
        registry=registry,
        database_url=DB_URL
    )
    
    # Range covering the latest release window
    date_range = DateRange(
        start=datetime(2026, 2, 20, tzinfo=UTC),
        end=datetime(2026, 2, 28, tzinfo=UTC)
    )
    
    required_indicators = [
        MacroIndicatorType.CPI,
        MacroIndicatorType.WPI,
        MacroIndicatorType.IIP,
        MacroIndicatorType.FII_FLOW,
        MacroIndicatorType.DII_FLOW,
        MacroIndicatorType.FX_RESERVES,
        MacroIndicatorType.RBI_BULLETIN,
        MacroIndicatorType.INDIA_US_10Y_SPREAD
    ]
    
    def fetch_indicator(indicator):
        print(f"\n--- Fetching {indicator.value} ---")
        try:
            records = scheduler.run_job(indicator, date_range)
            print(f"Ingested {len(records)} {indicator.value} records.")
        except Exception as e:
            print(f"Failed to fetch {indicator.value}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(required_indicators))) as executor:
        executor.map(fetch_indicator, required_indicators)
            
except Exception as e:
    print(f"Error orchestrating pipeline: {e}")

# Verify what was written
print("\n--- DB Contents ---")
Session = get_session(engine)
with Session() as session:
    results = session.execute(text("SELECT indicator_name, timestamp, value, unit, quality_status FROM macro_indicators")).fetchall()
    for row in results:
        print(f"{row.indicator_name} at {row.timestamp}: {row.value} {row.unit} [{row.quality_status}]")
    if not results:
        print("Database is empty.")
