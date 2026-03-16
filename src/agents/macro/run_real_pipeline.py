import concurrent.futures
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Dict, Tuple

from sqlalchemy import text
from src.agents.macro.client import DateRange, MacroClientInterface
from src.agents.macro.clients.akshare_client import AkShareClient
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


_INITIAL_BACKFILL_START: Dict[MacroIndicatorType, date] = {
    MacroIndicatorType.REPO_RATE: date(2000, 1, 1),
    MacroIndicatorType.US_10Y: date(2000, 1, 1),
    MacroIndicatorType.CPI: date(1957, 1, 1),
    MacroIndicatorType.WPI: date(1960, 1, 1),
    MacroIndicatorType.IIP: date(1994, 1, 1),
    MacroIndicatorType.FII_FLOW: date(2019, 1, 1),
    MacroIndicatorType.DII_FLOW: date(2019, 1, 1),
    MacroIndicatorType.FX_RESERVES: date(1950, 12, 1),
    MacroIndicatorType.RBI_BULLETIN: date(1998, 1, 1),
    MacroIndicatorType.INDIA_US_10Y_SPREAD: date(2011, 12, 1),
}

_LOOKBACK_DAYS: Dict[MacroIndicatorType, int] = {
    MacroIndicatorType.CPI: 365,
    MacroIndicatorType.WPI: 365,
    MacroIndicatorType.IIP: 365,
    MacroIndicatorType.FII_FLOW: 14,
    MacroIndicatorType.DII_FLOW: 14,
    MacroIndicatorType.REPO_RATE: 365,
    MacroIndicatorType.US_10Y: 14,
    MacroIndicatorType.FX_RESERVES: 365,
    MacroIndicatorType.RBI_BULLETIN: 365,
    MacroIndicatorType.INDIA_US_10Y_SPREAD: 365,
}


_MIN_HISTORY_ROWS: Dict[MacroIndicatorType, int] = {
    MacroIndicatorType.CPI: 120,
    MacroIndicatorType.WPI: 120,
    MacroIndicatorType.IIP: 120,
    MacroIndicatorType.FII_FLOW: 250,
    MacroIndicatorType.DII_FLOW: 250,
    MacroIndicatorType.REPO_RATE: 120,
    MacroIndicatorType.US_10Y: 1000,
    MacroIndicatorType.FX_RESERVES: 250,
    MacroIndicatorType.RBI_BULLETIN: 60,
    MacroIndicatorType.INDIA_US_10Y_SPREAD: 250,
}


@dataclass(frozen=True)
class IndicatorRangePlan:
    date_range: DateRange
    latest_date: date | None
    row_count: int
    min_history_rows: int
    force_full_backfill: bool


def _indicator_history_stats(session, indicator: MacroIndicatorType) -> tuple[date | None, int]:
    result = session.execute(
        text(
            """
            SELECT MAX(timestamp) AS latest_ts, COUNT(*) AS row_count
            FROM macro_indicators
            WHERE indicator_name = :indicator
            """
        ),
        {"indicator": indicator.value},
    ).mappings().one()
    latest_ts = result.get("latest_ts")
    latest_date = latest_ts.date() if latest_ts is not None else None
    row_count = int(result.get("row_count") or 0)
    return latest_date, row_count


def _build_indicator_range(
    indicator: MacroIndicatorType,
    now_date: date,
    latest_date: date | None,
    row_count: int,
) -> IndicatorRangePlan:
    initial_start = _INITIAL_BACKFILL_START.get(indicator, now_date - timedelta(days=90))
    lookback_days = _LOOKBACK_DAYS.get(indicator, 30)
    min_history_rows = _MIN_HISTORY_ROWS.get(indicator, 1)
    force_full_backfill = latest_date is None or row_count < min_history_rows

    if force_full_backfill:
        start = initial_start
    else:
        start = max(initial_start, latest_date - timedelta(days=lookback_days))

    if start > now_date:
        start = now_date

    return IndicatorRangePlan(
        date_range=DateRange(start=start, end=now_date),
        latest_date=latest_date,
        row_count=row_count,
        min_history_rows=min_history_rows,
        force_full_backfill=force_full_backfill,
    )


def run_macro():
    engine = get_engine()
    Session = get_session(engine)
    Base.metadata.create_all(engine)
    akshare = AkShareClient()
    
    # Set up registry with real clients
    registry: Dict[MacroIndicatorType, Tuple[MacroClientInterface, BaseParser]] = {
        MacroIndicatorType.CPI: (MOSPIClient(), CPIParser()),
        MacroIndicatorType.WPI: (MOSPIClient(), WPIParser()), # Reusing MOSPI for demo
        MacroIndicatorType.IIP: (MOSPIClient(), IIPParser()),
        MacroIndicatorType.FII_FLOW: (NSEDIIFIIClient(), FIIDIIParser()),
        MacroIndicatorType.DII_FLOW: (NSEDIIFIIClient(), FIIDIIParser()),
        # Parser is unused when client returns MacroIndicator objects directly.
        MacroIndicatorType.REPO_RATE: (akshare, CPIParser()),
        MacroIndicatorType.US_10Y: (akshare, CPIParser()),
        MacroIndicatorType.FX_RESERVES: (FXReservesClient(), FXReservesParser()),
        MacroIndicatorType.RBI_BULLETIN: (RBIClient(), RBIBulletinParser()),
        MacroIndicatorType.INDIA_US_10Y_SPREAD: (BondSpreadClient(), BondSpreadParser()),
    }

    from src.agents.macro.recorder import MacroSilverRecorder
    from src.db.silver_db_recorder import SilverDBRecorder
    db_recorder = SilverDBRecorder()
    recorder = MacroSilverRecorder(db_recorder=db_recorder)
    pipeline = MacroIngestPipeline(recorder)

    try:
        scheduler = MacroScheduler(
            config_path="configs/macro_monitor_runtime_v1.json",
            pipeline=pipeline,
            registry=registry,
        )
    
        now = datetime.now(UTC)
        now_date = now.date()
        
        required_indicators = [
            MacroIndicatorType.CPI,
            MacroIndicatorType.WPI,
            MacroIndicatorType.IIP,
            MacroIndicatorType.FII_FLOW,
            MacroIndicatorType.DII_FLOW,
            MacroIndicatorType.REPO_RATE,
            MacroIndicatorType.US_10Y,
            MacroIndicatorType.FX_RESERVES,
            MacroIndicatorType.RBI_BULLETIN,
            MacroIndicatorType.INDIA_US_10Y_SPREAD
        ]
        
        with Session() as session:
            indicator_plans = {}
            for indicator in required_indicators:
                latest_date, row_count = _indicator_history_stats(session, indicator)
                plan = _build_indicator_range(
                    indicator=indicator,
                    now_date=now_date,
                    latest_date=latest_date,
                    row_count=row_count,
                )
                indicator_plans[indicator] = plan
                if plan.force_full_backfill:
                    if latest_date is None:
                        reason = "no existing rows"
                    else:
                        reason = (
                            f"insufficient history "
                            f"(rows={plan.row_count}, min_required={plan.min_history_rows})"
                        )
                    print(
                        f"[macro-backfill] {indicator.value}: full backfill "
                        f"{plan.date_range.start} -> {plan.date_range.end} ({reason})"
                    )
                else:
                    print(
                        f"[macro-backfill] {indicator.value}: incremental "
                        f"{plan.date_range.start} -> {plan.date_range.end} "
                        f"(latest={plan.latest_date}, rows={plan.row_count})"
                    )

        def fetch_indicator(indicator):
            indicator_range = indicator_plans[indicator].date_range
            print(f"\n--- Fetching {indicator.value} ({indicator_range.start} -> {indicator_range.end}) ---")
            try:
                records = scheduler.run_job(indicator, indicator_range)
                return indicator.value, len(records), None
            except Exception as e:
                return indicator.value, 0, str(e)

        ingestion_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(required_indicators))) as executor:
            futures = [executor.submit(fetch_indicator, indicator) for indicator in required_indicators]
            for future in concurrent.futures.as_completed(futures):
                ingestion_results.append(future.result())

        print("\n--- Ingestion Summary ---")
        for indicator_name, count, error in sorted(ingestion_results):
            if error:
                print(f"{indicator_name}: FAILED ({error})")
            elif count == 0:
                print(f"{indicator_name}: no new records (up-to-date or source lag)")
            else:
                print(f"{indicator_name}: ingested {count} record(s)")
                
    except Exception as e:
        print(f"Error orchestrating pipeline: {e}")

    # Verify what was written
    print("\n--- DB Contents ---")
    with Session() as session:
        results = session.execute(text("SELECT indicator_name, timestamp, value, unit, quality_status FROM macro_indicators ORDER BY timestamp DESC LIMIT 5")).fetchall()
        for row in results:
            print(f"{row.indicator_name} at {row.timestamp}: {row.value} {row.unit} [{row.quality_status}]")
        if not results:
            print("Database is empty.")
    
    return 0

if __name__ == "__main__":
    run_macro()
