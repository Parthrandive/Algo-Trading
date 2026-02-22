import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.sentinel.broker_client import BrokerAPIClient
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.utils.history import normalize_symbol


def _build_failover_client() -> FailoverSentinelClient:
    primary = YFinanceClient()
    fallbacks = []

    broker_base_url = os.getenv("BROKER_API_BASE_URL")
    if broker_base_url:
        fallbacks.append(
            BrokerAPIClient(
                base_url=broker_base_url,
                api_key=os.getenv("BROKER_API_KEY"),
                access_token=os.getenv("BROKER_ACCESS_TOKEN"),
            )
        )

    # NSEPython is the default Week 2 fallback path.
    fallbacks.append(NSEPythonClient())
    return FailoverSentinelClient(primary, fallbacks, failure_threshold=2, cooldown_seconds=60, recovery_success_threshold=2)


def test_ingest():
    config = load_default_sentinel_config()
    print(f"Loaded runtime config: {config.version}")

    if len(sys.argv) > 1:
        symbol = normalize_symbol(sys.argv[1])
    else:
        symbol = config.symbol_universe.core_symbols[0]

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)

    print("Initializing failover client + Bronze/Silver pipeline...")
    failover_client = _build_failover_client()
    
    # Use the new Database Recorder
    from src.db.silver_db_recorder import SilverDBRecorder
    from src.db.init_db import init_database
    
    # Ensure DB is initialized
    init_database()
    silver_recorder = SilverDBRecorder()
    
    bronze_recorder = BronzeRecorder()
    pipeline = SentinelIngestPipeline(
        client=failover_client,
        silver_recorder=silver_recorder,
        bronze_recorder=bronze_recorder,
        session_rules=config.session_rules,
    )

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    try:
        bars = pipeline.ingest_historical(symbol, start_date, end_date, interval="1h")
        print(f"Ingested {len(bars)} bars into Silver DB.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if not bars:
        print("No bars fetched. Exiting.")
        return

    # Verify Silver output in DB
    from src.db.queries import get_bars
    db_bars = get_bars(symbol, start_date, end_date, interval="1h")
    
    if not db_bars.empty:
        print(f"SUCCESS: Silver data found in database. {len(db_bars)} rows retrieved for {symbol}.")
    else:
        print(f"FAILURE: No data found in the database for {symbol}.")

    # Verify Bronze output
    bronze_files = list(Path("data/bronze").rglob("events.jsonl"))
    if bronze_files:
        print(f"SUCCESS: Bronze events captured: {len(bronze_files)} partition file(s).")
    else:
        print("FAILURE: No Bronze events.jsonl files found under data/bronze.")

    print(f"Current degradation state: {failover_client.degradation_state.value}")

if __name__ == "__main__":
    test_ingest()
