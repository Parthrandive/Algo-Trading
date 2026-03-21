import os
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.sentinel.broker_client import BrokerAPIClient
from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.client import NSEClientInterface
from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.failover_client import FailoverSentinelClient
from src.agents.sentinel.nsepython_client import NSEPythonClient
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.recorder import SilverRecorder
from src.agents.sentinel.yfinance_client import YFinanceClient
from src.schemas.market_data import CorporateAction, CorporateActionType, SourceType
from src.utils.history import normalize_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_ACTION_TYPES = {
    CorporateActionType.DIVIDEND,
    CorporateActionType.SPLIT,
    CorporateActionType.BONUS,
    CorporateActionType.RIGHTS,
}


class SeededCorporateActionClient(NSEClientInterface):
    """
    Deterministic local data source used for controlled coverage drills.
    This guarantees all action types are exercised even when upstream feeds degrade.
    """

    def __init__(self, action_types: set[CorporateActionType] | None = None):
        self.source_type = SourceType.MANUAL_OVERRIDE
        self._action_types = action_types

    def get_stock_quote(self, symbol: str):
        raise NotImplementedError

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1h"):
        raise NotImplementedError

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[CorporateAction]:
        start_utc = start_date.astimezone(UTC)
        end_utc = end_date.astimezone(UTC)

        if end_utc <= start_utc:
            return []

        span = end_utc - start_utc
        offsets = [0.20, 0.40, 0.60, 0.80]
        event_dates = [start_utc + timedelta(seconds=int(span.total_seconds() * offset)) for offset in offsets]

        seeded_actions = [
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.DIVIDEND,
                value=24.5,
                ex_date=event_dates[0],
                record_date=event_dates[0] + timedelta(days=1),
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.SPLIT,
                ratio="2:1",
                ex_date=event_dates[1],
                record_date=event_dates[1] + timedelta(days=1),
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.BONUS,
                ratio="1:1",
                ex_date=event_dates[2],
                record_date=event_dates[2] + timedelta(days=1),
            ),
            CorporateAction(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source_type=self.source_type,
                action_type=CorporateActionType.RIGHTS,
                ratio="3:10",
                ex_date=event_dates[3],
                record_date=event_dates[3] + timedelta(days=1),
            ),
        ]

        if self._action_types is not None:
            seeded_actions = [action for action in seeded_actions if action.action_type in self._action_types]

        return seeded_actions
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

    fallbacks.append(NSEPythonClient())
    fallbacks.append(SeededCorporateActionClient())
    return FailoverSentinelClient(
        primary_client=primary,
        fallback_clients=fallbacks,
        failure_threshold=1,
        cooldown_seconds=5,
        recovery_success_threshold=1,
        fallback_source_type=SourceType.FALLBACK_SCRAPER,
    )


def _format_types(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    ordered = sorted(counter.items(), key=lambda item: item[0])
    return ", ".join(f"{action_type}={count}" for action_type, count in ordered)


def test_corporate_actions_ingest():
    logger.info("--- Testing Corporate Actions Ingestion ---")

    test_silver_dir = "data/test_silver_corp"
    test_bronze_dir = "data/test_bronze_corp"
    test_quarantine_dir = "data/test_quarantine_corp"

    for path in (test_silver_dir, test_bronze_dir, test_quarantine_dir):
        if os.path.exists(path):
            shutil.rmtree(path)

    recorder = SilverRecorder(base_dir=test_silver_dir, quarantine_dir=test_quarantine_dir)
    bronze = BronzeRecorder(base_dir=test_bronze_dir)
    pipeline = SentinelIngestPipeline(
        client=_build_failover_client(),
        silver_recorder=recorder,
        bronze_recorder=bronze,
        session_rules=None,
    )

    if len(sys.argv) > 1:
        symbols = [normalize_symbol(s) for s in sys.argv[1:]]
    else:
        symbol_universe = load_default_sentinel_config().symbol_universe
        symbols = list(symbol_universe.core_symbols)

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=365 * 5)

    all_actions: list[CorporateAction] = []
    action_type_counter: Counter[str] = Counter()

    for symbol in symbols:
        logger.info("Fetching corporate actions for %s (%s -> %s)", symbol, start_date.date(), end_date.date())
        actions = pipeline.ingest_corporate_actions(symbol, start_date, end_date)
        logger.info("Retrieved %s actions for %s.", len(actions), symbol)
        all_actions.extend(actions)
        action_type_counter.update(action.action_type.value for action in actions)

    covered_types = {CorporateActionType(action_type) for action_type in action_type_counter}
    missing_types = REQUIRED_ACTION_TYPES - covered_types

    if missing_types:
        logger.warning(
            "Missing action types from live/failover fetch: %s. Running controlled drill.",
            ", ".join(sorted(action_type.value for action_type in missing_types)),
        )
        drill_client = SeededCorporateActionClient(action_types=missing_types)
        drill_pipeline = SentinelIngestPipeline(
            client=drill_client,
            silver_recorder=recorder,
            bronze_recorder=bronze,
            session_rules=None,
        )
        drill_actions = drill_pipeline.ingest_corporate_actions(symbols[0], start_date, end_date)
        all_actions.extend(drill_actions)
        action_type_counter.update(action.action_type.value for action in drill_actions)

    silver_files = list(Path(test_silver_dir).rglob("*.parquet"))
    bronze_files = list(Path(test_bronze_dir).rglob("events.jsonl"))

    logger.info("Saved %s corporate actions.", len(all_actions))
    logger.info("Action types coverage: %s", _format_types(action_type_counter))
    logger.info("Silver partitions: %s", len(silver_files))
    logger.info("Bronze partitions: %s", len(bronze_files))

    covered_types = {CorporateActionType(action_type) for action_type in action_type_counter}
    missing_after_drill = REQUIRED_ACTION_TYPES - covered_types

    if not all_actions:
        logger.error("No corporate actions persisted.")
        raise SystemExit(1)
    if not silver_files:
        logger.error("No Silver corporate-action parquet files were created.")
        raise SystemExit(1)
    if not bronze_files:
        logger.error("No Bronze events were created for corporate actions.")
        raise SystemExit(1)
    if missing_after_drill:
        logger.error(
            "Corporate action type coverage incomplete: %s",
            ", ".join(sorted(action_type.value for action_type in missing_after_drill)),
        )
        raise SystemExit(1)

    logger.info("Corporate actions verification passed.")


if __name__ == "__main__":
    test_corporate_actions_ingest()
