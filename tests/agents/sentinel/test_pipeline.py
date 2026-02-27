from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentinel.bronze_recorder import BronzeRecorder
from src.agents.sentinel.client import NSEClientInterface
from src.agents.sentinel.config import load_default_sentinel_config
from src.agents.sentinel.pipeline import SentinelIngestPipeline
from src.agents.sentinel.recorder import SilverRecorder
from src.schemas.market_data import Bar, CorporateAction, CorporateActionType, SourceType, Tick


class StaticClient(NSEClientInterface):
    def __init__(self, bars: list[Bar], actions: list[CorporateAction] | None = None):
        self._bars = bars
        self._actions = actions or []

    def get_stock_quote(self, symbol: str) -> Tick:
        return Tick(
            symbol=symbol,
            timestamp=datetime.now(ZoneInfo("UTC")),
            source_type=SourceType.OFFICIAL_API,
            price=100.0,
            volume=10,
        )

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1h") -> list[Bar]:
        return self._bars

    def get_corporate_actions(self, symbol: str, start_date: datetime, end_date: datetime) -> list[CorporateAction]:
        return self._actions


def test_pipeline_writes_bronze_and_silver(tmp_path):
    ist = ZoneInfo("Asia/Kolkata")
    symbol = "PIPELINE_TEST.NS"

    bars = [
        Bar(
            symbol=symbol,
            timestamp=datetime(2026, 2, 19, 9, 15, tzinfo=ist),
            source_type=SourceType.OFFICIAL_API,
            interval="1h",
            open=100,
            high=105,
            low=99,
            close=103,
            volume=1000,
        ),
        Bar(
            symbol=symbol,
            timestamp=datetime(2026, 2, 19, 8, 0, tzinfo=ist),  # Out of session
            source_type=SourceType.OFFICIAL_API,
            interval="1h",
            open=101,
            high=106,
            low=100,
            close=104,
            volume=900,
        ),
    ]

    client = StaticClient(bars=bars)
    silver = SilverRecorder(base_dir=str(tmp_path / "silver"), quarantine_dir=str(tmp_path / "quarantine"))
    bronze = BronzeRecorder(base_dir=str(tmp_path / "bronze"))
    config = load_default_sentinel_config()

    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver,
        bronze_recorder=bronze,
        session_rules=config.session_rules,
    )

    persisted = pipeline.ingest_historical(
        symbol=symbol,
        start_date=datetime(2026, 2, 18, 0, 0, tzinfo=ZoneInfo("UTC")),
        end_date=datetime(2026, 2, 20, 0, 0, tzinfo=ZoneInfo("UTC")),
        interval="1h",
    )

    assert len(persisted) == 1

    bronze_files = list((tmp_path / "bronze").rglob("events.jsonl"))
    assert bronze_files

    silver_files = list((tmp_path / "silver").rglob("*.parquet"))
    assert silver_files

    df = pd.read_parquet(silver_files[0])
    assert "source_type" in df.columns
    assert "ingestion_timestamp_utc" in df.columns
    assert "ingestion_timestamp_ist" in df.columns


def test_pipeline_writes_corporate_actions_to_bronze_and_silver(tmp_path):
    symbol = "PIPELINE_CORP.NS"
    ex_date = datetime(2026, 2, 19, 0, 0, tzinfo=ZoneInfo("UTC"))
    actions = [
        CorporateAction(
            symbol=symbol,
            timestamp=datetime(2026, 2, 20, 10, 0, tzinfo=ZoneInfo("UTC")),
            source_type=SourceType.BROKER_API,
            action_type=CorporateActionType.DIVIDEND,
            value=12.5,
            ex_date=ex_date,
            record_date=ex_date + timedelta(days=1),
        ),
        CorporateAction(
            symbol=symbol,
            timestamp=datetime(2026, 2, 20, 10, 0, tzinfo=ZoneInfo("UTC")),
            source_type=SourceType.BROKER_API,
            action_type=CorporateActionType.BONUS,
            ratio="1:1",
            ex_date=ex_date + timedelta(days=30),
            record_date=ex_date + timedelta(days=31),
        ),
    ]

    client = StaticClient(bars=[], actions=actions)
    silver = SilverRecorder(base_dir=str(tmp_path / "silver"), quarantine_dir=str(tmp_path / "quarantine"))
    bronze = BronzeRecorder(base_dir=str(tmp_path / "bronze"))
    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver,
        bronze_recorder=bronze,
        session_rules=None,
    )

    persisted = pipeline.ingest_corporate_actions(
        symbol=symbol,
        start_date=datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
        end_date=datetime(2026, 12, 31, tzinfo=ZoneInfo("UTC")),
    )
    assert len(persisted) == 2

    bronze_files = list((tmp_path / "bronze").rglob("events.jsonl"))
    assert bronze_files

    corp_files = list((tmp_path / "silver" / "corporate_actions").rglob("*.parquet"))
    assert corp_files

    df = pd.concat([pd.read_parquet(file) for file in corp_files], ignore_index=True)
    assert "action_type" in df.columns
    assert "ex_date" in df.columns
    assert "source_type" in df.columns


def test_pipeline_writes_quote_to_bronze_and_silver(tmp_path):
    symbol = "PIPELINE_TICK.NS"
    client = StaticClient(bars=[])
    silver = SilverRecorder(base_dir=str(tmp_path / "silver"), quarantine_dir=str(tmp_path / "quarantine"))
    bronze = BronzeRecorder(base_dir=str(tmp_path / "bronze"))
    pipeline = SentinelIngestPipeline(
        client=client,
        silver_recorder=silver,
        bronze_recorder=bronze,
        session_rules=None,
    )

    tick = pipeline.ingest_quote(symbol=symbol)

    assert tick.symbol == symbol

    bronze_files = list((tmp_path / "bronze").rglob("events.jsonl"))
    assert bronze_files

    tick_files = list((tmp_path / "silver" / "ticks").rglob("*.parquet"))
    assert tick_files

    df = pd.concat([pd.read_parquet(file) for file in tick_files], ignore_index=True)
    assert "price" in df.columns
    assert "volume" in df.columns
    assert "source_type" in df.columns
