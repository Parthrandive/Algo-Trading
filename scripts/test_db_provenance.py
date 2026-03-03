import asyncio
from datetime import UTC, datetime
from src.schemas.market_data import Tick, SourceType, QualityFlag
from src.db.silver_db_recorder import SilverDBRecorder
from src.db.models import Base
from src.db.connection import get_engine, get_session
import os

DB_URL = "sqlite:///data/test_provenance.db"
os.makedirs("data", exist_ok=True)
engine = get_engine(DB_URL)
Base.metadata.create_all(engine)

recorder = SilverDBRecorder(DB_URL)

tick = Tick(
    symbol="RELIANCE",
    timestamp=datetime.now(UTC),
    source_type=SourceType.BROKER_API,
    price=2500.0,
    volume=100,
    quality_status=QualityFlag.PASS
)

print("Pre-save Tick Dump:")
print(tick.model_dump(by_alias=True))

recorder.save_ticks([tick])

print("\n--- DB Contents ---")
from sqlalchemy import text
Session = get_session(engine)
with Session() as session:
    results = session.execute(text("SELECT symbol, timestamp, source_type, schema_version, ingestion_timestamp_utc, ingestion_timestamp_ist FROM ticks")).fetchall()
    for row in results:
        print(f"{row.symbol} | Source: {row.source_type} | Schema: {row.schema_version} | UTC: {row.ingestion_timestamp_utc} | IST: {row.ingestion_timestamp_ist}")
