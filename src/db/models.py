from datetime import datetime
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class OHLCVBar(Base):
    __tablename__ = "ohlcv_bars"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    interval = Column(String(8), primary_key=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    vwap = Column(Float, nullable=True)
    source_type = Column(String(32), nullable=False)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")

class TickData(Base):
    __tablename__ = "ticks"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    price = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False, default=0)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    source_type = Column(String(32), nullable=False)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")

class CorporateActionDB(Base):
    __tablename__ = "corporate_actions"
    ex_date = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    action_type = Column(String(16), primary_key=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    ratio = Column(String(16), nullable=True)
    value = Column(Float, nullable=True)
    record_date = Column(DateTime(timezone=True), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source_type = Column(String(32), nullable=False)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")

class MacroIndicatorDB(Base):
    __tablename__ = "macro_indicators"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    indicator_name = Column(String(32), primary_key=True)
    value = Column(Float, nullable=False)
    unit = Column(String(16), nullable=False)
    period = Column(String(16), nullable=False)
    region = Column(String(16), nullable=False, default="India")
    source_type = Column(String(32), nullable=False)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")

class TextItemDB(Base):
    __tablename__ = "text_items"
    source_type = Column(String(32), primary_key=True)
    source_id = Column(String(256), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    content = Column(Text, nullable=False)
    item_type = Column(String(16), nullable=False)
    language = Column(String(16), nullable=False, default="en")
    url = Column(Text, nullable=True)
    author = Column(String(128), nullable=True)
    headline = Column(Text, nullable=True)
    publisher = Column(String(128), nullable=True)
    platform = Column(String(32), nullable=True)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    symbol = Column(String(32), nullable=True)
    quarter = Column(String(8), nullable=True)
    year = Column(Integer, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")

class QuarantineBar(Base):
    __tablename__ = "quarantine"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    interval = Column(String(8), primary_key=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    vwap = Column(Float, nullable=True)
    source_type = Column(String(32), nullable=False)
    quality_status = Column(String(8), nullable=False, default="fail")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")
    reason = Column(String(64), nullable=False, default="monotonicity_violation")
    quarantined_at = Column(DateTime(timezone=True), primary_key=True)

class IngestionLog(Base):
    __tablename__ = "ingestion_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=True)
    data_type = Column(String(16), nullable=False)
    source_type = Column(String(32), nullable=True)
    records_ingested = Column(Integer, nullable=False, default=0)
    records_quarantined = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Float, nullable=True)
