from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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
    schema_version = Column(String(8), nullable=False, default="1.1")

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
    dataset_snapshot_id = Column(String(128), nullable=True)
    code_hash = Column(String(64), nullable=True)


class LiveMarketObservationDB(Base):
    __tablename__ = "live_market_observations"
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(32), primary_key=True)
    observation_kind = Column(String(16), primary_key=True, default="quote")
    interval = Column(String(8), primary_key=True, default="")
    exchange = Column(String(16), nullable=False, default="NSE")
    asset_type = Column(String(16), nullable=False, default="equity")
    last_price = Column(Float, nullable=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(BigInteger, nullable=True)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    bar_timestamp = Column(DateTime(timezone=True), nullable=True)
    is_final_bar = Column(Boolean, nullable=False, default=False)
    source_type = Column(String(32), nullable=False)
    source_name = Column(String(64), nullable=True)
    source_status = Column(String(16), nullable=False, default="ok")
    freshness_status = Column(String(16), nullable=False, default="unknown")
    staleness_seconds = Column(Float, nullable=True)
    quality_status = Column(String(8), nullable=False, default="pass")
    ingestion_timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    ingestion_timestamp_ist = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class MarketDataQualityDB(Base):
    __tablename__ = "market_data_quality"
    symbol = Column(String(32), primary_key=True)
    interval = Column(String(8), primary_key=True)
    dataset_type = Column(String(16), primary_key=True)
    exchange = Column(String(16), nullable=False, default="NSE")
    asset_type = Column(String(16), nullable=False, default="equity")
    status = Column(String(16), nullable=False, default="partial")
    train_ready = Column(Boolean, nullable=False, default=False)
    first_timestamp = Column(DateTime(timezone=True), nullable=True)
    last_timestamp = Column(DateTime(timezone=True), nullable=True)
    row_count = Column(Integer, nullable=False, default=0)
    duplicate_count = Column(Integer, nullable=False, default=0)
    expected_rows = Column(Integer, nullable=True)
    missing_intervals = Column(Integer, nullable=True)
    gap_count = Column(Integer, nullable=True)
    largest_gap_intervals = Column(Integer, nullable=True)
    zero_volume_ratio = Column(Float, nullable=True)
    coverage_pct = Column(Float, nullable=True)
    history_days = Column(Integer, nullable=True)
    source_name = Column(String(128), nullable=True)
    source_type = Column(String(32), nullable=True)
    details_json = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class TechnicalPredictionDB(Base):
    __tablename__ = "technical_predictions"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "model_id", name="uq_technical_predictions_sym_ts_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    price_forecast = Column(Float, nullable=False)
    direction = Column(String(8), nullable=False)
    volatility_estimate = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    var_99 = Column(Float, nullable=False)
    es_95 = Column(Float, nullable=False)
    es_99 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_id = Column(String(128), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class RegimePredictionDB(Base):
    __tablename__ = "regime_predictions"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "model_id", name="uq_regime_predictions_sym_ts_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    regime_state = Column(String(32), nullable=False)
    transition_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    model_id = Column(String(128), nullable=False)
    details_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class SentimentScoreDB(Base):
    __tablename__ = "sentiment_scores"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "lane", "model_id", name="uq_sentiment_scores_sym_ts_lane_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    lane = Column(String(8), nullable=False)
    sentiment_class = Column(String(12), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    z_t = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    source_count = Column(Integer, nullable=False, default=0)
    model_id = Column(String(128), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class ConsensusSignalDB(Base):
    __tablename__ = "consensus_signals"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "model_id", name="uq_consensus_signals_sym_ts_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    final_direction = Column(String(8), nullable=False)
    final_confidence = Column(Float, nullable=False)
    technical_weight = Column(Float, nullable=False)
    regime_weight = Column(Float, nullable=False)
    sentiment_weight = Column(Float, nullable=False)
    crisis_mode = Column(Boolean, nullable=False, default=False)
    agent_divergence = Column(Boolean, nullable=False, default=False)
    transition_model = Column(String(8), nullable=False)
    model_id = Column(String(128), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class ModelCardDB(Base):
    __tablename__ = "model_cards"

    model_id = Column(String(128), primary_key=True)
    agent = Column(String(32), nullable=False)
    model_family = Column(String(32), nullable=False)
    version = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    metadata_json = Column(Text, nullable=False)
    performance_json = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="active")


class BacktestRunDB(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(128), ForeignKey("model_cards.model_id"), nullable=False)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    backtest_start = Column(DateTime(timezone=True), nullable=False)
    backtest_end = Column(DateTime(timezone=True), nullable=False)
    sharpe = Column(Float, nullable=True)
    sortino = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    coverage = Column(Float, nullable=True)
    params_json = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)


class PredictionLogDB(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent = Column(String(32), nullable=False)
    symbol = Column(String(32), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    prediction_json = Column(Text, nullable=False)
    model_id = Column(String(128), nullable=False)
    latency_ms = Column(Float, nullable=True)
    data_snapshot_id = Column(String(128), nullable=True)
