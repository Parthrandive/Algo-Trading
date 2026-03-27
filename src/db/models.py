from datetime import datetime, timezone
from uuid import uuid4
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
    lane = Column(String(16), nullable=False)
    source_id = Column(String(256), nullable=True)
    source_type = Column(String(32), nullable=True)
    sentiment_class = Column(String(12), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    z_t = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    source_count = Column(Integer, nullable=False, default=0)
    ttl_seconds = Column(Integer, nullable=True)
    freshness_flag = Column(String(16), nullable=True)
    headline_timestamp = Column(DateTime(timezone=True), nullable=True)
    score_timestamp = Column(DateTime(timezone=True), nullable=True)
    quality_status = Column(String(8), nullable=True)
    metadata_json = Column(Text, nullable=True)
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


class ObservationDB(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "snapshot_id", name="uq_observations_sym_ts_snapshot"),
    )

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    snapshot_id = Column(String(128), nullable=False)
    technical_direction = Column(String(8), nullable=False)
    technical_confidence = Column(Float, nullable=False)
    price_forecast = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    es_95 = Column(Float, nullable=False)
    regime_state = Column(String(32), nullable=False)
    regime_transition_prob = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    sentiment_z_t = Column(Float, nullable=True)
    consensus_direction = Column(String(8), nullable=False)
    consensus_confidence = Column(Float, nullable=False)
    crisis_mode = Column(Boolean, nullable=False, default=False)
    agent_divergence = Column(Boolean, nullable=False, default=False)
    orderbook_imbalance = Column(Float, nullable=True)
    queue_pressure = Column(Float, nullable=True)
    current_position = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    notional_exposure = Column(Float, nullable=False, default=0.0)
    portfolio_features_json = Column(Text, nullable=True)
    observation_schema_version = Column(String(8), nullable=False, default="1.0")
    quality_status = Column(String(8), nullable=False, default="pass")


StrategicObservationDB = ObservationDB


class RLPolicyDB(Base):
    __tablename__ = "rl_policies"

    policy_id = Column(String(128), primary_key=True)
    algorithm = Column(String(16), nullable=False)
    version = Column(String(16), nullable=False)
    status = Column(String(16), nullable=False, default="candidate")
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    artifact_path = Column(Text, nullable=False)
    observation_schema_version = Column(String(8), nullable=False)
    reward_function = Column(String(32), nullable=False)
    hyperparams_json = Column(Text, nullable=False)
    training_metrics_json = Column(Text, nullable=True)
    compression_method = Column(String(32), nullable=True)
    p99_inference_ms = Column(Float, nullable=True)
    p999_inference_ms = Column(Float, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class RLTrainingRunDB(Base):
    __tablename__ = "rl_training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(128), ForeignKey("rl_policies.policy_id"), nullable=False)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    training_start = Column(DateTime(timezone=True), nullable=False)
    training_end = Column(DateTime(timezone=True), nullable=False)
    episodes = Column(Integer, nullable=False)
    total_steps = Column(BigInteger, nullable=False)
    final_reward = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    sortino = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    dataset_snapshot_id = Column(String(128), nullable=True)
    code_hash = Column(String(64), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)


class TradeDecisionDB(Base):
    __tablename__ = "trade_decisions"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "policy_snapshot_id", name="uq_trade_decisions_sym_ts_snapshot"),
    )

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    observation_id = Column(Integer, nullable=True)
    observation_event_id = Column(String(64), nullable=True)
    observation_timestamp = Column(DateTime(timezone=True), nullable=True)
    policy_snapshot_id = Column(String(128), nullable=True)
    policy_id = Column(String(128), nullable=False)
    action = Column(String(16), nullable=False)
    action_size = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    entropy = Column(Float, nullable=True)
    sac_weight = Column(Float, nullable=True)
    ppo_weight = Column(Float, nullable=True)
    td3_weight = Column(Float, nullable=True)
    genetic_threshold = Column(Float, nullable=True)
    deliberation_used = Column(Boolean, nullable=False, default=False)
    risk_override = Column(String(16), nullable=True)
    risk_override_reason = Column(Text, nullable=True)
    decision_latency_ms = Column(Float, nullable=False, default=0.0)
    loop_type = Column(String(8), nullable=False, default="slow")
    policy_type = Column(String(16), nullable=False, default="teacher")
    is_placeholder = Column(Boolean, nullable=False, default=True)
    contract_version = Column(String(16), nullable=False, default="strat_exec_v1")
    schema_version = Column(String(8), nullable=False, default="1.0")


class RewardLogDB(Base):
    __tablename__ = "reward_logs"

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    decision_id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    policy_id = Column(String(128), nullable=False)
    reward_name = Column(String(32), nullable=False)  # backward-compatible alias
    reward_value = Column(Float, nullable=False)  # backward-compatible alias
    reward_function = Column(String(32), nullable=False, default="ra_drl_composite")
    total_reward = Column(Float, nullable=False, default=0.0)
    return_component = Column(Float, nullable=True)
    risk_penalty = Column(Float, nullable=True)
    regime_weight = Column(Float, nullable=True)
    sentiment_weight = Column(Float, nullable=True)
    transaction_cost = Column(Float, nullable=True)
    regime_state = Column(String(32), nullable=True)
    components_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class PolicySnapshotDB(Base):
    __tablename__ = "policy_snapshots"

    snapshot_id = Column(String(128), primary_key=True)
    policy_id = Column(String(128), nullable=False)
    policy_type = Column(String(16), nullable=False)
    generated_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    artifact_path = Column(Text, nullable=False)
    quality_status = Column(String(8), nullable=False, default="pass")
    source_type = Column(String(32), nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class OrderDB(Base):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("order_id", "created_at", name="uq_orders_orderid_created"),
    )

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    created_at = Column(DateTime(timezone=True), primary_key=True, nullable=False, default=lambda: datetime.now(timezone.utc))
    id = Column(Integer, nullable=True)
    order_id = Column(String(128), nullable=False)
    decision_id = Column(Integer, nullable=True)
    decision_event_id = Column(String(64), nullable=True)
    decision_timestamp = Column(DateTime(timezone=True), nullable=True)
    symbol = Column(String(32), nullable=False)
    exchange = Column(String(16), nullable=False, default="NSE")
    product_type = Column(String(16), nullable=False, default="equity")
    order_type = Column(String(16), nullable=False)
    side = Column(String(4), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)
    trigger_price = Column(Float, nullable=True)
    status = Column(String(16), nullable=False, default="pending")
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    broker_order_id = Column(String(128), nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    filled_quantity = Column(Integer, nullable=False, default=0)
    slippage_bps = Column(Float, nullable=True)
    model_version = Column(String(128), nullable=False)
    compliance_check_passed = Column(Boolean, nullable=False, default=True)
    rejection_reason = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    schema_version = Column(String(8), nullable=False, default="1.0")


class OrderFillDB(Base):
    __tablename__ = "order_fills"

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    fill_timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    order_id = Column(String(128), nullable=False)
    order_event_id = Column(String(64), nullable=True)
    order_created_at = Column(DateTime(timezone=True), nullable=True)
    fill_price = Column(Float, nullable=False)
    fill_quantity = Column(Integer, nullable=False)
    exchange_trade_id = Column(String(128), nullable=True)
    fees = Column(Float, nullable=True)
    impact_cost_bps = Column(Float, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class PortfolioSnapshotDB(Base):
    __tablename__ = "portfolio_snapshots"

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    position_qty = Column(Integer, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    market_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl_session = Column(Float, nullable=False)
    realized_pnl_cumulative = Column(Float, nullable=False)
    notional_exposure = Column(Float, nullable=False)
    net_exposure = Column(Float, nullable=False)
    gross_exposure = Column(Float, nullable=False)
    sector = Column(String(32), nullable=True)
    risk_budget_used_pct = Column(Float, nullable=True)
    operating_mode = Column(String(16), nullable=False, default="normal")
    decision_id = Column(Integer, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class StudentPolicyDB(Base):
    __tablename__ = "student_policies"

    student_id = Column(String(128), primary_key=True)
    teacher_policy_id = Column(String(128), ForeignKey("rl_policies.policy_id"), nullable=False)
    version = Column(String(16), nullable=False)
    status = Column(String(16), nullable=False, default="candidate")
    compression_method = Column(String(32), nullable=False)
    compression_ratio = Column(Float, nullable=True)
    teacher_agreement_pct = Column(Float, nullable=False)
    crisis_agreement_pct = Column(Float, nullable=False)
    p99_inference_ms = Column(Float, nullable=False)
    p999_inference_ms = Column(Float, nullable=False)
    artifact_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    demoted_at = Column(DateTime(timezone=True), nullable=True)
    drift_threshold = Column(Float, nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class DistillationRunDB(Base):
    __tablename__ = "distillation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(128), ForeignKey("student_policies.student_id"), nullable=False)
    teacher_policy_id = Column(String(128), ForeignKey("rl_policies.policy_id"), nullable=False)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    epochs = Column(Integer, nullable=False)
    avg_day_agreement = Column(Float, nullable=False)
    crisis_slice_agreement = Column(Float, nullable=False)
    kl_divergence = Column(Float, nullable=True)
    inference_latency_p99 = Column(Float, nullable=False)
    dataset_snapshot_id = Column(String(128), nullable=True)
    code_hash = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class DeliberationLogDB(Base):
    __tablename__ = "deliberation_logs"

    event_id = Column(String(64), primary_key=True, default=lambda: uuid4().hex)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    deliberation_type = Column(String(32), nullable=False)
    input_snapshot_id = Column(String(128), nullable=True)
    output_snapshot_id = Column(String(128), nullable=True)
    duration_ms = Column(Float, nullable=False)
    result_json = Column(Text, nullable=True)
    triggered_refresh = Column(Boolean, nullable=False, default=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class ImpactEventDB(Base):
    __tablename__ = "impact_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    bucket = Column(String(32), nullable=False)
    event_type = Column(String(32), nullable=False)
    breach = Column(Boolean, nullable=False, default=False)
    participation_rate = Column(Float, nullable=False, default=0.0)
    slippage_delta_bps = Column(Float, nullable=False, default=0.0)
    impact_score = Column(Float, nullable=False, default=0.0)
    size_multiplier = Column(Float, nullable=False, default=1.0)
    cooldown_until = Column(DateTime(timezone=True), nullable=True)
    risk_override = Column(String(16), nullable=True)
    reasons_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class RiskCapEventDB(Base):
    __tablename__ = "risk_cap_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    asset_cluster = Column(String(64), nullable=False)
    regime = Column(String(16), nullable=False)
    cap_fraction = Column(Float, nullable=False)
    changed = Column(Boolean, nullable=False, default=False)
    event_type = Column(String(32), nullable=False)
    false_trigger_rate = Column(Float, nullable=False, default=0.0)
    auto_adjustment_paused = Column(Boolean, nullable=False, default=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class OrderBookFeatureEventDB(Base):
    __tablename__ = "orderbook_feature_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    quality_flag = Column(String(16), nullable=False)
    degraded = Column(Boolean, nullable=False, default=False)
    degradation_reason = Column(String(64), nullable=True)
    imbalance = Column(Float, nullable=False, default=0.0)
    queue_pressure = Column(Float, nullable=False, default=0.0)
    schema_version = Column(String(8), nullable=False, default="1.0")


class FastLoopLatencyEventDB(Base):
    __tablename__ = "fastloop_latency_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    stage = Column(String(64), nullable=False)
    mode = Column(String(16), nullable=False, default="normal")
    event_type = Column(String(32), nullable=False)
    reason = Column(String(128), nullable=False, default="")
    sample_count = Column(Integer, nullable=False, default=0)
    p50_ms = Column(Float, nullable=False, default=0.0)
    p95_ms = Column(Float, nullable=False, default=0.0)
    p99_ms = Column(Float, nullable=False, default=0.0)
    p999_ms = Column(Float, nullable=False, default=0.0)
    jitter_ms = Column(Float, nullable=False, default=0.0)
    schema_version = Column(String(8), nullable=False, default="1.0")


class LatencyBenchmarkArtifactDB(Base):
    __tablename__ = "latency_benchmark_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(128), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    passed = Column(Boolean, nullable=False, default=False)
    reasons_json = Column(Text, nullable=True)
    artifact_json = Column(Text, nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class PromotionGateEventDB(Base):
    __tablename__ = "promotion_gate_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    policy_id = Column(String(128), nullable=False)
    from_stage = Column(String(16), nullable=False)
    to_stage = Column(String(16), nullable=False)
    approved = Column(Boolean, nullable=False, default=False)
    reasons_json = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class RollbackDrillEventDB(Base):
    __tablename__ = "rollback_drill_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    failed_model_id = Column(String(128), nullable=False)
    reverted_to = Column(String(128), nullable=True)
    executed = Column(Boolean, nullable=False, default=False)
    mttr_seconds = Column(Float, nullable=False, default=0.0)
    reasons_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class XAITradeExplanationDB(Base):
    __tablename__ = "xai_trade_explanations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(128), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    top_feature_contributions_json = Column(Text, nullable=False)
    agent_contributions_json = Column(Text, nullable=False)
    signal_family_contributions_json = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)
    schema_version = Column(String(8), nullable=False, default="1.0")


class PnLAttributionEventDB(Base):
    __tablename__ = "pnl_attribution_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(128), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    sector = Column(String(64), nullable=False)
    agent = Column(String(64), nullable=False)
    signal_family = Column(String(64), nullable=False)
    realized_pnl = Column(Float, nullable=False)
    schema_version = Column(String(8), nullable=False, default="1.0")


class OperationalMetricsSnapshotDB(Base):
    __tablename__ = "operational_metrics_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    decision_staleness_avg_s = Column(Float, nullable=False, default=0.0)
    feature_lag_avg_s = Column(Float, nullable=False, default=0.0)
    mode_switch_frequency = Column(Integer, nullable=False, default=0)
    ood_trigger_rate = Column(Integer, nullable=False, default=0)
    kill_switch_false_positives = Column(Integer, nullable=False, default=0)
    mttr_avg_s = Column(Float, nullable=False, default=0.0)
    schema_version = Column(String(8), nullable=False, default="1.0")
