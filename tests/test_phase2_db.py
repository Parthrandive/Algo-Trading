from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from src.agents.sentiment.schemas import SentimentLane
import src.db.queries as db_queries
from src.agents.regime.schemas import RegimePrediction, RegimeState, RiskLevel
from src.agents.technical.schemas import TechnicalPrediction
from src.db.models import (
    Base,
    BacktestRunDB,
    ConsensusSignalDB,
    ModelCardDB,
    PredictionLogDB,
    RegimePredictionDB,
    SentimentScoreDB,
    TechnicalPredictionDB,
)
from src.db.phase2_recorder import Phase2Recorder


def _build_sqlite_recorder():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    recorder = Phase2Recorder(engine=engine, session_factory=session_factory)
    return engine, session_factory, recorder


def test_phase2_recorder_bootstraps_missing_tables():
    engine = create_engine("sqlite:///:memory:")

    recorder = Phase2Recorder(engine=engine)

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    expected = {
        "technical_predictions",
        "regime_predictions",
        "sentiment_scores",
        "consensus_signals",
        "model_cards",
        "backtest_runs",
        "prediction_log",
    }

    assert expected.issubset(table_names)
    assert recorder.Session is not None


def test_phase2_tables_are_created():
    engine, _, _ = _build_sqlite_recorder()
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    expected = {
        "technical_predictions",
        "regime_predictions",
        "sentiment_scores",
        "consensus_signals",
        "model_cards",
        "backtest_runs",
        "prediction_log",
    }
    assert expected.issubset(table_names)


def test_sentiment_lane_schema_supports_daily_aggregate():
    lane_column = SentimentScoreDB.__table__.c.lane
    assert lane_column.type.length >= len(SentimentLane.DAILY_AGG.value)


def test_technical_prediction_upsert_and_prediction_log(monkeypatch):
    engine, session_factory, recorder = _build_sqlite_recorder()
    monkeypatch.setattr(db_queries, "get_engine", lambda: engine)

    ts = datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
    first = TechnicalPrediction(
        symbol="USDINR=X",
        timestamp=ts,
        price_forecast=83.21,
        direction="up",
        volatility_estimate=0.012,
        var_95=-0.01,
        var_99=-0.02,
        es_95=-0.015,
        es_99=-0.025,
        confidence=0.61,
        model_id="ensemble_arima_cnn_garch_v1.0",
    )
    second = first.model_copy(update={"price_forecast": 83.45, "confidence": 0.72})

    recorder.save_technical_prediction(first, latency_ms=12.3, data_snapshot_id="snap_1")
    recorder.save_technical_prediction(second, latency_ms=9.1, data_snapshot_id="snap_2")

    with session_factory() as session:
        stored = session.execute(select(TechnicalPredictionDB)).scalar_one()
        logs = session.execute(select(PredictionLogDB).order_by(PredictionLogDB.id.asc())).scalars().all()

    assert stored.price_forecast == 83.45
    assert stored.confidence == 0.72
    assert len(logs) == 2
    assert logs[0].agent == "technical"
    assert logs[1].data_snapshot_id == "snap_2"

    queried = db_queries.get_technical_predictions("USDINR=X", ts, ts)
    assert len(queried) == 1
    assert queried.iloc[0]["price_forecast"] == 83.45
    assert queried.iloc[0]["model_id"] == "ensemble_arima_cnn_garch_v1.0"

    audit = db_queries.get_prediction_log("technical", ts, ts)
    assert len(audit) == 2
    assert audit.iloc[0]["prediction_json"]["symbol"] == "USDINR=X"


def test_regime_prediction_round_trip(monkeypatch):
    engine, session_factory, recorder = _build_sqlite_recorder()
    monkeypatch.setattr(db_queries, "get_engine", lambda: engine)

    ts = datetime(2026, 3, 15, 9, 5, tzinfo=timezone.utc)
    prediction = RegimePrediction(
        symbol="RELIANCE.NS",
        timestamp=ts,
        regime_state=RegimeState.BULL,
        transition_probability=0.18,
        confidence=0.77,
        risk_level=RiskLevel.FULL_RISK,
        model_id="ensemble_hmm_pearl_ood_v1.0",
        details={"warning": False, "probabilities": {"Bull": 0.8, "Bear": 0.2}},
    )

    recorder.save_regime_prediction(prediction, latency_ms=4.2)

    with session_factory() as session:
        stored = session.execute(select(RegimePredictionDB)).scalar_one()
        assert stored.regime_state == "Bull"
        assert stored.risk_level == "full_risk"

    queried = db_queries.get_regime_predictions("RELIANCE.NS", ts, ts)
    assert len(queried) == 1
    assert queried.iloc[0]["details_json"]["probabilities"]["Bull"] == 0.8


def test_model_card_crud_and_backtest_query(monkeypatch):
    engine, _, recorder = _build_sqlite_recorder()
    monkeypatch.setattr(db_queries, "get_engine", lambda: engine)

    now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    card = {
        "model_id": "ensemble_arima_cnn_garch_v1.0",
        "agent": "technical",
        "model_family": "ensemble",
        "version": "1.0",
        "created_at": now,
        "updated_at": now,
        "performance": {"val_directional_accuracy": 0.56},
        "status": "active",
    }
    recorder.save_model_card(card)
    recorder.save_backtest_run(
        {
            "model_id": "ensemble_arima_cnn_garch_v1.0",
            "run_timestamp": now,
            "backtest_start": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "backtest_end": datetime(2025, 12, 31, tzinfo=timezone.utc),
            "sharpe": 1.2,
            "coverage": 0.94,
            "params": {"neutral_threshold": 0.0045},
            "notes": "baseline run",
        }
    )

    stored = db_queries.get_model_card("ensemble_arima_cnn_garch_v1.0")
    assert stored is not None
    assert stored["agent"] == "technical"
    assert stored["performance_json"]["val_directional_accuracy"] == 0.56

    backtests = db_queries.get_backtest_runs("ensemble_arima_cnn_garch_v1.0")
    assert len(backtests) == 1
    assert backtests.iloc[0]["params_json"]["neutral_threshold"] == 0.0045


def test_sentiment_and_consensus_upserts():
    _, session_factory, recorder = _build_sqlite_recorder()
    ts = datetime(2026, 3, 15, 11, 0, tzinfo=timezone.utc)

    recorder.save_sentiment_score(
        {
            "symbol": "INFOSYS.NS",
            "timestamp": ts,
            "lane": "slow",
            "sentiment_class": "positive",
            "sentiment_score": 0.42,
            "z_t": 1.3,
            "confidence": 0.81,
            "source_count": 17,
            "model_id": "finbert_indian_v1.0",
        }
    )
    recorder.save_consensus_signal(
        {
            "symbol": "INFOSYS.NS",
            "timestamp": ts,
            "final_direction": "buy",
            "final_confidence": 0.74,
            "technical_weight": 0.45,
            "regime_weight": 0.35,
            "sentiment_weight": 0.20,
            "crisis_mode": False,
            "agent_divergence": False,
            "transition_model": "LSTAR",
            "model_id": "consensus_v1.0",
        }
    )

    with session_factory() as session:
        assert session.execute(select(SentimentScoreDB)).scalar_one().sentiment_class == "positive"
        assert session.execute(select(ConsensusSignalDB)).scalar_one().final_direction == "buy"
        assert session.execute(select(BacktestRunDB)).scalars().all() == []
        assert session.execute(select(ModelCardDB)).scalars().all() == []
