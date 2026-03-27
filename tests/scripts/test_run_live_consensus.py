from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from scripts import run_live_consensus
from src.db.models import Base, ConsensusSignalDB, RegimePredictionDB, SentimentScoreDB, TechnicalPredictionDB


def _build_sqlite_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'phase2_live_consensus.db'}"


def _seed_symbol_rows(db_url: str, *, symbol: str, include_symbol_sentiment: bool) -> datetime:
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    ts = datetime(2026, 3, 25, 9, 0, tzinfo=UTC)

    with SessionLocal() as session:
        session.add(
            TechnicalPredictionDB(
                symbol=symbol,
                timestamp=ts,
                price_forecast=101.2,
                direction="up",
                volatility_estimate=0.012,
                var_95=-0.01,
                var_99=-0.02,
                es_95=-0.015,
                es_99=-0.025,
                confidence=0.71,
                model_id="tech_v1",
                schema_version="1.0",
            )
        )
        session.add(
            RegimePredictionDB(
                symbol=symbol,
                timestamp=ts,
                regime_state="Bull",
                transition_probability=0.22,
                confidence=0.74,
                risk_level="full_risk",
                model_id="regime_v1",
                details_json='{"ood": {"warning": false, "alien": false}}',
                schema_version="1.0",
            )
        )
        session.add(
            SentimentScoreDB(
                symbol=symbol if include_symbol_sentiment else None,
                timestamp=ts,
                lane="slow",
                source_id="doc-1",
                source_type="rss_feed",
                sentiment_class="positive",
                sentiment_score=0.33,
                z_t=0.21,
                confidence=0.69,
                source_count=5,
                freshness_flag="fresh",
                model_id="finbert_v1",
                schema_version="1.0",
            )
        )
        session.commit()
    return ts


def test_run_live_consensus_persists_signal_for_explicit_symbol(tmp_path):
    db_url = _build_sqlite_url(tmp_path)
    _seed_symbol_rows(db_url, symbol="RELIANCE.NS", include_symbol_sentiment=True)

    output_path = tmp_path / "live_consensus_report.json"
    args = SimpleNamespace(
        symbols="RELIANCE.NS",
        db_url=db_url,
        sentiment_lane="slow",
        model_id="consensus_live_test_v1",
        output=str(output_path),
        strict=True,
    )

    payload = run_live_consensus.run_live_consensus(args)

    assert payload["failures"] == []
    assert len(payload["results"]) == 1
    assert payload["results"][0]["symbol"] == "RELIANCE.NS"
    assert payload["results"][0]["consensus"]["final_direction"] in {"buy", "sell", "neutral"}
    assert output_path.exists()

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with SessionLocal() as session:
        rows = session.execute(select(ConsensusSignalDB).order_by(ConsensusSignalDB.id.asc())).scalars().all()
    assert len(rows) == 1
    assert rows[0].symbol == "RELIANCE.NS"
    assert rows[0].model_id == "consensus_live_test_v1"


def test_run_live_consensus_uses_market_level_sentiment_fallback(tmp_path):
    db_url = _build_sqlite_url(tmp_path)
    fallback_ts = _seed_symbol_rows(db_url, symbol="TATASTEEL.NS", include_symbol_sentiment=False)

    args = SimpleNamespace(
        symbols="TATASTEEL.NS",
        db_url=db_url,
        sentiment_lane="slow",
        model_id="consensus_live_test_v1",
        output=str(tmp_path / "fallback.json"),
        strict=True,
    )

    payload = run_live_consensus.run_live_consensus(args)

    assert payload["failures"] == []
    assert len(payload["results"]) == 1
    assert payload["results"][0]["symbol"] == "TATASTEEL.NS"
    assert payload["results"][0]["sentiment_timestamp"].startswith(fallback_ts.strftime("%Y-%m-%dT%H:%M:%S"))
