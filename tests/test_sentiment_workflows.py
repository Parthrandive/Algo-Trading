from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.sentiment.models import FinBERTSentimentModel
from src.agents.sentiment.training import (
    load_examples_from_sources,
    persist_training_artifact,
    train_sklearn_sentiment_model,
)
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.db.models import Base, ModelCardDB, SentimentScoreDB, TextItemDB
from src.db.phase2_recorder import Phase2Recorder


def _build_sqlite_recorder():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    recorder = Phase2Recorder(engine=engine, session_factory=session_factory)
    return session_factory, recorder


def test_nightly_batch_persists_document_scores_and_daily_aggregates(tmp_path):
    session_factory, recorder = _build_sqlite_recorder()
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        session.add_all(
            [
                TextItemDB(
                    source_type="rss_feed",
                    source_id="doc_1",
                    timestamp=now - timedelta(hours=1),
                    content="RBI commentary keeps banking shares firm",
                    item_type="news",
                    headline="RBI commentary keeps banking shares firm",
                    publisher="Economic Times",
                    symbol="SBIN.NS",
                    language="en",
                    quality_status="pass",
                    ingestion_timestamp_utc=now,
                    ingestion_timestamp_ist=now,
                    schema_version="1.0",
                ),
                TextItemDB(
                    source_type="social_media",
                    source_id="doc_2",
                    timestamp=now - timedelta(hours=2),
                    content="Weak guidance hurts IT sentiment on the street",
                    item_type="social",
                    platform="X",
                    symbol="INFY.NS",
                    language="en",
                    quality_status="pass",
                    ingestion_timestamp_utc=now,
                    ingestion_timestamp_ist=now,
                    schema_version="1.0",
                ),
            ]
        )
        session.commit()

    agent = SentimentAgent.from_default_components(phase2_recorder=recorder, persist_predictions=True)
    batch = agent.run_nightly_batch(as_of_utc=now, lookback_hours=24)

    assert len(batch.document_predictions) == 2
    assert len(batch.symbol_aggregates) == 2
    assert batch.market_aggregate is not None
    assert agent.get_z_t("SBIN.NS") != 0.0 or agent.get_z_t("INFY.NS") != 0.0

    with session_factory() as session:
        stored_scores = session.execute(select(SentimentScoreDB)).scalars().all()
        stored_card = session.execute(select(ModelCardDB)).scalar_one()

    assert len(stored_scores) == 5
    assert stored_card.agent == "sentiment"


def test_local_training_artifact_can_bootstrap_slow_model(tmp_path):
    examples = load_examples_from_sources(include_bootstrap=True)
    pipeline, report = train_sklearn_sentiment_model(examples, seed=7, val_ratio=0.33)
    persist_training_artifact(
        output_dir=tmp_path,
        pipeline=pipeline,
        model_id="finbert_indian_v1",
        version="1.0",
        training_report=report,
        dataset_sizes={"synthetic_bootstrap": len(examples)},
        thresholds={
            "positive": {"precision_min": 0.75, "recall_min": 0.72},
            "neutral": {"precision_min": 0.70, "recall_min": 0.68},
            "negative": {"precision_min": 0.78, "recall_min": 0.74},
        },
        synthetic_data=True,
    )

    model = FinBERTSentimentModel.bootstrap(
        model_id="ProsusAI/finbert",
        enable_hf_pipeline=False,
        artifact_dir=tmp_path,
    )
    prediction = model.predict("RBI rate cut lifts banking stocks and market sentiment")

    assert model.using_fallback is False
    assert prediction.label.value in {"positive", "neutral", "negative"}
