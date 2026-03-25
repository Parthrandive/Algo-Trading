from datetime import datetime

from sqlalchemy import create_engine, text

import src.db.queries as db_queries


def test_get_sentiment_scores_supports_legacy_schema(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol VARCHAR(32),
                    timestamp DATETIME NOT NULL,
                    lane VARCHAR(16) NOT NULL,
                    sentiment_class VARCHAR(12) NOT NULL,
                    sentiment_score FLOAT NOT NULL,
                    z_t FLOAT,
                    confidence FLOAT NOT NULL,
                    source_count INTEGER NOT NULL DEFAULT 0,
                    ttl_seconds INTEGER,
                    freshness_flag VARCHAR(16),
                    headline_timestamp DATETIME,
                    score_timestamp DATETIME,
                    quality_status VARCHAR(8),
                    metadata_json TEXT,
                    model_id VARCHAR(128) NOT NULL,
                    schema_version VARCHAR(8) NOT NULL DEFAULT '1.0'
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO sentiment_scores (
                    symbol, timestamp, lane, sentiment_class, sentiment_score, z_t, confidence,
                    source_count, freshness_flag, metadata_json, model_id, schema_version
                ) VALUES (
                    'RELIANCE.NS', '2026-03-25 09:00:00', 'slow', 'positive', 0.42, 1.1, 0.88,
                    4, 'fresh', '{"source":"legacy"}', 'finbert_indian_v1.0', '1.0'
                )
                """
            )
        )

    monkeypatch.setattr(db_queries, "get_engine", lambda: engine)

    start = datetime(2026, 3, 25, 8, 59)
    end = datetime(2026, 3, 25, 9, 1)
    df = db_queries.get_sentiment_scores(symbol="RELIANCE.NS", start=start, end=end, lane="slow")

    assert len(df) == 1
    assert df.iloc[0]["sentiment_score"] == 0.42
    assert df.iloc[0]["metadata_json"]["source"] == "legacy"
