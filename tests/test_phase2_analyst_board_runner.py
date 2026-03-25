from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.agents.consensus import ConsensusAgent
from src.agents.phase2_orchestrator import Phase2AnalystBoardRunner
from src.agents.regime.schemas import RegimePrediction, RegimeState, RiskLevel
from src.agents.sentiment.schemas import (
    CacheFreshnessState,
    SentimentLabel,
    SentimentLane,
    SentimentPrediction,
    SentimentQualityStatus,
)
from src.agents.technical.schemas import TechnicalPrediction
from src.db.models import Base, ConsensusSignalDB
from src.db.phase2_recorder import Phase2Recorder


def _build_sqlite_recorder():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory, Phase2Recorder(engine=engine, session_factory=session_factory)


class _FakeTechnicalAgent:
    def predict(self, symbol: str, *, limit: int, data_snapshot_id: str | None = None) -> TechnicalPrediction:
        return TechnicalPrediction(
            symbol=symbol,
            timestamp=datetime(2026, 3, 22, 9, 0, tzinfo=UTC),
            price_forecast=101.2,
            direction="up",
            volatility_estimate=0.012,
            var_95=-0.02,
            var_99=-0.03,
            es_95=-0.025,
            es_99=-0.035,
            confidence=0.78,
            model_id="ensemble_arima_cnn_garch_v1.0",
        )


class _FakeRegimeAgent:
    def detect_regime(self, symbol: str, *, limit: int, data_snapshot_id: str | None = None) -> RegimePrediction:
        return RegimePrediction(
            symbol=symbol,
            timestamp=datetime(2026, 3, 22, 9, 0, tzinfo=UTC),
            regime_state=RegimeState.BULL,
            transition_probability=0.18,
            confidence=0.74,
            risk_level=RiskLevel.FULL_RISK,
            model_id="ensemble_hmm_pearl_ood_v1.0",
            details={
                "ood": {"warning": False, "alien": False},
                "pearl": {
                    "probabilities": {
                        "Bull": 0.72,
                        "Bear": 0.08,
                        "Sideways": 0.10,
                        "Crisis": 0.04,
                        "RBI-Band transition": 0.04,
                        "Alien": 0.02,
                    }
                },
            },
        )


class _FakeSentimentAgent:
    def __init__(
        self,
        *,
        freshness_state: CacheFreshnessState = CacheFreshnessState.FRESH,
        quality_status: SentimentQualityStatus = SentimentQualityStatus.PASS,
    ) -> None:
        self.freshness_state = freshness_state
        self.quality_status = quality_status
        self.batch_calls = 0

    def get_cached_sentiment(self, symbol: str, *, as_of_utc: datetime | None = None) -> SentimentPrediction:
        expired = self.freshness_state == CacheFreshnessState.EXPIRED
        return SentimentPrediction(
            source_id=f"{symbol}-sentiment",
            text_hash="hash",
            lane=SentimentLane.FAST,
            label=SentimentLabel.POSITIVE,
            symbol=symbol,
            source_type="rss_feed",
            score=0.35,
            confidence=0.66,
            generated_at_utc=as_of_utc or datetime.now(UTC),
            score_timestamp_utc=as_of_utc or datetime.now(UTC),
            ttl_seconds=3600,
            model_name="keyword_rule_v1",
            freshness_state=self.freshness_state,
            quality_status=self.quality_status,
            reduced_risk_mode=expired,
            fallback_mode="technical_only_reduced_risk" if expired else None,
        )

    def get_z_t(self, symbol: str, *, as_of_utc: datetime | None = None) -> float:
        return 0.28

    def run_nightly_batch(self, *, as_of_utc: datetime, lookback_hours: int):
        self.batch_calls += 1
        return {"as_of_utc": as_of_utc.isoformat(), "lookback_hours": lookback_hours}


def test_phase2_runner_builds_full_payload_and_persists_consensus_signal():
    session_factory, recorder = _build_sqlite_recorder()
    consensus_agent = ConsensusAgent(phase2_recorder=recorder)
    runner = Phase2AnalystBoardRunner(
        technical_agent=_FakeTechnicalAgent(),
        regime_agent=_FakeRegimeAgent(),
        sentiment_agent=_FakeSentimentAgent(),
        consensus_agent=consensus_agent,
        context_provider=lambda symbol, limit: {
            "feature_timestamp_utc": "2026-03-22T09:00:00+00:00",
            "volatility": 0.012,
            "macro_differential": 0.14,
            "rbi_signal": -0.05,
        },
    )

    result = runner.run_symbol(
        symbol="SBIN.NS",
        snapshot_id="phase2_test_001",
        technical_limit=120,
        regime_limit=240,
        as_of_utc=datetime(2026, 3, 22, 9, 5, tzinfo=UTC),
        emit_phase3_observation=True,
    )

    assert result["technical"]["score"] > 0.0
    assert result["regime"]["score"] > 0.0
    assert result["sentiment"]["freshness_flag"] == "fresh"
    assert result["consensus"]["final_direction"] == "buy"
    assert result["phase3_observation"]["source_type"] == "internal_pipeline"
    assert abs(sum(result["phase3_observation"]["regime_probabilities"].values()) - 1.0) < 1e-6

    with session_factory() as session:
        stored = session.execute(select(ConsensusSignalDB)).scalar_one()

    assert stored.model_id == "consensus_weighted_v1"
    assert stored.final_direction == "buy"


def test_phase2_runner_marks_expired_sentiment_as_warn_but_keeps_pipeline_valid():
    _, recorder = _build_sqlite_recorder()
    consensus_agent = ConsensusAgent(phase2_recorder=recorder)
    runner = Phase2AnalystBoardRunner(
        technical_agent=_FakeTechnicalAgent(),
        regime_agent=_FakeRegimeAgent(),
        sentiment_agent=_FakeSentimentAgent(
            freshness_state=CacheFreshnessState.EXPIRED,
            quality_status=SentimentQualityStatus.WARN,
        ),
        consensus_agent=consensus_agent,
        context_provider=lambda symbol, limit: {
            "volatility": 0.010,
            "macro_differential": 0.05,
            "rbi_signal": 0.0,
        },
    )

    result = runner.run_symbol(
        symbol="INFY.NS",
        snapshot_id="phase2_test_002",
        technical_limit=120,
        regime_limit=240,
        as_of_utc=datetime(2026, 3, 22, 9, 10, tzinfo=UTC),
        emit_phase3_observation=True,
    )

    assert result["sentiment"]["freshness_flag"] == "expired"
    assert result["sentiment"]["source_count"] == 0
    assert result["context"]["quality_status"] == "warn"
    assert result["phase3_observation"]["quality_status"] == "warn"


def test_phase2_runner_can_refresh_sentiment_batch():
    sentiment_agent = _FakeSentimentAgent()
    runner = Phase2AnalystBoardRunner(sentiment_agent=sentiment_agent)

    payload = runner.refresh_sentiment_cache(
        as_of_utc=datetime(2026, 3, 22, 9, 15, tzinfo=UTC),
        lookback_hours=24,
    )

    assert sentiment_agent.batch_calls == 1
    assert payload["lookback_hours"] == 24
