from __future__ import annotations

from datetime import UTC, datetime

from src.agents.consensus import (
    AgentSignal,
    ConsensusAgent,
    ConsensusInput,
    ConsensusRegimeRiskLevel,
    ConsensusRiskMode,
    ConsensusTransitionModel,
)


def _payload(
    *,
    technical_score: float = 0.6,
    regime_score: float = 0.4,
    sentiment_score: float = 0.2,
    volatility: float = 0.2,
    crisis_probability: float = 0.1,
    sentiment_is_stale: bool = False,
    sentiment_is_missing: bool = False,
    regime_ood_warning: bool = False,
    regime_ood_alien: bool = False,
    regime_risk_level: ConsensusRegimeRiskLevel = ConsensusRegimeRiskLevel.FULL_RISK,
) -> ConsensusInput:
    return ConsensusInput(
        technical=AgentSignal(name="technical", score=technical_score, confidence=0.78),
        regime=AgentSignal(name="regime", score=regime_score, confidence=0.75),
        sentiment=AgentSignal(name="sentiment", score=sentiment_score, confidence=0.70),
        volatility=volatility,
        macro_differential=0.15,
        rbi_signal=0.05,
        sentiment_quantile=0.60,
        crisis_probability=crisis_probability,
        sentiment_is_stale=sentiment_is_stale,
        sentiment_is_missing=sentiment_is_missing,
        regime_ood_warning=regime_ood_warning,
        regime_ood_alien=regime_ood_alien,
        regime_risk_level=regime_risk_level,
        generated_at_utc=datetime.now(UTC),
    )


def test_consensus_score_is_bounded():
    agent = ConsensusAgent()
    result = agent.run(
        _payload(
            technical_score=1.0,
            regime_score=1.0,
            sentiment_score=1.0,
            crisis_probability=1.0,
            volatility=1.5,
        )
    )

    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_high_volatility_switches_to_estar():
    agent = ConsensusAgent()
    result = agent.run(_payload(volatility=0.5))

    assert result.transition_model == ConsensusTransitionModel.ESTAR


def test_crisis_probability_increases_regime_weight():
    agent = ConsensusAgent()
    low_crisis = agent.run(_payload(crisis_probability=0.05))
    high_crisis = agent.run(_payload(crisis_probability=0.90))

    assert high_crisis.weights["regime"] > low_crisis.weights["regime"]
    assert high_crisis.weights["technical"] < low_crisis.weights["technical"]


def test_text_flood_alone_does_not_flip_consensus_direction():
    agent = ConsensusAgent()
    result = agent.run(
        _payload(
            technical_score=0.95,
            regime_score=0.82,
            sentiment_score=-0.35,
            volatility=0.22,
            crisis_probability=0.10,
        )
    )

    assert result.score > 0.0


def test_max_divergence_forces_protective_mode():
    agent = ConsensusAgent(divergence_warn_threshold=0.35, divergence_protective_threshold=0.70)
    result = agent.run(
        _payload(
            technical_score=1.0,
            regime_score=-1.0,
            sentiment_score=0.9,
            volatility=0.25,
            crisis_probability=0.1,
        )
    )

    assert result.risk_mode == ConsensusRiskMode.PROTECTIVE
    assert result.score == 0.0


def test_stale_and_missing_sentiment_auto_reduce_weight():
    agent = ConsensusAgent()
    fresh = agent.run(_payload(sentiment_is_stale=False, sentiment_is_missing=False, sentiment_score=0.7))
    stale = agent.run(_payload(sentiment_is_stale=True, sentiment_is_missing=False, sentiment_score=0.7))
    missing = agent.run(_payload(sentiment_is_stale=True, sentiment_is_missing=True, sentiment_score=0.7))

    assert missing.weights["sentiment"] < stale.weights["sentiment"] < fresh.weights["sentiment"]


def test_ood_alien_state_triggers_protective_routing():
    agent = ConsensusAgent()
    result = agent.run(
        _payload(
            regime_ood_alien=True,
            regime_risk_level=ConsensusRegimeRiskLevel.NEUTRAL_CASH,
        )
    )

    assert result.risk_mode == ConsensusRiskMode.PROTECTIVE
    assert result.score == 0.0


def test_regime_warning_triggers_reduced_mode():
    agent = ConsensusAgent()
    result = agent.run(
        _payload(
            regime_ood_warning=True,
            regime_risk_level=ConsensusRegimeRiskLevel.REDUCED_RISK,
        )
    )

    assert result.risk_mode == ConsensusRiskMode.REDUCED
