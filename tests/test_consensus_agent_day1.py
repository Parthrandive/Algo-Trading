from __future__ import annotations

from datetime import UTC, datetime

from src.agents.consensus import (
    AgentSignal,
    ConsensusAgent,
    ConsensusInput,
    ConsensusRiskMode,
    ConsensusTransitionModel,
)


def _payload(
    *,
    technical_score: float,
    regime_score: float,
    sentiment_score: float,
    volatility: float,
    crisis_probability: float,
) -> ConsensusInput:
    now = datetime.now(UTC)
    return ConsensusInput(
        technical=AgentSignal(name="technical", score=technical_score, confidence=0.72),
        regime=AgentSignal(name="regime", score=regime_score, confidence=0.75),
        sentiment=AgentSignal(name="sentiment", score=sentiment_score, confidence=0.68),
        volatility=volatility,
        macro_differential=0.2,
        rbi_signal=0.1,
        sentiment_quantile=0.60,
        crisis_probability=crisis_probability,
        generated_at_utc=now,
    )


def test_transition_model_switches_by_volatility():
    agent = ConsensusAgent()

    low_vol = _payload(
        technical_score=0.4,
        regime_score=0.2,
        sentiment_score=0.1,
        volatility=0.20,
        crisis_probability=0.2,
    )
    high_vol = _payload(
        technical_score=0.4,
        regime_score=0.2,
        sentiment_score=0.1,
        volatility=0.45,
        crisis_probability=0.2,
    )

    low_result = agent.run(low_vol)
    high_result = agent.run(high_vol)

    assert low_result.transition_model == ConsensusTransitionModel.LSTAR
    assert high_result.transition_model == ConsensusTransitionModel.ESTAR
    assert 0.0 <= low_result.transition_score <= 1.0
    assert 0.0 <= high_result.transition_score <= 1.0


def test_crisis_weight_is_capped_in_output_contract():
    agent = ConsensusAgent(max_crisis_weight=0.65)
    payload = _payload(
        technical_score=0.3,
        regime_score=0.2,
        sentiment_score=0.1,
        volatility=0.30,
        crisis_probability=0.95,
    )

    result = agent.run(payload)
    assert result.crisis_weight == 0.65


def test_divergence_triggers_protective_mode_and_neutral_score():
    agent = ConsensusAgent(divergence_warn_threshold=0.35, divergence_protective_threshold=0.70)
    payload = _payload(
        technical_score=1.0,
        regime_score=-1.0,
        sentiment_score=0.9,
        volatility=0.25,
        crisis_probability=0.1,
    )

    result = agent.run(payload)

    assert result.risk_mode == ConsensusRiskMode.PROTECTIVE
    assert result.score == 0.0
    assert result.divergence_score >= 0.70
