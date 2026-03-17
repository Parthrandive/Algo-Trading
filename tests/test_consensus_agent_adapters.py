from __future__ import annotations

from src.agents.consensus import ConsensusAgent, build_consensus_input_from_phase2_payload
from src.agents.consensus.schemas import ConsensusRiskMode


def test_build_consensus_input_from_phase2_payload_maps_fields():
    payload = {
        "technical": {"score": 0.4, "confidence": 0.7, "is_protective": False},
        "regime": {"score": 0.2, "confidence": 0.8, "is_protective": True},
        "sentiment": {"score": -0.1, "confidence": 0.65, "is_protective": False},
        "context": {
            "volatility": 0.42,
            "macro_differential": 0.25,
            "rbi_signal": -0.1,
            "sentiment_quantile": 0.55,
            "crisis_probability": 0.4,
            "generated_at_utc": "2026-03-16T10:30:00+00:00",
        },
    }

    consensus_input = build_consensus_input_from_phase2_payload(payload)

    assert consensus_input.technical.score == 0.4
    assert consensus_input.regime.is_protective is True
    assert consensus_input.sentiment.confidence == 0.65
    assert consensus_input.volatility == 0.42
    assert consensus_input.crisis_probability == 0.4


def test_consensus_agent_loads_runtime_config_defaults_and_runs():
    agent = ConsensusAgent.from_default_components()
    payload = {
        "technical": {"score": 0.6, "confidence": 0.75},
        "regime": {"score": 0.5, "confidence": 0.72},
        "sentiment": {"score": 0.1, "confidence": 0.66},
        "context": {
            "volatility": 0.2,
            "macro_differential": 0.1,
            "rbi_signal": 0.05,
            "sentiment_quantile": 0.61,
            "crisis_probability": 0.15,
            "generated_at_utc": "2026-03-16T10:30:00+00:00",
        },
    }

    consensus_input = build_consensus_input_from_phase2_payload(payload)
    result = agent.run(consensus_input)

    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.transition_score <= 1.0
    assert result.risk_mode in {
        ConsensusRiskMode.NORMAL,
        ConsensusRiskMode.REDUCED,
        ConsensusRiskMode.PROTECTIVE,
    }
