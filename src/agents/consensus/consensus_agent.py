from __future__ import annotations

import json
import math
from pathlib import Path

from src.agents.consensus.schemas import (
    ConsensusInput,
    ConsensusOutput,
    ConsensusRiskMode,
    ConsensusTransitionModel,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "configs" / "consensus_agent_runtime_v1.json"


class ConsensusAgent:
    def __init__(
        self,
        *,
        volatility_switch_threshold: float = 0.35,
        lstar_logistic_gain: float = 4.0,
        max_crisis_weight: float = 0.7,
        divergence_warn_threshold: float = 0.45,
        divergence_protective_threshold: float = 0.75,
        safety_bias_boost: float = 0.1,
        reduced_mode_scale: float = 0.5,
        technical_base_weight: float = 0.42,
        regime_base_weight: float = 0.35,
        sentiment_base_weight: float = 0.23,
    ):
        self.volatility_switch_threshold = max(0.0, volatility_switch_threshold)
        self.lstar_logistic_gain = max(0.1, lstar_logistic_gain)
        self.max_crisis_weight = self._clamp(max_crisis_weight, 0.0, 1.0)
        self.divergence_warn_threshold = self._clamp(divergence_warn_threshold, 0.0, 1.0)
        self.divergence_protective_threshold = self._clamp(divergence_protective_threshold, 0.0, 1.0)
        self.safety_bias_boost = self._clamp(safety_bias_boost, 0.0, 0.5)
        self.reduced_mode_scale = self._clamp(reduced_mode_scale, 0.1, 1.0)
        self.technical_base_weight = max(0.01, technical_base_weight)
        self.regime_base_weight = max(0.01, regime_base_weight)
        self.sentiment_base_weight = max(0.01, sentiment_base_weight)

    @classmethod
    def from_default_components(cls, runtime_config_path: Path | None = None) -> "ConsensusAgent":
        config_path = runtime_config_path or DEFAULT_RUNTIME_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            runtime_config = json.load(handle)

        transition = runtime_config.get("transition", {})
        routing = runtime_config.get("routing", {})
        risk_modes = runtime_config.get("risk_modes", {})
        weights = runtime_config.get("weights", {})

        return cls(
            volatility_switch_threshold=float(transition.get("volatility_switch_threshold", 0.35)),
            lstar_logistic_gain=float(transition.get("lstar_logistic_gain", 4.0)),
            max_crisis_weight=float(routing.get("max_crisis_weight", 0.7)),
            safety_bias_boost=float(routing.get("safety_bias_boost", 0.1)),
            divergence_warn_threshold=float(risk_modes.get("divergence_warn_threshold", 0.45)),
            divergence_protective_threshold=float(risk_modes.get("divergence_protective_threshold", 0.75)),
            reduced_mode_scale=float(risk_modes.get("reduced_mode_scale", 0.5)),
            technical_base_weight=float(weights.get("technical_base", 0.42)),
            regime_base_weight=float(weights.get("regime_base", 0.35)),
            sentiment_base_weight=float(weights.get("sentiment_base", 0.23)),
        )

    def run(self, payload: ConsensusInput) -> ConsensusOutput:
        transition_model, transition_score = self._select_transition(payload)
        weights = self._compute_weights(payload=payload, transition_score=transition_score)

        raw_score = (
            (payload.technical.score * weights["technical"])
            + (payload.regime.score * weights["regime"])
            + (payload.sentiment.score * weights["sentiment"])
        )
        divergence_score = self._compute_divergence(payload)
        risk_mode = self._resolve_risk_mode(divergence_score)

        score = self._apply_risk_mode(raw_score, risk_mode)
        confidence = self._compute_confidence(payload=payload, transition_score=transition_score)

        return ConsensusOutput(
            score=self._clamp(score, -1.0, 1.0),
            confidence=self._clamp(confidence, 0.0, 1.0),
            transition_score=self._clamp(transition_score, 0.0, 1.0),
            transition_model=transition_model,
            risk_mode=risk_mode,
            divergence_score=divergence_score,
            crisis_weight=min(payload.crisis_probability, self.max_crisis_weight),
            weights=weights,
            generated_at_utc=payload.generated_at_utc,
        )

    def compute_lstar_transition(self, payload: ConsensusInput) -> float:
        signal = (
            (0.55 * payload.volatility)
            + (0.20 * abs(payload.macro_differential))
            + (0.15 * abs(payload.rbi_signal))
            + (0.10 * payload.sentiment_quantile)
        )
        centered = signal - 0.5
        return 1.0 / (1.0 + math.exp(-(self.lstar_logistic_gain * centered)))

    def compute_estar_transition(self, payload: ConsensusInput) -> float:
        x = (
            (0.50 * payload.volatility)
            + (0.25 * abs(payload.macro_differential))
            + (0.15 * abs(payload.rbi_signal))
            + (0.10 * payload.sentiment_quantile)
        )
        return 1.0 - math.exp(-(x**2))

    def _select_transition(self, payload: ConsensusInput) -> tuple[ConsensusTransitionModel, float]:
        if payload.volatility >= self.volatility_switch_threshold:
            return ConsensusTransitionModel.ESTAR, self.compute_estar_transition(payload)
        return ConsensusTransitionModel.LSTAR, self.compute_lstar_transition(payload)

    def _compute_weights(self, *, payload: ConsensusInput, transition_score: float) -> dict[str, float]:
        technical_weight = self.technical_base_weight + (0.08 * (1.0 - transition_score))
        regime_weight = self.regime_base_weight + (0.10 * transition_score)
        sentiment_weight = self.sentiment_base_weight - (0.04 * transition_score)

        crisis_weight = min(payload.crisis_probability, self.max_crisis_weight)
        regime_weight += 0.15 * crisis_weight
        technical_weight -= 0.10 * crisis_weight
        sentiment_weight -= 0.05 * crisis_weight

        if payload.technical.is_protective:
            technical_weight += self.safety_bias_boost
        if payload.regime.is_protective:
            regime_weight += self.safety_bias_boost
        if payload.sentiment.is_protective:
            sentiment_weight += self.safety_bias_boost

        technical_weight = max(0.01, technical_weight)
        regime_weight = max(0.01, regime_weight)
        sentiment_weight = max(0.01, sentiment_weight)

        total = technical_weight + regime_weight + sentiment_weight
        return {
            "technical": technical_weight / total,
            "regime": regime_weight / total,
            "sentiment": sentiment_weight / total,
        }

    def _compute_divergence(self, payload: ConsensusInput) -> float:
        scores = [payload.technical.score, payload.regime.score, payload.sentiment.score]
        spread = max(scores) - min(scores)
        return self._clamp(spread / 2.0, 0.0, 1.0)

    def _resolve_risk_mode(self, divergence_score: float) -> ConsensusRiskMode:
        if divergence_score >= self.divergence_protective_threshold:
            return ConsensusRiskMode.PROTECTIVE
        if divergence_score >= self.divergence_warn_threshold:
            return ConsensusRiskMode.REDUCED
        return ConsensusRiskMode.NORMAL

    def _apply_risk_mode(self, score: float, risk_mode: ConsensusRiskMode) -> float:
        if risk_mode == ConsensusRiskMode.PROTECTIVE:
            return 0.0
        if risk_mode == ConsensusRiskMode.REDUCED:
            return score * self.reduced_mode_scale
        return score

    def _compute_confidence(self, *, payload: ConsensusInput, transition_score: float) -> float:
        base_confidence = (
            (payload.technical.confidence + payload.regime.confidence + payload.sentiment.confidence) / 3.0
        )
        confidence_penalty = 0.25 * payload.crisis_probability
        confidence_bonus = 0.10 * (1.0 - transition_score)
        return base_confidence - confidence_penalty + confidence_bonus

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))
