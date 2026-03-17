from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from src.agents.consensus.schemas import AgentSignal, ConsensusInput


def build_consensus_input(
    *,
    technical: Mapping[str, Any],
    regime: Mapping[str, Any],
    sentiment: Mapping[str, Any],
    context: Mapping[str, Any],
) -> ConsensusInput:
    generated_at = _coerce_datetime(context.get("generated_at_utc"))
    return ConsensusInput(
        technical=_coerce_signal(technical, "technical"),
        regime=_coerce_signal(regime, "regime"),
        sentiment=_coerce_signal(sentiment, "sentiment"),
        volatility=_coerce_float(context.get("volatility"), default=0.0),
        macro_differential=_coerce_float(context.get("macro_differential"), default=0.0),
        rbi_signal=_coerce_float(context.get("rbi_signal"), default=0.0),
        sentiment_quantile=_coerce_float(context.get("sentiment_quantile"), default=0.5),
        crisis_probability=_coerce_float(context.get("crisis_probability"), default=0.0),
        generated_at_utc=generated_at,
    )


def build_consensus_input_from_phase2_payload(payload: Mapping[str, Any]) -> ConsensusInput:
    required_sections = ("technical", "regime", "sentiment", "context")
    missing = [key for key in required_sections if key not in payload]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"phase2 payload missing required sections: {missing_list}")

    technical = _expect_mapping(payload["technical"], name="technical")
    regime = _expect_mapping(payload["regime"], name="regime")
    sentiment = _expect_mapping(payload["sentiment"], name="sentiment")
    context = _expect_mapping(payload["context"], name="context")

    return build_consensus_input(
        technical=technical,
        regime=regime,
        sentiment=sentiment,
        context=context,
    )


def _coerce_signal(signal: Mapping[str, Any], default_name: str) -> AgentSignal:
    score = _coerce_float(signal.get("score"), default=0.0)
    confidence = _coerce_float(signal.get("confidence"), default=0.5)
    name = str(signal.get("name", default_name))
    is_protective = bool(signal.get("is_protective", False))
    return AgentSignal(
        name=name,
        score=score,
        confidence=confidence,
        is_protective=is_protective,
    )


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return datetime.now(UTC)


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _expect_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping")
