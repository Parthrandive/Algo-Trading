from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from src.agents.consensus.schemas import (
    AgentSignal,
    ConsensusInput,
    ConsensusRegimeRiskLevel,
)


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
        sentiment_is_stale=_coerce_sentiment_stale(sentiment, context),
        sentiment_is_missing=_coerce_sentiment_missing(sentiment, context),
        regime_ood_warning=_coerce_bool(
            regime.get("ood_warning", context.get("regime_ood_warning")),
            default=False,
        ),
        regime_ood_alien=_coerce_bool(
            regime.get("ood_alien", context.get("regime_ood_alien")),
            default=False,
        ),
        regime_risk_level=_coerce_regime_risk_level(regime, context),
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


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "stale", "missing"}:
            return True
        if normalized in {"false", "0", "no", "n", "fresh"}:
            return False
    return default


def _coerce_sentiment_stale(sentiment: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    freshness = sentiment.get("freshness_flag", context.get("sentiment_freshness"))
    if isinstance(freshness, str) and freshness.strip().lower() in {"stale", "missing", "expired"}:
        return True
    return _coerce_bool(sentiment.get("is_stale", context.get("sentiment_is_stale")), default=False)


def _coerce_sentiment_missing(sentiment: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    freshness = sentiment.get("freshness_flag", context.get("sentiment_freshness"))
    if isinstance(freshness, str) and freshness.strip().lower() in {"missing", "expired"}:
        return True
    source_count = sentiment.get("source_count", context.get("sentiment_source_count"))
    if source_count is not None and _coerce_float(source_count, default=0.0) <= 0.0:
        return True
    return _coerce_bool(sentiment.get("is_missing", context.get("sentiment_is_missing")), default=False)


def _coerce_regime_risk_level(
    regime: Mapping[str, Any],
    context: Mapping[str, Any],
) -> ConsensusRegimeRiskLevel:
    value = regime.get("risk_level", context.get("regime_risk_level", ConsensusRegimeRiskLevel.FULL_RISK.value))
    if isinstance(value, ConsensusRegimeRiskLevel):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in ConsensusRegimeRiskLevel:
            if member.value == normalized:
                return member
    return ConsensusRegimeRiskLevel.FULL_RISK


def _expect_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping")
