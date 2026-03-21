from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from src.agents.strategic_executive.schemas import (
    ConsensusSnapshot,
    MacroSnapshot,
    MicrostructureSnapshot,
    Phase3Observation,
    QualityStatus,
    RegimeProbabilities,
    SentimentSnapshot,
    SourceType,
    TechnicalRiskSnapshot,
)

_REGIME_KEY_ALIASES = {
    "bull": "bull",
    "bear": "bear",
    "sideways": "sideways",
    "crisis": "crisis",
    "rbi_band_transition": "rbi_band_transition",
    "rbi_band": "rbi_band_transition",
    "rbi_band_shift": "rbi_band_transition",
    "rbi-band_transition": "rbi_band_transition",
    "rbi-band transition": "rbi_band_transition",
    "rbi_band_transition_state": "rbi_band_transition",
    "alien": "alien",
}

_REGIME_STATE_TO_KEY = {
    "bull": "bull",
    "bear": "bear",
    "sideways": "sideways",
    "crisis": "crisis",
    "rbi-band transition": "rbi_band_transition",
    "rbi_band_transition": "rbi_band_transition",
    "alien": "alien",
}


def build_phase3_observation_from_phase2_payload(payload: Mapping[str, Any]) -> Phase3Observation:
    required_sections = ("technical", "regime", "sentiment", "consensus", "context")
    missing_sections = [key for key in required_sections if key not in payload]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"phase2 payload missing required sections: {missing}")

    technical = _expect_mapping(payload["technical"], name="technical")
    regime = _expect_mapping(payload["regime"], name="regime")
    sentiment = _expect_mapping(payload["sentiment"], name="sentiment")
    consensus = _expect_mapping(payload["consensus"], name="consensus")
    context = _expect_mapping(payload["context"], name="context")
    microstructure = _expect_mapping(payload.get("microstructure", {}), name="microstructure")

    generated_at = _coerce_datetime(context.get("generated_at_utc") or payload.get("generated_at"))
    expires_at = _coerce_datetime(
        context.get("expires_at_utc") or payload.get("expires_at"),
        default=generated_at + timedelta(seconds=60),
    )
    symbol = _coerce_symbol(payload, technical, regime, sentiment, consensus)

    return Phase3Observation(
        snapshot_id=_coerce_str(context.get("snapshot_id") or payload.get("snapshot_id") or "phase3-snapshot"),
        symbol=symbol,
        generated_at=generated_at,
        expires_at=expires_at,
        schema_version="1.0",
        quality_status=_coerce_quality_status(context.get("quality_status") or payload.get("quality_status")),
        source_type=_coerce_source_type(context.get("source_type") or payload.get("source_type")),
        technical=TechnicalRiskSnapshot(
            volatility_estimate=_coerce_float(
                technical.get("volatility_estimate") or technical.get("volatility"),
                default=0.0,
            ),
            var_95=_coerce_float(technical.get("var_95"), default=0.0),
            var_99=_coerce_float(technical.get("var_99"), default=0.0),
            es_95=_coerce_float(technical.get("es_95"), default=0.0),
            es_99=_coerce_float(technical.get("es_99"), default=0.0),
        ),
        regime_probabilities=_extract_regime_probabilities(regime),
        macro=MacroSnapshot(
            macro_differential=_coerce_float(
                context.get("macro_differential") or payload.get("macro_differential"),
                default=0.0,
            ),
            rbi_signal=_coerce_float(context.get("rbi_signal"), default=0.0),
        ),
        sentiment=SentimentSnapshot(
            score=_coerce_float(
                sentiment.get("sentiment_score") or sentiment.get("score"),
                default=0.0,
            ),
            confidence=_coerce_float(sentiment.get("confidence"), default=0.0),
            freshness_state=_coerce_str(
                sentiment.get("freshness_state") or sentiment.get("sentiment_freshness") or "unknown"
            ),
            quality_status=_coerce_quality_status(
                sentiment.get("quality_status") or sentiment.get("quality_flag")
            ),
        ),
        consensus=ConsensusSnapshot(
            score=_coerce_float(consensus.get("score"), default=0.0),
            confidence=_coerce_float(
                consensus.get("confidence") or consensus.get("final_confidence"),
                default=0.0,
            ),
            risk_mode=_coerce_str(consensus.get("risk_mode") or "normal"),
            crisis_probability=_coerce_float(
                consensus.get("crisis_probability") or context.get("crisis_probability"),
                default=0.0,
            ),
        ),
        microstructure=MicrostructureSnapshot(
            feed_is_healthy=bool(microstructure.get("feed_is_healthy", False)),
            order_book_imbalance=_coerce_optional_float(microstructure.get("order_book_imbalance")),
            queue_pressure=_coerce_optional_float(microstructure.get("queue_pressure")),
            quality_flags=_coerce_quality_flags(microstructure.get("quality_flags")),
        ),
    )


def _extract_regime_probabilities(regime: Mapping[str, Any]) -> RegimeProbabilities:
    base = {
        "bull": 0.0,
        "bear": 0.0,
        "sideways": 0.0,
        "crisis": 0.0,
        "rbi_band_transition": 0.0,
        "alien": 0.0,
    }

    probabilities: Mapping[str, Any] | None = None
    if isinstance(regime.get("probabilities"), Mapping):
        probabilities = _expect_mapping(regime.get("probabilities"), name="regime.probabilities")
    else:
        details = regime.get("details")
        if isinstance(details, Mapping):
            pearl = details.get("pearl")
            if isinstance(pearl, Mapping) and isinstance(pearl.get("probabilities"), Mapping):
                probabilities = _expect_mapping(pearl.get("probabilities"), name="regime.details.pearl.probabilities")

    if probabilities is not None:
        for key, value in probabilities.items():
            canonical = _canonical_regime_key(key)
            if canonical is None:
                continue
            base[canonical] = _coerce_float(value, default=0.0)

    total = sum(base.values())
    if total <= 0.0:
        state_key = _canonical_state_key(regime.get("regime_state"))
        if state_key is None:
            state_key = "sideways"
        base[state_key] = 1.0
        total = 1.0

    normalized = {key: float(value / total) for key, value in base.items()}
    return RegimeProbabilities(**normalized)


def _canonical_regime_key(raw_key: Any) -> str | None:
    if raw_key is None:
        return None
    normalized = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
    return _REGIME_KEY_ALIASES.get(normalized)


def _canonical_state_key(raw_state: Any) -> str | None:
    if raw_state is None:
        return None
    normalized = str(raw_state).strip().lower()
    return _REGIME_STATE_TO_KEY.get(normalized)


def _coerce_symbol(payload: Mapping[str, Any], *sections: Mapping[str, Any]) -> str:
    raw = payload.get("symbol")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    for section in sections:
        value = section.get("symbol")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _coerce_datetime(value: Any, *, default: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    if default is not None:
        if default.tzinfo is None:
            return default.replace(tzinfo=UTC)
        return default.astimezone(UTC)
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


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_quality_status(value: Any) -> QualityStatus:
    if isinstance(value, QualityStatus):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for item in QualityStatus:
            if item.value == normalized:
                return item
    return QualityStatus.PASS


def _coerce_source_type(value: Any) -> SourceType:
    if isinstance(value, SourceType):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for item in SourceType:
            if item.value == normalized:
                return item
    return SourceType.INTERNAL_PIPELINE


def _coerce_quality_flags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _expect_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping")
