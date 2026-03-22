from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping

from src.agents.consensus import ConsensusAgent, build_consensus_input
from src.agents.regime.data_loader import RegimeDataLoader
from src.agents.regime.regime_agent import RegimeAgent
from src.agents.regime.schemas import RegimePrediction, RiskLevel
from src.agents.sentiment.schemas import SentimentPrediction
from src.agents.strategic_executive.adapters import build_phase3_observation_from_phase2_payload
from src.agents.technical.schemas import TechnicalPrediction

REGIME_SCORE_MAP = {
    "Bull": 0.85,
    "Bear": -0.85,
    "Sideways": 0.0,
    "Crisis": -0.6,
    "RBI-Band transition": -0.2,
    "Alien": -1.0,
}
PROTECTIVE_REGIME_STATES = {"Bear", "Crisis", "RBI-Band transition", "Alien"}
RISK_LEVEL_TO_FLOAT = {
    RiskLevel.FULL_RISK.value: 0.0,
    RiskLevel.REDUCED_RISK.value: 0.5,
    RiskLevel.NEUTRAL_CASH.value: 1.0,
}
WARN_FRESHNESS_STATES = {"stale", "expired", "miss", "error"}
CONSENSUS_NEUTRAL_BAND = 0.05


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected mapping-compatible value, got {type(value)!r}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def direction_to_score(direction: str, confidence: float) -> float:
    normalized = str(direction).strip().lower()
    magnitude = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    if normalized == "up":
        return magnitude
    if normalized == "down":
        return -magnitude
    return 0.0


def technical_prediction_to_payload(prediction: TechnicalPrediction | Mapping[str, Any]) -> dict[str, Any]:
    payload = _to_mapping(prediction)
    direction = str(payload.get("direction", "neutral")).lower()
    confidence = _clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)
    score = direction_to_score(direction, confidence)
    payload.update(
        {
            "name": "technical",
            "score": score,
            "confidence": confidence,
            "is_protective": direction == "down",
        }
    )
    return payload


def regime_prediction_to_payload(prediction: RegimePrediction | Mapping[str, Any]) -> dict[str, Any]:
    payload = _to_mapping(prediction)
    details = payload.get("details")
    details = dict(details) if isinstance(details, Mapping) else {}
    ood = details.get("ood")
    ood = dict(ood) if isinstance(ood, Mapping) else {}
    state = str(payload.get("regime_state", "Sideways"))
    risk_level = str(payload.get("risk_level", RiskLevel.FULL_RISK.value))
    probabilities = None
    pearl = details.get("pearl")
    if isinstance(pearl, Mapping):
        pearl_probabilities = pearl.get("probabilities")
        if isinstance(pearl_probabilities, Mapping):
            probabilities = dict(pearl_probabilities)
    payload.update(
        {
            "name": "regime",
            "score": REGIME_SCORE_MAP.get(state, 0.0),
            "confidence": _clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            "is_protective": risk_level != RiskLevel.FULL_RISK.value or state in PROTECTIVE_REGIME_STATES,
            "risk_level": risk_level,
            "ood_warning": _safe_bool(ood.get("warning"), False),
            "ood_alien": _safe_bool(ood.get("alien"), False),
        }
    )
    if probabilities is not None:
        payload["probabilities"] = probabilities
    return payload


def sentiment_prediction_to_payload(
    prediction: SentimentPrediction | Mapping[str, Any],
    *,
    z_t: float | None = None,
) -> dict[str, Any]:
    payload = _to_mapping(prediction)
    freshness_state = str(payload.get("freshness_state", payload.get("freshness_flag", "miss"))).lower()
    score = _clamp(_safe_float(payload.get("score", payload.get("sentiment_score", 0.0)), 0.0), -1.0, 1.0)
    confidence = _clamp(_safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)
    source_count = 0 if freshness_state in {"miss", "expired", "error"} else 1
    payload.update(
        {
            "name": "sentiment",
            "score": score,
            "sentiment_score": score,
            "confidence": confidence,
            "freshness_flag": freshness_state,
            "source_count": source_count,
            "is_protective": score < 0.0 or _safe_bool(payload.get("reduced_risk_mode"), False),
            "z_t": score if z_t is None else _clamp(_safe_float(z_t, score), -1.0, 1.0),
        }
    )
    return payload


def build_phase2_context(
    *,
    snapshot_id: str,
    generated_at_utc: datetime,
    technical_payload: Mapping[str, Any] | None,
    regime_payload: Mapping[str, Any] | None,
    sentiment_payload: Mapping[str, Any] | None,
    context_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context_features = dict(context_features or {})
    volatility = max(
        0.0,
        _safe_float(
            context_features.get("volatility"),
            _safe_float((technical_payload or {}).get("volatility_estimate"), 0.0),
        ),
    )
    risk_level = str((regime_payload or {}).get("risk_level", RiskLevel.FULL_RISK.value))
    crisis_probability = _clamp(
        0.5 * min(1.0, volatility / 0.03) + 0.5 * RISK_LEVEL_TO_FLOAT.get(risk_level, 0.0),
        0.0,
        1.0,
    )
    sentiment_score = _safe_float((sentiment_payload or {}).get("score"), 0.0)
    freshness_flag = str((sentiment_payload or {}).get("freshness_flag", "miss")).lower()
    regime_warning = _safe_bool((regime_payload or {}).get("ood_warning"), False)
    regime_alien = _safe_bool((regime_payload or {}).get("ood_alien"), False)
    quality_status = "warn" if freshness_flag in WARN_FRESHNESS_STATES or regime_warning or regime_alien else "pass"
    context = {
        "snapshot_id": snapshot_id,
        "generated_at_utc": generated_at_utc.isoformat(),
        "expires_at_utc": (generated_at_utc + timedelta(seconds=60)).isoformat(),
        "quality_status": quality_status,
        "source_type": "internal_pipeline",
        "volatility": volatility,
        "macro_differential": _safe_float(
            context_features.get("macro_differential"),
            _safe_float(context_features.get("macro_regime_shock"), 0.0),
        ),
        "rbi_signal": _safe_float(context_features.get("rbi_signal"), _safe_float(context_features.get("macro_directional_flag"), 0.0)),
        "sentiment_quantile": _clamp((sentiment_score + 1.0) / 2.0, 0.0, 1.0),
        "crisis_probability": crisis_probability,
        "sentiment_freshness": freshness_flag,
        "regime_ood_warning": regime_warning,
        "regime_ood_alien": regime_alien,
    }
    feature_timestamp = context_features.get("feature_timestamp_utc")
    if feature_timestamp is not None:
        context["feature_timestamp_utc"] = feature_timestamp
    return context


def build_phase2_payload(
    *,
    symbol: str,
    snapshot_id: str,
    generated_at_utc: datetime,
    technical_payload: Mapping[str, Any] | None,
    regime_payload: Mapping[str, Any] | None,
    sentiment_payload: Mapping[str, Any] | None,
    context_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "snapshot_id": snapshot_id,
        "technical": None if technical_payload is None else dict(technical_payload),
        "regime": None if regime_payload is None else dict(regime_payload),
        "sentiment": None if sentiment_payload is None else dict(sentiment_payload),
        "context": build_phase2_context(
            snapshot_id=snapshot_id,
            generated_at_utc=generated_at_utc,
            technical_payload=technical_payload,
            regime_payload=regime_payload,
            sentiment_payload=sentiment_payload,
            context_features=context_features,
        ),
    }
    return payload


def consensus_output_to_payload(
    consensus_output: Any,
    *,
    crisis_probability: float,
    divergence_warn_threshold: float,
) -> dict[str, Any]:
    payload = _to_mapping(consensus_output)
    score = _safe_float(payload.get("score"), 0.0)
    final_direction = "neutral"
    if score > CONSENSUS_NEUTRAL_BAND:
        final_direction = "buy"
    elif score < -CONSENSUS_NEUTRAL_BAND:
        final_direction = "sell"
    weights = payload.get("weights")
    weights = dict(weights) if isinstance(weights, Mapping) else {}
    payload.update(
        {
            "final_direction": final_direction,
            "final_confidence": _safe_float(payload.get("confidence"), 0.0),
            "technical_weight": _safe_float(weights.get("technical"), 0.0),
            "regime_weight": _safe_float(weights.get("regime"), 0.0),
            "sentiment_weight": _safe_float(weights.get("sentiment"), 0.0),
            "crisis_probability": _clamp(_safe_float(crisis_probability, 0.0), 0.0, 1.0),
            "crisis_mode": _safe_float(crisis_probability, 0.0) >= 0.5,
            "agent_divergence": _safe_float(payload.get("divergence_score"), 0.0) >= divergence_warn_threshold,
        }
    )
    return payload


def persist_consensus_signal(
    *,
    consensus_agent: ConsensusAgent,
    symbol: str,
    snapshot_id: str,
    generated_at_utc: datetime,
    consensus_payload: Mapping[str, Any],
    model_id: str = "consensus_weighted_v1",
) -> None:
    if consensus_agent.phase2_recorder is None:
        return
    consensus_agent.phase2_recorder.save_consensus_signal(
        {
            "symbol": symbol,
            "timestamp": generated_at_utc,
            "final_direction": str(consensus_payload.get("final_direction", "neutral")),
            "final_confidence": _safe_float(consensus_payload.get("final_confidence"), 0.0),
            "technical_weight": _safe_float(consensus_payload.get("technical_weight"), 0.0),
            "regime_weight": _safe_float(consensus_payload.get("regime_weight"), 0.0),
            "sentiment_weight": _safe_float(consensus_payload.get("sentiment_weight"), 0.0),
            "crisis_mode": _safe_bool(consensus_payload.get("crisis_mode"), False),
            "agent_divergence": _safe_bool(consensus_payload.get("agent_divergence"), False),
            "transition_model": str(consensus_payload.get("transition_model", "lstar")),
            "model_id": model_id,
            "schema_version": str(consensus_payload.get("schema_version", "1.0")),
        },
        data_snapshot_id=snapshot_id,
    )


class Phase2AnalystBoardRunner:
    def __init__(
        self,
        *,
        technical_agent: Any | None = None,
        regime_agent: Any | None = None,
        sentiment_agent: Any | None = None,
        consensus_agent: ConsensusAgent | None = None,
        context_provider: Callable[[str, int], Mapping[str, Any]] | None = None,
    ) -> None:
        self.technical_agent = technical_agent
        self.regime_agent = regime_agent
        self.sentiment_agent = sentiment_agent
        self.consensus_agent = consensus_agent
        self.context_provider = context_provider or self._default_context_provider

    def refresh_sentiment_cache(self, *, as_of_utc: datetime, lookback_hours: int) -> Any | None:
        if self.sentiment_agent is None:
            return None
        return self.sentiment_agent.run_nightly_batch(as_of_utc=as_of_utc, lookback_hours=lookback_hours)

    def run_symbol(
        self,
        *,
        symbol: str,
        snapshot_id: str,
        technical_limit: int,
        regime_limit: int,
        as_of_utc: datetime | None = None,
        emit_phase3_observation: bool = False,
    ) -> dict[str, Any]:
        generated_at = (as_of_utc or datetime.now(UTC)).astimezone(UTC)
        result: dict[str, Any] = {
            "symbol": symbol,
            "technical": None,
            "regime": None,
            "sentiment": None,
            "consensus": None,
            "context": None,
            "phase3_observation": None,
            "skipped_agents": [],
        }

        if self.technical_agent is None:
            result["skipped_agents"].append("technical")
        else:
            technical_prediction = self.technical_agent.predict(
                symbol,
                limit=technical_limit,
                data_snapshot_id=snapshot_id,
            )
            if technical_prediction is None:
                raise RuntimeError(f"technical agent returned no prediction for {symbol}")
            result["technical"] = technical_prediction_to_payload(technical_prediction)

        if self.regime_agent is None:
            result["skipped_agents"].append("regime")
        else:
            regime_prediction = self.regime_agent.detect_regime(
                symbol,
                limit=regime_limit,
                data_snapshot_id=snapshot_id,
            )
            result["regime"] = regime_prediction_to_payload(regime_prediction)

        if self.sentiment_agent is None:
            result["skipped_agents"].append("sentiment")
        else:
            sentiment_prediction = self.sentiment_agent.get_cached_sentiment(symbol, as_of_utc=generated_at)
            sentiment_z_t = self.sentiment_agent.get_z_t(symbol, as_of_utc=generated_at)
            result["sentiment"] = sentiment_prediction_to_payload(sentiment_prediction, z_t=sentiment_z_t)

        if result["technical"] and result["regime"] and result["sentiment"]:
            context_features = dict(self.context_provider(symbol, regime_limit))
            result["context"] = build_phase2_context(
                snapshot_id=snapshot_id,
                generated_at_utc=generated_at,
                technical_payload=result["technical"],
                regime_payload=result["regime"],
                sentiment_payload=result["sentiment"],
                context_features=context_features,
            )
        else:
            if self.consensus_agent is not None:
                result["skipped_agents"].append("consensus")
            return result

        if self.consensus_agent is None:
            result["skipped_agents"].append("consensus")
            return result

        consensus_input = build_consensus_input(
            technical=result["technical"],
            regime=result["regime"],
            sentiment=result["sentiment"],
            context=result["context"],
        )
        consensus_output = self.consensus_agent.run(consensus_input)
        result["consensus"] = consensus_output_to_payload(
            consensus_output,
            crisis_probability=_safe_float(result["context"]["crisis_probability"], 0.0),
            divergence_warn_threshold=self.consensus_agent.divergence_warn_threshold,
        )
        persist_consensus_signal(
            consensus_agent=self.consensus_agent,
            symbol=symbol,
            snapshot_id=snapshot_id,
            generated_at_utc=generated_at,
            consensus_payload=result["consensus"],
        )

        full_payload = {
            "symbol": symbol,
            "snapshot_id": snapshot_id,
            "technical": result["technical"],
            "regime": result["regime"],
            "sentiment": result["sentiment"],
            "consensus": result["consensus"],
            "context": result["context"],
        }
        if emit_phase3_observation:
            observation = build_phase3_observation_from_phase2_payload(full_payload)
            result["phase3_observation"] = observation.model_dump(mode="json")
        return result

    def _default_context_provider(self, symbol: str, limit: int) -> Mapping[str, Any]:
        loader = RegimeDataLoader()
        if self.regime_agent is not None and hasattr(self.regime_agent, "loader"):
            loader = self.regime_agent.loader
        raw = loader.load_features(symbol=symbol, limit=limit)
        if raw.empty:
            return {}
        prepared = RegimeAgent._prepare_features(raw)
        if prepared.empty:
            return {}
        latest = prepared.iloc[-1]
        feature_timestamp = latest.get("timestamp")
        if hasattr(feature_timestamp, "isoformat"):
            feature_timestamp = feature_timestamp.isoformat()
        return {
            "feature_timestamp_utc": feature_timestamp,
            "volatility": _safe_float(latest.get("rolling_vol_20"), 0.0),
            "macro_differential": _safe_float(latest.get("macro_regime_shock"), _safe_float(latest.get("macro_regime_index"), 0.0)),
            "rbi_signal": _safe_float(latest.get("macro_directional_flag"), 0.0),
        }
