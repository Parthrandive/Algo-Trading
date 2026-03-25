import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, or_, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.consensus import ConsensusAgent, build_consensus_input
from src.agents.regime.regime_agent import RegimeAgent
from src.agents.regime.schemas import RegimeState, RiskLevel
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.agents.sentiment.schemas import SentimentLane
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.technical_agent import TechnicalAgent
from src.db.connection import get_engine, get_session
from src.db.models import TextItemDB
from src.db.phase2_recorder import Phase2Recorder
from config.symbols import SymbolValidationResult, dedupe_symbols, discover_training_symbols, is_forex, validate_equity_symbol


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run available Phase 2 agents and persist their predictions.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Equity symbols to score, e.g. RELIANCE.NS POWERGRID.NS LT.NS",
    )
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Optional DATABASE_URL override.")
    parser.add_argument("--interval", default="1h", help="Interval used for runtime symbol discovery.")
    parser.add_argument("--models-dir", default="data/models", help="Directory containing trained technical model artifacts.")
    parser.add_argument("--technical-limit", type=int, default=300, help="History bars used by the technical agent loader.")
    parser.add_argument("--regime-limit", type=int, default=800, help="Gold-feature rows used by the regime agent.")
    parser.add_argument("--skip-technical", action="store_true", help="Skip technical predictions.")
    parser.add_argument("--skip-regime", action="store_true", help="Skip regime predictions.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment predictions.")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus predictions.")
    parser.add_argument("--sentiment-lookback-hours", type=int, default=24, help="Lookback window (hours) for text items used by sentiment scoring.")
    parser.add_argument("--sentiment-max-records", type=int, default=200, help="Maximum text records to score per symbol.")
    parser.add_argument(
        "--sentiment-lane",
        choices=[SentimentLane.FAST.value, SentimentLane.SLOW.value],
        default=SentimentLane.FAST.value,
        help="Sentiment lane for inference.",
    )
    parser.add_argument(
        "--consensus-neutral-band",
        type=float,
        default=0.05,
        help="Absolute score band used to map consensus score to BUY/SELL/NEUTRAL.",
    )
    parser.add_argument("--disable-persist", action="store_true", help="Do not write predictions to the DB.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any symbol fails.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to data/reports/phase2_predictions_<timestamp>.json",
    )
    return parser.parse_args()


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("data/reports") / f"phase2_predictions_{timestamp}.json"


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coerce_datetime_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_recent_text_items(
    *,
    session_factory,
    symbol: str,
    end_utc: datetime,
    lookback_hours: int,
    max_records: int,
) -> list[TextItemDB]:
    start_utc = end_utc - timedelta(hours=max(1, int(lookback_hours)))
    quality_filter = ("pass", "warn")

    with session_factory() as session:
        symbol_stmt = (
            select(TextItemDB)
            .where(
                and_(
                    TextItemDB.timestamp >= start_utc,
                    TextItemDB.timestamp <= end_utc,
                    TextItemDB.quality_status.in_(quality_filter),
                    TextItemDB.symbol == symbol,
                )
            )
            .order_by(TextItemDB.timestamp.desc())
            .limit(max(1, int(max_records)))
        )
        symbol_rows = list(session.execute(symbol_stmt).scalars().all())

        remaining = max(0, int(max_records) - len(symbol_rows))
        if remaining == 0:
            return symbol_rows

        global_stmt = (
            select(TextItemDB)
            .where(
                and_(
                    TextItemDB.timestamp >= start_utc,
                    TextItemDB.timestamp <= end_utc,
                    TextItemDB.quality_status.in_(quality_filter),
                    or_(TextItemDB.symbol.is_(None), TextItemDB.symbol == ""),
                )
            )
            .order_by(TextItemDB.timestamp.desc())
            .limit(remaining)
        )
        global_rows = list(session.execute(global_stmt).scalars().all())
        return symbol_rows + global_rows


def _sentiment_class_from_score(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def _technical_signal(payload: dict[str, Any]) -> dict[str, Any]:
    direction = str(payload.get("direction", "neutral")).strip().lower()
    confidence = _clamp(float(payload.get("confidence", 0.5) or 0.5), 0.0, 1.0)
    sign_map = {"up": 1.0, "buy": 1.0, "down": -1.0, "sell": -1.0}
    sign = sign_map.get(direction, 0.0)
    score = 0.0 if sign == 0.0 else sign * max(0.15, confidence)
    return {
        "name": "technical",
        "score": _clamp(float(score), -1.0, 1.0),
        "confidence": confidence,
        "is_protective": sign < 0.0,
    }


def _regime_signal(payload: dict[str, Any]) -> dict[str, Any]:
    regime_state = str(payload.get("regime_state", RegimeState.SIDEWAYS.value))
    score_map = {
        RegimeState.BULL.value: 0.70,
        RegimeState.SIDEWAYS.value: 0.0,
        RegimeState.BEAR.value: -0.70,
        RegimeState.RBI_BAND_TRANSITION.value: -0.20,
        RegimeState.CRISIS.value: -0.90,
        RegimeState.ALIEN.value: -1.00,
    }
    score = score_map.get(regime_state, 0.0)
    confidence = _clamp(float(payload.get("confidence", 0.5) or 0.5), 0.0, 1.0)
    risk_level = str(payload.get("risk_level", RiskLevel.REDUCED_RISK.value))
    is_protective = risk_level in {RiskLevel.REDUCED_RISK.value, RiskLevel.NEUTRAL_CASH.value} or score < 0.0
    return {
        "name": "regime",
        "score": _clamp(float(score), -1.0, 1.0),
        "confidence": confidence,
        "is_protective": is_protective,
    }


def _sentiment_signal(payload: dict[str, Any]) -> dict[str, Any]:
    score = _clamp(float(payload.get("sentiment_score", 0.0) or 0.0), -1.0, 1.0)
    confidence = _clamp(float(payload.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    sentiment_class = str(payload.get("sentiment_class", "neutral")).lower()
    return {
        "name": "sentiment",
        "score": score,
        "confidence": confidence,
        "is_protective": sentiment_class == "negative",
    }


def _build_consensus_context(
    *,
    technical_payload: dict[str, Any],
    regime_payload: dict[str, Any],
    sentiment_payload: dict[str, Any],
    generated_at_utc: datetime,
) -> dict[str, Any]:
    volatility_raw = abs(float(technical_payload.get("volatility_estimate", 0.0) or 0.0))
    volatility = _clamp(volatility_raw * 25.0, 0.0, 1.0)

    details = regime_payload.get("details") if isinstance(regime_payload.get("details"), dict) else {}
    macro_features = details.get("macro_features") if isinstance(details, dict) else {}
    if not isinstance(macro_features, dict):
        macro_features = {}

    macro_index = float(macro_features.get("macro_regime_index", 0.0) or 0.0)
    macro_differential = _clamp(macro_index / 3.0, -1.0, 1.0)
    rbi_signal = _clamp(float(macro_features.get("macro_directional_flag", 0.0) or 0.0), -1.0, 1.0)

    transition_probability = _clamp(float(regime_payload.get("transition_probability", 0.0) or 0.0), 0.0, 1.0)
    regime_state = str(regime_payload.get("regime_state", ""))
    risk_level = str(regime_payload.get("risk_level", ""))
    crisis_probability = transition_probability
    if regime_state in {RegimeState.CRISIS.value, RegimeState.ALIEN.value}:
        crisis_probability = max(crisis_probability, 0.85)
    elif risk_level == RiskLevel.REDUCED_RISK.value:
        crisis_probability = max(crisis_probability, 0.45)

    sentiment_score = _clamp(float(sentiment_payload.get("sentiment_score", 0.0) or 0.0), -1.0, 1.0)
    sentiment_quantile = _clamp((sentiment_score + 1.0) / 2.0, 0.0, 1.0)

    return {
        "volatility": volatility,
        "macro_differential": macro_differential,
        "rbi_signal": rbi_signal,
        "sentiment_quantile": sentiment_quantile,
        "crisis_probability": _clamp(crisis_probability, 0.0, 1.0),
        "generated_at_utc": generated_at_utc,
    }


def _score_to_direction(score: float, neutral_band: float, risk_mode: str) -> str:
    if risk_mode == "protective":
        return "NEUTRAL"
    if score > neutral_band:
        return "BUY"
    if score < -neutral_band:
        return "SELL"
    return "NEUTRAL"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else _default_output_path()
    _ensure_output_dir(output_path)

    effective_db_url = args.db_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(effective_db_url)
    engine = get_engine(effective_db_url)
    session_factory = get_session(engine)

    def validate_symbol(symbol: str):
        try:
            frame = loader.load_historical_bars(
                symbol,
                limit=max(args.technical_limit, args.regime_limit),
                use_nse_fallback=False,
                min_fallback_rows=100,
                interval=args.interval,
            )
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        return validate_equity_symbol(symbol=symbol, frame=frame, interval=args.interval)

    discovery = discover_training_symbols(
        interval=args.interval,
        requested_symbols=dedupe_symbols(args.symbols or []) or None,
        database_url=args.db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    symbols_to_score = list(discovery.active_symbols)
    if not symbols_to_score:
        raise SystemExit("No active equity symbols available for scoring.")

    technical_agent = None
    regime_agent = None
    sentiment_agent = None
    consensus_agent = None
    recorder = None

    if not args.skip_technical:
        technical_agent = TechnicalAgent(
            db_url=effective_db_url,
            models_dir=args.models_dir,
            persist_predictions=not args.disable_persist,
        )
    if not args.skip_regime:
        regime_agent = RegimeAgent(
            database_url=effective_db_url,
            persist_predictions=not args.disable_persist,
        )
    if not args.skip_sentiment:
        sentiment_agent = SentimentAgent.from_default_components()
    if not args.skip_consensus:
        consensus_agent = ConsensusAgent.from_default_components()
    if not args.disable_persist and (sentiment_agent is not None or consensus_agent is not None):
        recorder = Phase2Recorder(effective_db_url)

    snapshot_id = f"phase2_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []
    failures: list[dict] = []

    for symbol in symbols_to_score:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
        symbol_result = {
            "symbol": symbol,
            "technical": None,
            "regime": None,
            "sentiment": None,
            "consensus": None,
            "skipped_agents": [],
        }

        if technical_agent is None:
            symbol_result["skipped_agents"].append("technical")
        else:
            try:
                pred = technical_agent.predict(symbol, limit=args.technical_limit, data_snapshot_id=snapshot_id)
                symbol_result["technical"] = None if pred is None else pred.model_dump(mode="json")
            except Exception as exc:
                logger.exception("Technical prediction failed for %s: %s", symbol, exc)
                failures.append({"symbol": symbol, "agent": "technical", "error": str(exc)})

        if regime_agent is None:
            symbol_result["skipped_agents"].append("regime")
        else:
            try:
                pred = regime_agent.detect_regime(symbol, limit=args.regime_limit, data_snapshot_id=snapshot_id)
                symbol_result["regime"] = pred.model_dump(mode="json")
            except Exception as exc:
                logger.exception("Regime prediction failed for %s: %s", symbol, exc)
                failures.append({"symbol": symbol, "agent": "regime", "error": str(exc)})

        if sentiment_agent is None:
            symbol_result["skipped_agents"].append("sentiment")
        else:
            try:
                scoring_time = datetime.now(timezone.utc)
                lane = SentimentLane(args.sentiment_lane)
                text_rows = _load_recent_text_items(
                    session_factory=session_factory,
                    symbol=symbol,
                    end_utc=scoring_time,
                    lookback_hours=max(1, int(args.sentiment_lookback_hours)),
                    max_records=max(1, int(args.sentiment_max_records)),
                )

                predictions = []
                for row in text_rows:
                    payload_row = {
                        "source_id": str(row.source_id),
                        "headline": str(row.headline or ""),
                        "content": str(row.content or ""),
                        "normalized_content": str(row.content or ""),
                        "language": str(row.language or "en"),
                    }
                    as_of_utc = _coerce_datetime_utc(row.timestamp)
                    predictions.append(
                        sentiment_agent.score_textual_payload(
                            payload_row,
                            lane=lane,
                            as_of_utc=as_of_utc,
                        )
                    )

                aggregate = sentiment_agent.compute_daily_z_t(predictions, as_of_utc=scoring_time)
                sentiment_payload = {
                    "symbol": symbol,
                    "timestamp": aggregate.generated_at_utc.isoformat(),
                    "lane": lane.value,
                    "sentiment_class": _sentiment_class_from_score(aggregate.weighted_sentiment_score),
                    "sentiment_score": float(aggregate.weighted_sentiment_score),
                    "z_t": float(aggregate.z_t),
                    "confidence": float(aggregate.sentiment_confidence),
                    "source_count": int(len(predictions)),
                    "model_id": str(predictions[0].model_name if predictions else f"sentiment_{lane.value}_v1.0"),
                    "quality_status": aggregate.quality_status.value,
                    "schema_version": "1.0",
                }
                if recorder is not None:
                    recorder.save_sentiment_score(sentiment_payload, data_snapshot_id=snapshot_id)
                symbol_result["sentiment"] = sentiment_payload
            except Exception as exc:
                logger.exception("Sentiment prediction failed for %s: %s", symbol, exc)
                failures.append({"symbol": symbol, "agent": "sentiment", "error": str(exc)})

        if consensus_agent is None:
            symbol_result["skipped_agents"].append("consensus")
        else:
            try:
                if symbol_result["technical"] is None or symbol_result["regime"] is None:
                    symbol_result["skipped_agents"].append("consensus")
                    symbol_result["consensus_unavailable_reason"] = "requires technical and regime predictions"
                else:
                    sentiment_payload = symbol_result.get("sentiment") or {
                        "sentiment_class": "neutral",
                        "sentiment_score": 0.0,
                        "confidence": 0.0,
                    }
                    consensus_time = datetime.now(timezone.utc)
                    context = _build_consensus_context(
                        technical_payload=symbol_result["technical"],
                        regime_payload=symbol_result["regime"],
                        sentiment_payload=sentiment_payload,
                        generated_at_utc=consensus_time,
                    )
                    consensus_input = build_consensus_input(
                        technical=_technical_signal(symbol_result["technical"]),
                        regime=_regime_signal(symbol_result["regime"]),
                        sentiment=_sentiment_signal(sentiment_payload),
                        context=context,
                    )
                    consensus_output = consensus_agent.run(consensus_input)
                    direction = _score_to_direction(
                        score=float(consensus_output.score),
                        neutral_band=max(0.0, float(args.consensus_neutral_band)),
                        risk_mode=str(consensus_output.risk_mode.value),
                    )
                    consensus_payload = {
                        "symbol": symbol,
                        "timestamp": consensus_output.generated_at_utc.isoformat(),
                        "final_direction": direction,
                        "final_confidence": float(consensus_output.confidence),
                        "technical_weight": float(consensus_output.weights.get("technical", 0.0)),
                        "regime_weight": float(consensus_output.weights.get("regime", 0.0)),
                        "sentiment_weight": float(consensus_output.weights.get("sentiment", 0.0)),
                        "crisis_mode": bool(consensus_output.risk_mode.value == "protective" or consensus_output.crisis_weight >= 0.5),
                        "agent_divergence": bool(consensus_output.divergence_score >= consensus_agent.divergence_warn_threshold),
                        "transition_model": str(consensus_output.transition_model.value),
                        "model_id": "consensus_lstar_estar_v1.0",
                        "schema_version": consensus_output.schema_version,
                        "score": float(consensus_output.score),
                        "risk_mode": str(consensus_output.risk_mode.value),
                        "transition_score": float(consensus_output.transition_score),
                        "divergence_score": float(consensus_output.divergence_score),
                        "crisis_weight": float(consensus_output.crisis_weight),
                    }
                    if recorder is not None:
                        recorder.save_consensus_signal(consensus_payload, data_snapshot_id=snapshot_id)
                    symbol_result["consensus"] = consensus_payload
            except Exception as exc:
                logger.exception("Consensus prediction failed for %s: %s", symbol, exc)
                failures.append({"symbol": symbol, "agent": "consensus", "error": str(exc)})

        symbol_result["not_implemented_agents"] = []
        results.append(symbol_result)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "db_url": effective_db_url,
        "persist_enabled": not args.disable_persist,
        "symbols": symbols_to_score,
        "results": results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote Phase 2 prediction report to %s", output_path)

    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
