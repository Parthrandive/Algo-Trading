from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agents.consensus import ConsensusAgent, build_consensus_input
from src.agents.phase2_orchestrator import (
    build_phase2_context,
    consensus_output_to_payload,
    regime_prediction_to_payload,
    sentiment_prediction_to_payload,
    technical_prediction_to_payload,
)
from src.db.connection import get_engine, get_session
from src.db.models import RegimePredictionDB, SentimentScoreDB, TechnicalPredictionDB
from src.db.phase2_recorder import Phase2Recorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/reports")

WARN_FRESHNESS_STATES = {"stale", "expired", "missing", "miss", "error"}


@dataclass(frozen=True)
class SignalRows:
    technical: TechnicalPredictionDB
    regime: RegimePredictionDB
    sentiment: SentimentScoreDB | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live consensus from latest Phase-2 DB predictions.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. If omitted, symbols are auto-discovered from latest technical/regime predictions.",
    )
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Optional DATABASE_URL override.")
    parser.add_argument("--sentiment-lane", default="slow", help="Sentiment lane to read (default: slow).")
    parser.add_argument("--model-id", default="consensus_weighted_v1", help="Model id stored in consensus_signals.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to data/reports/live_consensus_<timestamp>.json",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any symbol fails.")
    return parser.parse_args()


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"live_consensus_{timestamp}.json"


def _parse_symbols(raw_symbols: str | None) -> list[str]:
    if not raw_symbols:
        return []
    values = [token.strip() for token in str(raw_symbols).split(",") if token.strip()]
    # Preserve order while deduping.
    return list(dict.fromkeys(values))


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_symbols(session: Session, requested: list[str]) -> list[str]:
    if requested:
        return requested

    technical_symbols = {
        str(symbol)
        for symbol in session.execute(select(TechnicalPredictionDB.symbol).distinct()).scalars().all()
        if symbol
    }
    regime_symbols = {
        str(symbol)
        for symbol in session.execute(select(RegimePredictionDB.symbol).distinct()).scalars().all()
        if symbol
    }
    resolved = sorted(technical_symbols & regime_symbols)
    if not resolved:
        raise RuntimeError(
            "No symbols found with both technical and regime predictions. "
            "Run the Phase-2 agents first."
        )
    return resolved


def _fetch_latest_rows(
    session: Session,
    *,
    symbol: str,
    sentiment_lane: str,
) -> SignalRows:
    technical = (
        session.execute(
            select(TechnicalPredictionDB)
            .where(TechnicalPredictionDB.symbol == symbol)
            .order_by(TechnicalPredictionDB.timestamp.desc(), TechnicalPredictionDB.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if technical is None:
        raise RuntimeError(f"{symbol}: no technical prediction rows found.")

    regime = (
        session.execute(
            select(RegimePredictionDB)
            .where(RegimePredictionDB.symbol == symbol)
            .order_by(RegimePredictionDB.timestamp.desc(), RegimePredictionDB.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if regime is None:
        raise RuntimeError(f"{symbol}: no regime prediction rows found.")

    sentiment = (
        session.execute(
            select(SentimentScoreDB)
            .where(SentimentScoreDB.symbol == symbol, SentimentScoreDB.lane == sentiment_lane)
            .order_by(SentimentScoreDB.timestamp.desc(), SentimentScoreDB.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if sentiment is None:
        # Fallback to market-level sentiment row when symbol-level lane data is absent.
        sentiment = (
            session.execute(
                select(SentimentScoreDB)
                .where(SentimentScoreDB.symbol.is_(None), SentimentScoreDB.lane == sentiment_lane)
                .order_by(SentimentScoreDB.timestamp.desc(), SentimentScoreDB.id.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )

    return SignalRows(technical=technical, regime=regime, sentiment=sentiment)


def _sentiment_payload_from_row(symbol: str, row: SentimentScoreDB | None, generated_at_utc: datetime) -> dict[str, Any]:
    if row is None:
        return sentiment_prediction_to_payload(
            {
                "symbol": symbol,
                "timestamp": generated_at_utc,
                "score": 0.0,
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "freshness_flag": "missing",
                "source_count": 0,
                "reduced_risk_mode": True,
            },
            z_t=0.0,
        )

    freshness_flag = str(row.freshness_flag or "missing").lower()
    reduced_mode = freshness_flag in WARN_FRESHNESS_STATES
    source_count = int(row.source_count or 0)
    return sentiment_prediction_to_payload(
        {
            "symbol": symbol,
            "timestamp": row.timestamp,
            "score": row.sentiment_score,
            "sentiment_score": row.sentiment_score,
            "confidence": row.confidence,
            "freshness_flag": freshness_flag,
            "source_count": source_count,
            "reduced_risk_mode": reduced_mode,
        },
        z_t=row.z_t,
    )


def _build_phase2_inputs(symbol: str, rows: SignalRows, generated_at_utc: datetime, snapshot_id: str) -> dict[str, Any]:
    regime_details = _safe_json_loads(rows.regime.details_json)
    technical_payload = technical_prediction_to_payload(
        {
            "symbol": symbol,
            "timestamp": rows.technical.timestamp,
            "direction": rows.technical.direction,
            "confidence": rows.technical.confidence,
            "volatility_estimate": rows.technical.volatility_estimate,
        }
    )
    regime_payload = regime_prediction_to_payload(
        {
            "symbol": symbol,
            "timestamp": rows.regime.timestamp,
            "regime_state": rows.regime.regime_state,
            "confidence": rows.regime.confidence,
            "risk_level": rows.regime.risk_level,
            "details": regime_details,
        }
    )
    sentiment_payload = _sentiment_payload_from_row(symbol, rows.sentiment, generated_at_utc)

    context_features = {
        "volatility": _safe_float(rows.technical.volatility_estimate, 0.0),
        "macro_differential": _safe_float(
            regime_details.get("macro_regime_shock", regime_details.get("macro_regime_index")),
            0.0,
        ),
        "rbi_signal": _safe_float(regime_details.get("macro_directional_flag"), 0.0),
    }
    context_payload = build_phase2_context(
        snapshot_id=snapshot_id,
        generated_at_utc=generated_at_utc,
        technical_payload=technical_payload,
        regime_payload=regime_payload,
        sentiment_payload=sentiment_payload,
        context_features=context_features,
    )
    return {
        "technical": technical_payload,
        "regime": regime_payload,
        "sentiment": sentiment_payload,
        "context": context_payload,
    }


def _persist_consensus_signal(
    recorder: Phase2Recorder,
    *,
    symbol: str,
    generated_at_utc: datetime,
    model_id: str,
    snapshot_id: str,
    latency_ms: float,
    consensus_payload: dict[str, Any],
) -> None:
    recorder.save_consensus_signal(
        {
            "symbol": symbol,
            "timestamp": generated_at_utc,
            "final_direction": str(consensus_payload.get("final_direction", "neutral")),
            "final_confidence": _safe_float(consensus_payload.get("final_confidence"), 0.0),
            "technical_weight": _safe_float(consensus_payload.get("technical_weight"), 0.0),
            "regime_weight": _safe_float(consensus_payload.get("regime_weight"), 0.0),
            "sentiment_weight": _safe_float(consensus_payload.get("sentiment_weight"), 0.0),
            "crisis_mode": bool(consensus_payload.get("crisis_mode", False)),
            "agent_divergence": bool(consensus_payload.get("agent_divergence", False)),
            "transition_model": str(consensus_payload.get("transition_model", "lstar")),
            "model_id": model_id,
            "schema_version": str(consensus_payload.get("schema_version", "1.0")),
        },
        latency_ms=latency_ms,
        data_snapshot_id=snapshot_id,
    )


def run_live_consensus(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_symbols = _parse_symbols(args.symbols)
    generated_at_utc = datetime.now(UTC)
    snapshot_id = f"live_consensus_{generated_at_utc.strftime('%Y%m%d_%H%M%S')}"

    recorder = Phase2Recorder(database_url=args.db_url)
    engine = get_engine(args.db_url)
    SessionLocal = get_session(engine)

    consensus_agent = ConsensusAgent.from_default_components(phase2_recorder=recorder)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with SessionLocal() as session:
        symbols = _resolve_symbols(session, requested_symbols)
        logger.info("Running live consensus for symbols: %s", symbols)

        for symbol in symbols:
            try:
                rows = _fetch_latest_rows(session, symbol=symbol, sentiment_lane=args.sentiment_lane)
                phase2_payload = _build_phase2_inputs(
                    symbol=symbol,
                    rows=rows,
                    generated_at_utc=generated_at_utc,
                    snapshot_id=snapshot_id,
                )
                consensus_input = build_consensus_input(
                    technical=phase2_payload["technical"],
                    regime=phase2_payload["regime"],
                    sentiment=phase2_payload["sentiment"],
                    context=phase2_payload["context"],
                )

                started = perf_counter()
                consensus_output = consensus_agent.run(consensus_input)
                latency_ms = (perf_counter() - started) * 1000.0

                consensus_payload = consensus_output_to_payload(
                    consensus_output,
                    crisis_probability=_safe_float(phase2_payload["context"].get("crisis_probability"), 0.0),
                    divergence_warn_threshold=consensus_agent.divergence_warn_threshold,
                )
                _persist_consensus_signal(
                    recorder,
                    symbol=symbol,
                    generated_at_utc=generated_at_utc,
                    model_id=args.model_id,
                    snapshot_id=snapshot_id,
                    latency_ms=latency_ms,
                    consensus_payload=consensus_payload,
                )

                sentiment_ts = rows.sentiment.timestamp.isoformat() if rows.sentiment is not None else None
                results.append(
                    {
                        "symbol": symbol,
                        "technical_timestamp": rows.technical.timestamp.isoformat(),
                        "regime_timestamp": rows.regime.timestamp.isoformat(),
                        "sentiment_timestamp": sentiment_ts,
                        "consensus": consensus_payload,
                        "latency_ms": round(float(latency_ms), 4),
                    }
                )
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
                logger.exception("Consensus run failed for %s: %s", symbol, exc)

    payload = {
        "generated_at_utc": generated_at_utc.isoformat(),
        "snapshot_id": snapshot_id,
        "symbols_requested": requested_symbols,
        "sentiment_lane": args.sentiment_lane,
        "model_id": args.model_id,
        "results": results,
        "failures": failures,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote live consensus report to %s", output_path)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_live_consensus(args)
    if payload["failures"] and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
