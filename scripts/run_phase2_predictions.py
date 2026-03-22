import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import (
    SymbolValidationResult,
    dedupe_symbols,
    discover_training_symbols,
    validate_equity_symbol,
)
from src.agents.consensus import ConsensusAgent
from src.agents.phase2_orchestrator import Phase2AnalystBoardRunner
from src.agents.regime.regime_agent import RegimeAgent
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.technical_agent import TechnicalAgent
from src.db.phase2_recorder import Phase2Recorder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated Phase 2 Analyst Board pipeline.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to score. If omitted, discover active equity symbols from the data pipeline.",
    )
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Optional DATABASE_URL override.")
    parser.add_argument("--interval", default="1h", help="Interval used for runtime symbol discovery.")
    parser.add_argument("--models-dir", default="data/models", help="Directory containing trained technical model artifacts.")
    parser.add_argument("--technical-limit", type=int, default=300, help="History bars used by the technical agent loader.")
    parser.add_argument("--regime-limit", type=int, default=800, help="Gold-feature rows used by the regime agent.")
    parser.add_argument("--skip-technical", action="store_true", help="Skip technical predictions.")
    parser.add_argument("--skip-regime", action="store_true", help="Skip regime predictions.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment/cache reads.")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus aggregation.")
    parser.add_argument("--disable-persist", action="store_true", help="Do not write predictions to the DB.")
    parser.add_argument(
        "--refresh-sentiment-batch",
        action="store_true",
        help="Run the slow-lane sentiment batch first so cached z_t and aggregates are refreshed outside the execution path.",
    )
    parser.add_argument(
        "--sentiment-lookback-hours",
        type=int,
        default=24,
        help="Lookback window used when refreshing the sentiment batch.",
    )
    parser.add_argument(
        "--emit-phase3-observation",
        action="store_true",
        help="Include the derived Phase 3 observation payload for each symbol.",
    )
    parser.add_argument(
        "--register-model-cards",
        action="store_true",
        help="Register sentiment and consensus model cards before running the symbol loop.",
    )
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


def _resolve_symbols_to_score(args: argparse.Namespace) -> list[str]:
    explicit_symbols = dedupe_symbols(args.symbols or [])
    if explicit_symbols:
        logger.info("Using explicitly requested symbols: %s", explicit_symbols)
        return explicit_symbols

    loader = DataLoader(args.db_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"))

    def validate_symbol(symbol: str) -> SymbolValidationResult:
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
        requested_symbols=None,
        database_url=args.db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    symbols_to_score = list(discovery.active_symbols)
    if not symbols_to_score:
        raise SystemExit("No active equity symbols available for scoring.")
    return symbols_to_score


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else _default_output_path()
    _ensure_output_dir(output_path)
    symbols_to_score = _resolve_symbols_to_score(args)

    shared_recorder = None
    if not args.disable_persist:
        shared_recorder = Phase2Recorder(database_url=args.db_url)

    technical_agent = None
    regime_agent = None
    sentiment_agent = None
    consensus_agent = None
    if not args.skip_technical:
        technical_agent = TechnicalAgent(
            db_url=args.db_url,
            models_dir=args.models_dir,
            persist_predictions=not args.disable_persist,
        )
    if not args.skip_regime:
        regime_agent = RegimeAgent(
            database_url=args.db_url,
            persist_predictions=not args.disable_persist,
        )
    if not args.skip_sentiment:
        sentiment_agent = SentimentAgent.from_default_components(
            database_url=args.db_url,
            phase2_recorder=shared_recorder,
            persist_predictions=not args.disable_persist,
        )
    if not args.skip_consensus:
        consensus_agent = ConsensusAgent.from_default_components(
            phase2_recorder=shared_recorder,
        )

    runner = Phase2AnalystBoardRunner(
        technical_agent=technical_agent,
        regime_agent=regime_agent,
        sentiment_agent=sentiment_agent,
        consensus_agent=consensus_agent,
    )

    snapshot_id = f"phase2_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []
    failures: list[dict] = []
    generated_at = datetime.now(UTC)

    if args.register_model_cards:
        try:
            if sentiment_agent is not None:
                sentiment_agent.register_model_card(
                    extra_metadata={"status": "paper_ready", "last_registered_snapshot_id": snapshot_id}
                )
            if consensus_agent is not None:
                consensus_agent.register_model_cards(
                    extra_metadata={"status": "paper_ready", "last_registered_snapshot_id": snapshot_id}
                )
        except Exception as exc:
            logger.exception("Model card registration failed: %s", exc)
            failures.append({"agent": "model_cards", "error": str(exc)})

    if args.refresh_sentiment_batch:
        if sentiment_agent is None:
            logger.warning("Sentiment batch refresh requested but sentiment agent is skipped.")
        else:
            try:
                runner.refresh_sentiment_cache(
                    as_of_utc=generated_at,
                    lookback_hours=args.sentiment_lookback_hours,
                )
            except Exception as exc:
                logger.exception("Sentiment batch refresh failed: %s", exc)
                failures.append({"agent": "sentiment_batch", "error": str(exc)})

    for symbol in symbols_to_score:
        try:
            symbol_result = runner.run_symbol(
                symbol=symbol,
                snapshot_id=snapshot_id,
                technical_limit=args.technical_limit,
                regime_limit=args.regime_limit,
                as_of_utc=generated_at,
                emit_phase3_observation=args.emit_phase3_observation,
            )
            results.append(symbol_result)
        except Exception as exc:
            logger.exception("Phase 2 pipeline failed for %s: %s", symbol, exc)
            failures.append({"symbol": symbol, "agent": "phase2_pipeline", "error": str(exc)})

    payload = {
        "timestamp_utc": generated_at.isoformat(),
        "snapshot_id": snapshot_id,
        "db_url": args.db_url or os.getenv("DATABASE_URL"),
        "persist_enabled": not args.disable_persist,
        "symbols": symbols_to_score,
        "phase3_observation_emitted": args.emit_phase3_observation,
        "results": results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote Phase 2 prediction report to %s", output_path)

    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
