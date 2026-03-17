import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.regime.regime_agent import RegimeAgent
from src.agents.technical.technical_agent import TechnicalAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run available Phase 2 agents and persist their predictions.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["USDINR=X"],
        help="Symbols to score, e.g. USDINR=X POWERGRID.NS LT.NS",
    )
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Optional DATABASE_URL override.")
    parser.add_argument("--models-dir", default="data/models", help="Directory containing trained technical model artifacts.")
    parser.add_argument("--technical-limit", type=int, default=300, help="History bars used by the technical agent loader.")
    parser.add_argument("--regime-limit", type=int, default=800, help="Gold-feature rows used by the regime agent.")
    parser.add_argument("--skip-technical", action="store_true", help="Skip technical predictions.")
    parser.add_argument("--skip-regime", action="store_true", help="Skip regime predictions.")
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


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else _default_output_path()
    _ensure_output_dir(output_path)

    technical_agent = None
    regime_agent = None
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

    snapshot_id = f"phase2_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []
    failures: list[dict] = []

    for symbol in args.symbols:
        symbol_result = {
            "symbol": symbol,
            "technical": None,
            "regime": None,
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

        symbol_result["not_implemented_agents"] = ["sentiment", "consensus"]
        results.append(symbol_result)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "db_url": args.db_url or os.getenv("DATABASE_URL"),
        "persist_enabled": not args.disable_persist,
        "symbols": args.symbols,
        "results": results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote Phase 2 prediction report to %s", output_path)

    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
