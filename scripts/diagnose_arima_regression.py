import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import EQUITY_SYMBOLS, dedupe_symbols, discover_pipeline_equity_symbols, is_forex
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("diagnose_arima_regression")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose ARIMA-LSTM regression artifacts by symbol.")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols.")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval.")
    parser.add_argument("--run-id", type=str, default=None, help="Expected run_id to validate against artifacts.")
    parser.add_argument("--models-dir", type=str, default="data/models/arima_lstm", help="ARIMA artifact root.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Defaults to data/reports/arima_diagnostics/arima_diag_<run>.json",
    )
    parser.add_argument("--limit", type=int, default=600, help="Bars for log-return consistency check.")
    return parser


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return dedupe_symbols([s.strip() for s in raw.split(",") if s.strip()])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _check_log_return_consistency(loader: DataLoader, symbol: str, interval: str, limit: int) -> dict[str, Any]:
    try:
        bars = loader.load_historical_bars(symbol=symbol, interval=interval, limit=limit)
        if bars.empty or "close" not in bars.columns:
            return {"status": "missing_data"}

        feat = engineer_features(bars, is_forex=is_forex(symbol))
        if "log_return" not in feat.columns or "close" not in feat.columns:
            return {"status": "missing_log_return_feature"}

        close = pd.to_numeric(feat["close"], errors="coerce")
        manual = np.log(close / close.shift(1))
        model_lr = pd.to_numeric(feat["log_return"], errors="coerce")
        valid = manual.notna() & model_lr.notna()
        if not bool(valid.any()):
            return {"status": "no_valid_overlap"}

        diff = (manual[valid] - model_lr[valid]).abs()
        return {
            "status": "ok",
            "overlap_rows": int(valid.sum()),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def main() -> int:
    args = _build_parser().parse_args()
    expected_run_id = str(args.run_id).strip() if args.run_id else None
    models_dir = Path(args.models_dir)

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        symbols = dedupe_symbols(EQUITY_SYMBOLS)
    if not symbols:
        symbols = discover_pipeline_equity_symbols(interval=args.interval)

    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "expected_run_id": expected_run_id,
        "symbols": [],
    }

    for symbol in symbols:
        canonical = symbol.replace(".", "_")
        meta_path = models_dir / canonical / "training_meta.json"
        entry: dict[str, Any] = {
            "symbol": symbol,
            "meta_path": str(meta_path),
            "meta_exists": meta_path.exists(),
        }
        if not meta_path.exists():
            logger.warning("[%s] Missing ARIMA training_meta: %s", symbol, meta_path)
            report["symbols"].append(entry)
            continue

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        hyper = payload.get("hyperparameters", {})
        pred_dist = metrics.get("prediction_distribution", {}).get("test", {})

        run_id = str(payload.get("run_id", "")).strip()
        entry.update(
            {
                "run_id": run_id,
                "run_id_match": (expected_run_id is None) or (run_id == expected_run_id),
                "trained_as_part_of_universe": bool(payload.get("trained_as_part_of_universe", False)),
                "base_model_symbol": str(hyper.get("base_model_symbol", "")),
                "class_threshold": _safe_float(hyper.get("class_threshold")),
                "class_threshold_min": _safe_float(hyper.get("class_threshold_min")),
                "effective_class_threshold": _safe_float(hyper.get("effective_class_threshold")),
                "test_mse": _safe_float(metrics.get("test_mse")),
                "test_accuracy": _safe_float(metrics.get("test_accuracy")),
                "directional_accuracy": _safe_float(metrics.get("directional_accuracy")),
                "up_precision": _safe_float(metrics.get("up_precision")),
                "up_recall": _safe_float(metrics.get("up_recall")),
                "down_precision": _safe_float(metrics.get("down_precision")),
                "down_recall": _safe_float(metrics.get("down_recall")),
                "prediction_distribution_test": pred_dist,
            }
        )
        entry["log_return_consistency"] = _check_log_return_consistency(
            loader=loader,
            symbol=symbol,
            interval=args.interval,
            limit=int(args.limit),
        )
        logger.info(
            "[%s] run_id=%s mse=%.6f dir_acc=%.4f threshold=%.6f up_recall=%.4f down_recall=%.4f",
            symbol,
            run_id,
            entry["test_mse"],
            entry["directional_accuracy"],
            entry["effective_class_threshold"],
            entry["up_recall"],
            entry["down_recall"],
        )
        report["symbols"].append(entry)

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "reports" / "arima_diagnostics" / f"arima_diag_{expected_run_id or 'latest'}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote ARIMA diagnostic report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
