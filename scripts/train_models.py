import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config.symbols import (
    SplitCounts,
    SymbolValidationResult,
    dedupe_symbols,
    discover_training_symbols,
    is_forex,
    validate_equity_symbol,
)


def run_command(command: list[str]) -> bool:
    """Run a shell command and stream output. Return True if successful."""
    cmd_str = " ".join(command)
    logger.info("Running: %s", cmd_str)
    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as exc:
        logger.error("Command failed with exit code %s: %s", exc.returncode, cmd_str)
        return False
    except Exception as exc:
        logger.error("Failed to execute command '%s': %s", cmd_str, exc)
        return False


def _default_db_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")


def _default_min_rows(interval: str) -> int:
    if interval == "1h":
        return 1200
    return 300


def _default_backfill_days(interval: str) -> int:
    if interval == "1h":
        return 365
    return 900


def _market_row_count(symbol: str, interval: str, db_url: str) -> int:
    from src.db.connection import get_engine

    engine = get_engine(db_url)
    query = text(
        """
        SELECT COUNT(*) AS n
        FROM ohlcv_bars
        WHERE symbol = :symbol
          AND interval = :interval
        """
    )
    with engine.connect() as conn:
        count = conn.execute(query, {"symbol": symbol, "interval": interval}).scalar()
    return int(count or 0)


def _latest_market_timestamp(symbol: str, interval: str, db_url: str) -> str | None:
    from src.db.connection import get_engine

    engine = get_engine(db_url)
    query = text(
        """
        SELECT MAX(timestamp) AS max_ts
        FROM ohlcv_bars
        WHERE symbol = :symbol
          AND interval = :interval
        """
    )
    with engine.connect() as conn:
        value = conn.execute(query, {"symbol": symbol, "interval": interval}).scalar()
    return None if value is None else str(value)


def _historical_quality(symbol: str, interval: str):
    try:
        from src.db.queries import get_market_data_quality

        return get_market_data_quality(symbol, interval, dataset_type="historical")
    except Exception as exc:
        logger.warning("Failed to read historical quality gate for %s: %s", symbol, exc)
        return None


def _run_preprocessing_for_symbol(symbol: str) -> bool:
    try:
        from src.agents.preprocessing.pipeline import PreprocessingPipeline

        snapshot_id = f"train_prep_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        pipeline = PreprocessingPipeline(config_path="configs/transform_config_v1.json")
        output = pipeline.process_snapshot(
            market_source_path="db_virtual",
            macro_source_path="db_virtual",
            text_source_path="db_virtual",
            snapshot_id=snapshot_id,
            symbol_filter=symbol,
        )
        logger.info("Preprocessing complete for %s: %s gold rows.", symbol, len(output.records))
        return True
    except Exception as exc:
        logger.error("Preprocessing failed for %s: %s", symbol, exc)
        return False


def _prepare_data_if_needed(args: argparse.Namespace) -> bool:
    symbol = args.symbol
    interval = args.interval
    db_url = _default_db_url()
    min_rows = args.min_rows if args.min_rows is not None else _default_min_rows(interval)
    backfill_days = args.backfill_days if args.backfill_days is not None else _default_backfill_days(interval)

    try:
        current_rows = _market_row_count(symbol, interval, db_url)
        latest_ts = _latest_market_timestamp(symbol, interval, db_url)
        logger.info(
            "Current market rows for %s (%s): %s (latest=%s)",
            symbol,
            interval,
            current_rows,
            latest_ts,
        )
    except Exception as exc:
        logger.error("Failed to inspect current market coverage: %s", exc)
        return False

    if current_rows < min_rows:
        logger.warning(
            "Rows below threshold (%s < %s). Auto-backfill will run for %s days.",
            current_rows,
            min_rows,
            backfill_days,
        )
        backfill_cmd = [
            sys.executable,
            str(Path("scripts") / "backfill_historical.py"),
            "--symbols",
            symbol,
            "--days",
            str(backfill_days),
            "--interval",
            interval,
            "--workers",
            str(args.backfill_workers),
            "--skip-recent-hours",
            "1",
            "--continue-on-error",
            "--force-refresh",
        ]
        if args.use_nse:
            # Backfill pipeline already uses failover providers; keep this flag for parity/logging.
            logger.info("use-nse flag set; backfill failover providers remain enabled.")
        if not run_command(backfill_cmd):
            return False

    if not args.skip_macro_refresh:
        macro_cmd = [sys.executable, "-m", "src.agents.macro.run_real_pipeline"]
        if not run_command(macro_cmd):
            return False

    if not args.skip_preprocessing_prep:
        if not _run_preprocessing_for_symbol(symbol):
            return False

    refreshed_rows = _market_row_count(symbol, interval, db_url)
    refreshed_latest = _latest_market_timestamp(symbol, interval, db_url)
    logger.info(
        "Post-prep market rows for %s (%s): %s (latest=%s)",
        symbol,
        interval,
        refreshed_rows,
        refreshed_latest,
    )

    quality = _historical_quality(symbol, interval)
    if quality is not None:
        logger.info(
            "Historical quality for %s (%s): status=%s train_ready=%s coverage=%s row_count=%s",
            symbol,
            interval,
            quality.get("status"),
            quality.get("train_ready"),
            quality.get("coverage_pct"),
            quality.get("row_count"),
        )
        if not quality.get("train_ready"):
            logger.error(
                "Symbol %s is gated out of training by historical quality rules: %s",
                symbol,
                quality.get("details_json"),
            )
            return False

    if refreshed_rows < min_rows:
        msg = (
            f"Rows still below threshold after prep ({refreshed_rows} < {min_rows}). "
            f"Increase --backfill-days or lower --min-rows."
        )
        if args.strict_min_rows:
            logger.error(msg)
            return False
        logger.warning(msg)
    return True


def _quality_gate_split_counts(df: pd.DataFrame) -> SplitCounts:
    n_rows = len(df)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return SplitCounts(
        train_rows=train_end,
        val_rows=val_end - train_end,
        test_rows=n_rows - val_end,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified Technical Agent Model Training Entrypoint.")
    parser.add_argument("--symbol", default=None, help="Single symbol override for all models")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to train when --symbol is not provided",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Optional skip flags
    parser.add_argument("--skip-arima-lstm", action="store_true", help="Skip ARIMA-LSTM training")
    parser.add_argument("--skip-cnn-pattern", action="store_true", help="Skip CNN Pattern training")
    parser.add_argument("--skip-garch-var", action="store_true", help="Skip GARCH VaR fitting")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip Day 5 walk-forward backtest")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1d", help="Candle interval for all models, e.g. 1d, 1h")

    # Auto prep controls
    parser.add_argument("--auto-prepare-data", dest="auto_prepare_data", action="store_true", help="Auto-backfill and preprocess before training")
    parser.add_argument("--no-auto-prepare-data", dest="auto_prepare_data", action="store_false", help="Skip auto data prep")
    parser.set_defaults(auto_prepare_data=True)
    parser.add_argument("--min-rows", type=int, default=None, help="Minimum market rows required before training")
    parser.add_argument("--backfill-days", type=int, default=None, help="Days of history to fetch when rows are insufficient")
    parser.add_argument("--backfill-workers", type=int, default=3, help="Workers for backfill_historical")
    parser.add_argument("--strict-min-rows", action="store_true", help="Fail if row threshold is still not met after prep")
    parser.add_argument("--skip-macro-refresh", action="store_true", help="Skip macro ingestion refresh during prep")
    parser.add_argument("--skip-preprocessing-prep", action="store_true", help="Skip preprocessing step during prep")

    args = parser.parse_args()

    db_url = _default_db_url()
    requested_symbols = [args.symbol] if args.symbol else []
    if args.symbols:
        requested_symbols.extend([s.strip() for s in args.symbols.split(",") if s.strip()])
    requested_symbols = dedupe_symbols(requested_symbols)

    def validate_symbol(symbol: str):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.symbol = symbol
        if trial_args.auto_prepare_data and not _prepare_data_if_needed(trial_args):
            return SymbolValidationResult(symbol=symbol, is_active=False, reason="auto_prepare_failed")
        from src.agents.technical.data_loader import DataLoader

        loader = DataLoader(db_url)
        try:
            frame = loader.load_historical_bars(
                symbol,
                limit=args.limit,
                use_nse_fallback=args.use_nse,
                min_fallback_rows=_default_min_rows(args.interval),
                interval=args.interval,
            )
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        return validate_equity_symbol(
            symbol=symbol,
            frame=frame,
            interval=args.interval,
            split_counts=_quality_gate_split_counts(frame),
        )

    discovery = discover_training_symbols(
        interval=args.interval,
        requested_symbols=requested_symbols or None,
        database_url=db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    training_symbols = list(discovery.active_symbols)
    if not training_symbols:
        logger.error("No symbols provided.")
        sys.exit(1)

    logger.info("=== Starting Unified Training for %s at %s interval ===", training_symbols, args.interval)

    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
        args.symbol = symbol

        common_args = ["--symbol", symbol, "--seed", str(args.seed), "--interval", args.interval]
        if args.limit is not None:
            common_args.extend(["--limit", str(args.limit)])
        if args.use_nse:
            common_args.append("--use-nse")

        logger.info(f"--- Running pipeline for {symbol} ---")

        # 1. ARIMA-LSTM
        if not args.skip_arima_lstm:
            script_path = os.path.join("scripts", "train_arima_lstm.py")
            cmd = [sys.executable, script_path] + common_args
            if not run_command(cmd):
                sys.exit(1)

        # 2. CNN Pattern
        if not args.skip_cnn_pattern:
            script_path = os.path.join("scripts", "train_cnn_pattern.py")
            cmd = [sys.executable, script_path] + common_args
            if not run_command(cmd):
                sys.exit(1)

        # 3. GARCH VaR
        if not args.skip_garch_var:
            script_path = os.path.join("scripts", "train_garch_var.py")
            cmd = [sys.executable, script_path] + common_args + ["--run-backtest"]
            if not run_command(cmd):
                sys.exit(1)

        # 4. Backtest & Ablation
        if not args.skip_backtest:
            script_path = os.path.join("scripts", "run_backtest.py")
            cmd = [sys.executable, script_path] + common_args
            if not run_command(cmd):
                sys.exit(1)
    # 5. Generate Model Cards
    logger.info("Generating Model Cards...")
    script_path = os.path.join("scripts", "generate_model_cards.py")
    cmd = [sys.executable, script_path]
    run_command(cmd)

    logger.info("=== All requested training and backtesting completed successfully! ===")


if __name__ == "__main__":
    main()
