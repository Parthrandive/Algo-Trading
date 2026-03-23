import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import EQUITY_SYMBOLS, dedupe_symbols, discover_pipeline_equity_symbols

logger = logging.getLogger("run_daily_training")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional Model Training Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["daily", "weekly"],
        default="daily",
        help="Training mode: 'daily' (incremental/append) or 'weekly' (full retrain)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Target symbols to train (defaults to EQUITY_SYMBOLS in config)",
    )
    return parser


def setup_logger() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def run_cmd(cmd: list[str]) -> bool:
    cmd_str = " ".join(cmd)
    logger.info(f"> {cmd_str}")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception running command: {e}")
        return False


def run_training_pipeline(symbols: list[str], mode: str) -> bool:
    all_success = True
    python_exec = sys.executable
    scripts_dir = PROJECT_ROOT / "scripts"
    
    logger.info(f"=== Starting {mode.upper()} Training Pipeline for {len(symbols)} symbols ===")
    
    # 1. GARCH VaR Training
    logger.info("--- Stage 1: GARCH VaR Training ---")
    for symbol in symbols:
        garch_cmd = [python_exec, str(scripts_dir / "train_garch_var.py"), "--symbol", symbol, "--interval", "1h"]
        if mode == "daily":
            logger.info(f"[{symbol}] GARCH: Performing incremental refit (on close prices)")
            # If garch had an --incremental flag, we'd pass it here
        if not run_cmd(garch_cmd):
            all_success = False
        
    # 2. Regime Model Training (Dependent on GARCH)
    logger.info("--- Stage 2: Regime Model Training ---")
    regime_cmd = [python_exec, str(scripts_dir / "train_regime_model.py"), "--interval", "1h", "--symbols"] + symbols
    if mode == "daily":
        # Incremental
        regime_cmd.extend(["--limit", "500"])
    else:
        # Full historical
        regime_cmd.extend(["--limit", "2000"])
    if not run_cmd(regime_cmd):
        all_success = False

    # 3. ARIMA-LSTM Training (Dependent on Regime)
    logger.info("--- Stage 3: Combined Universe ARIMA-LSTM Agent ---")
    symbols_str = ",".join(symbols) if symbols else ""
    arima_cmd = [python_exec, str(scripts_dir / "train_universe_arima_lstm.py"), "--interval", "1h"]
    if symbols_str:
        arima_cmd.extend(["--symbols", symbols_str])
        
    if mode == "daily":
        arima_cmd.extend(["--epochs", "5"])
    else:
        arima_cmd.extend(["--epochs", "150"])
        
    if not run_cmd(arima_cmd):
        all_success = False
        
    # 4. CNN Pattern Training (Weekly Only due to expense)
    if mode == "weekly":
        logger.info("--- Stage 4: Combined Universe CNN Pattern Agent ---")
        cnn_cmd = [python_exec, str(scripts_dir / "train_universe_cnn.py"), "--interval", "1h"]
        if symbols_str:
            cnn_cmd.extend(["--symbols", symbols_str])
        cnn_cmd.extend(["--epochs", "100"])
        if not run_cmd(cnn_cmd):
            all_success = False
    else:
        logger.info("--- Stage 4: Combined Universe CNN Pattern Agent (SKIPPED in daily mode) ---")
        
    # 5. XGBoost Meta-Layer (Weekly Only, requires full base models)
    if mode == "weekly":
        logger.info("--- Stage 5: XGBoost Meta-Layer ---")
        xgb_cmd = [python_exec, str(scripts_dir / "train_universe_xgboost.py"), "--interval", "1h"]
        if not run_cmd(xgb_cmd):
            all_success = False
    else:
        logger.info("--- Stage 5: XGBoost Meta-Layer (SKIPPED in daily mode) ---")
        
    # 6. Validation (Run Tests)
    logger.info("--- Stage 6: Validation & Tests ---")
    test_cmd = ["pytest", "tests/agents/preprocessing/test_leakage.py", "-v"]
    if not run_cmd(test_cmd):
        logger.warning("Leakage test failed!")
        all_success = False

    if mode == "weekly":
        test_xgb = ["pytest", "tests/test_xgboost_meta_layer.py", "-v"]
        if not run_cmd(test_xgb):
            all_success = False
            
    return all_success


def main() -> int:
    setup_logger()
    args = _build_parser().parse_args()
    
    symbols = args.symbols if args.symbols else EQUITY_SYMBOLS
    symbols = dedupe_symbols(symbols)
    
    if not symbols:
        logger.info("No explicit symbols provided. Auto-discovering from database...")
        try:
            symbols = discover_pipeline_equity_symbols(interval="1h")
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            
    if not symbols:
        logger.error("No valid symbols discovered or provided for training.")
        return 1
        
    start_time = time.time()
    success = run_training_pipeline(symbols, args.mode)
    duration = time.time() - start_time
    
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"=== Pipeline {status} in {duration:.1f} seconds ===")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
