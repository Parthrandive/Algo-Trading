import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import EQUITY_SYMBOLS, dedupe_symbols, discover_pipeline_equity_symbols

logger = logging.getLogger("run_daily_training")
REQUIRED_ARTIFACT_META_FIELDS = (
    "run_id",
    "interval",
    "symbol_canonical",
    "source_script",
    "feature_schema_version",
)


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
    parser.add_argument(
        "--arima-daily-filter-mode",
        choices=["hard", "soft"],
        default="soft",
        help="Daily trend filter mode for universe ARIMA-LSTM retraining.",
    )
    parser.add_argument(
        "--arima-daily-up-penalty",
        type=float,
        default=0.15,
        help="UP confidence penalty used by ARIMA soft daily filter mode.",
    )
    parser.add_argument(
        "--arima-class-threshold-min",
        type=float,
        default=0.001,
        help="Minimum adaptive threshold floor for ARIMA label calibration.",
    )
    parser.add_argument(
        "--regime-min-gold-rows",
        type=int,
        default=120,
        help="Minimum Gold rows required before regime trainer falls back to ohlcv_bars.",
    )
    parser.add_argument(
        "--cnn-standalone-symbols",
        type=str,
        default="RELIANCE.NS,HDFCBANK.NS",
        help="Comma-separated symbols forced to use CNN standalone (skip XGBoost override).",
    )
    parser.add_argument(
        "--xgb-arima-readiness-symbols",
        type=str,
        default="RELIANCE.NS",
        help="Comma-separated symbols that must satisfy ARIMA directional readiness before XGBoost is allowed.",
    )
    parser.add_argument(
        "--xgb-arima-min-directional-accuracy",
        type=float,
        default=0.50,
        help="Minimum ARIMA directional_accuracy required for symbols in --xgb-arima-readiness-symbols.",
    )
    parser.add_argument(
        "--xgb-daily-regime-filter-mode",
        choices=["off", "long_only_above_ma200"],
        default="long_only_above_ma200",
        help="Daily regime filter mode applied in XGBoost stage.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["fixed", "atr", "percentile"],
        default="atr",
        help="Label construction mode for all training scripts.",
    )
    parser.add_argument(
        "--atr-k",
        type=float,
        default=0.5,
        help="ATR multiplier k for threshold = k × ATR/close.",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR lookback period for label construction.",
    )
    parser.add_argument(
        "--disable-arima-daily-filter",
        action="store_true",
        help="Disable the UP->NEUTRAL daily trend filter in ARIMA-LSTM (for A/B experiment).",
    )
    parser.add_argument(
        "--regime-hmm-overrides",
        type=str,
        default="TATASTEEL.NS:2,INFY.NS:2,TCS.NS:3,HDFCBANK.NS:3",
        help='Per-symbol HMM component overrides, e.g. "TATASTEEL.NS:2,INFY.NS:2".',
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_symbol_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return dedupe_symbols(parts)


def _validate_artifact_contract(
    symbols: list[str],
    run_id: str,
    interval: str = "1h",
    xgb_optional_symbols: Iterable[str] | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    xgb_optional = {str(s).strip() for s in (xgb_optional_symbols or [])}
    layers = [
        ("arima_lstm", PROJECT_ROOT / "data" / "models" / "arima_lstm"),
        ("cnn_pattern", PROJECT_ROOT / "data" / "models" / "cnn_pattern"),
        ("xgboost", PROJECT_ROOT / "data" / "models" / "xgboost"),
    ]
    for symbol in symbols:
        canonical_dir = symbol.replace(".", "_")
        for layer_name, root in layers:
            if layer_name == "xgboost" and symbol in xgb_optional:
                continue
            meta_path = root / canonical_dir / "training_meta.json"
            if not meta_path.exists():
                errors.append(f"{layer_name}:{symbol} missing training_meta.json at {meta_path}")
                continue
            try:
                payload = _load_json(meta_path)
            except Exception as exc:
                errors.append(f"{layer_name}:{symbol} cannot parse {meta_path}: {exc}")
                continue
            for field in REQUIRED_ARTIFACT_META_FIELDS:
                if field not in payload:
                    errors.append(f"{layer_name}:{symbol} missing field '{field}'")
            payload_run_id = str(payload.get("run_id", "")).strip()
            payload_interval = str(payload.get("interval", "")).strip()
            if payload_run_id != run_id:
                errors.append(f"{layer_name}:{symbol} run_id mismatch {payload_run_id} != {run_id}")
            if payload_interval != interval:
                errors.append(f"{layer_name}:{symbol} interval mismatch {payload_interval} != {interval}")
            split_counts = payload.get("split_counts")
            if isinstance(split_counts, dict):
                train_rows = int(split_counts.get("train", 0) or 0)
                val_rows = int(split_counts.get("val", 0) or 0)
                test_rows = int(split_counts.get("test", 0) or 0)
                if min(train_rows, val_rows, test_rows) <= 0:
                    errors.append(
                        f"{layer_name}:{symbol} has empty split counts "
                        f"(train={train_rows}, val={val_rows}, test={test_rows})"
                    )
    return len(errors) == 0, errors


def _validate_retrain_evidence(
    symbols: list[str],
    xgb_optional_symbols: Iterable[str] | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    xgb_optional = {str(s).strip() for s in (xgb_optional_symbols or [])}
    for symbol in symbols:
        canonical_dir = symbol.replace(".", "_")
        arima_meta = PROJECT_ROOT / "data" / "models" / "arima_lstm" / canonical_dir / "training_meta.json"
        cnn_meta = PROJECT_ROOT / "data" / "models" / "cnn_pattern" / canonical_dir / "training_meta.json"
        xgb_meta = PROJECT_ROOT / "data" / "models" / "xgboost" / canonical_dir / "training_meta.json"
        xgb_bundle = PROJECT_ROOT / "data" / "models" / "xgboost" / canonical_dir / "evaluation_bundle.json"

        if not arima_meta.exists():
            errors.append(f"arima evidence missing for {symbol}: {arima_meta}")
        else:
            payload = _load_json(arima_meta)
            if "metrics" not in payload:
                errors.append(f"arima evidence incomplete for {symbol}: missing metrics")
        if not cnn_meta.exists():
            errors.append(f"cnn evidence missing for {symbol}: {cnn_meta}")
        else:
            payload = _load_json(cnn_meta)
            if "metrics" not in payload:
                errors.append(f"cnn evidence incomplete for {symbol}: missing metrics")
        if symbol not in xgb_optional:
            if not xgb_meta.exists():
                errors.append(f"xgboost evidence missing for {symbol}: {xgb_meta}")
            else:
                payload = _load_json(xgb_meta)
                if "metrics" not in payload:
                    errors.append(f"xgboost evidence incomplete for {symbol}: missing metrics")
            if not xgb_bundle.exists():
                errors.append(f"xgboost evaluation bundle missing for {symbol}: {xgb_bundle}")
    return len(errors) == 0, errors


def _arima_directional_readiness(
    symbols: Iterable[str],
    min_directional_accuracy: float,
) -> tuple[dict[str, bool], list[str]]:
    readiness: dict[str, bool] = {}
    notes: list[str] = []
    threshold = float(min_directional_accuracy)
    for symbol in symbols:
        canonical_dir = symbol.replace(".", "_")
        meta_path = PROJECT_ROOT / "data" / "models" / "arima_lstm" / canonical_dir / "training_meta.json"
        if not meta_path.exists():
            readiness[symbol] = False
            notes.append(f"{symbol}: missing ARIMA training_meta at {meta_path}")
            continue
        payload = _load_json(meta_path)
        metrics = payload.get("metrics", {})
        directional_accuracy = float(metrics.get("directional_accuracy", 0.0) or 0.0)
        ready = directional_accuracy >= threshold
        readiness[symbol] = ready
        notes.append(
            f"{symbol}: directional_accuracy={directional_accuracy:.4f} "
            f"(threshold={threshold:.4f}) -> {'ready' if ready else 'blocked'}"
        )
    return readiness, notes


def run_training_pipeline(symbols: list[str], mode: str, run_id: str, args: argparse.Namespace) -> bool:
    all_success = True
    python_exec = sys.executable
    scripts_dir = PROJECT_ROOT / "scripts"
    
    logger.info(f"=== Starting {mode.upper()} Training Pipeline for {len(symbols)} symbols ===")
    logger.info("Run ID: %s", run_id)
    logger.info("Governance mode: paper/shadow only retraining; no live promotion in this pipeline.")
    cnn_standalone_symbols = set(parse_symbol_csv(args.cnn_standalone_symbols))
    arima_gate_symbols = set(parse_symbol_csv(args.xgb_arima_readiness_symbols))
    logger.info("CNN standalone override symbols: %s", sorted(cnn_standalone_symbols))
    
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
    regime_cmd.extend(["--min-gold-rows", str(int(args.regime_min_gold_rows))])
    if args.regime_hmm_overrides:
        regime_cmd.extend(["--symbol-hmm-overrides", str(args.regime_hmm_overrides)])
    if not run_cmd(regime_cmd):
        all_success = False

    # 3. ARIMA-LSTM Training (Dependent on Regime)
    logger.info("--- Stage 3: Combined Universe ARIMA-LSTM Agent ---")
    symbols_str = ",".join(symbols) if symbols else ""
    arima_cmd = [python_exec, str(scripts_dir / "train_universe_arima_lstm.py"), "--interval", "1h"]
    if symbols_str:
        arima_cmd.extend(["--symbols", symbols_str])
    arima_cmd.extend(["--run-id", run_id])
    arima_cmd.extend(
        [
            "--daily-filter-mode",
            str(args.arima_daily_filter_mode),
            "--daily-up-penalty",
            str(float(args.arima_daily_up_penalty)),
            "--class-threshold-min",
            str(float(args.arima_class_threshold_min)),
            "--label-mode",
            str(args.label_mode),
            "--atr-k",
            str(float(args.atr_k)),
            "--atr-period",
            str(int(args.atr_period)),
        ]
    )
    if args.disable_arima_daily_filter:
        arima_cmd.append("--disable-daily-trend-filter")
        
    if mode == "daily":
        arima_cmd.extend(["--epochs", "5"])
    else:
        arima_cmd.extend(["--epochs", "150"])
        
    if not run_cmd(arima_cmd):
        all_success = False

    arima_diag_cmd = [
        python_exec,
        str(scripts_dir / "diagnose_arima_regression.py"),
        "--interval",
        "1h",
        "--symbols",
        symbols_str,
        "--run-id",
        run_id,
    ]
    if not run_cmd(arima_diag_cmd):
        all_success = False

    arima_readiness, arima_readiness_notes = _arima_directional_readiness(
        arima_gate_symbols,
        min_directional_accuracy=float(args.xgb_arima_min_directional_accuracy),
    )
    arima_blocked_symbols = {s for s, ready in arima_readiness.items() if not ready}
    for note in arima_readiness_notes:
        logger.info("ARIMA readiness: %s", note)
        
    # 4. CNN Pattern Training (Weekly Only due to expense)
    if mode == "weekly":
        logger.info("--- Stage 4: Combined Universe CNN Pattern Agent ---")
        cnn_cmd = [python_exec, str(scripts_dir / "train_universe_cnn.py"), "--interval", "1h"]
        if symbols_str:
            cnn_cmd.extend(["--symbols", symbols_str])
        cnn_cmd.extend(["--epochs", "100", "--run-id", run_id])
        if not run_cmd(cnn_cmd):
            all_success = False
    else:
        logger.info("--- Stage 4: Combined Universe CNN Pattern Agent (SKIPPED in daily mode) ---")
        
    # 5. XGBoost Meta-Layer (Weekly Only, requires full base models)
    if mode == "weekly":
        logger.info("--- Stage 5: XGBoost Meta-Layer ---")
        xgb_symbols = [
            s for s in symbols
            if s not in cnn_standalone_symbols and s not in arima_blocked_symbols
        ]
        xgb_optional_symbols = set(cnn_standalone_symbols) | set(arima_blocked_symbols)
        if not xgb_symbols:
            logger.warning("All symbols bypassed for XGBoost. CNN standalone or ARIMA readiness gates are active.")
        else:
            xgb_symbols_str = ",".join(xgb_symbols)
            logger.info("XGBoost active symbols: %s", xgb_symbols)
        xgb_cmd = [
            python_exec,
            str(scripts_dir / "train_universe_xgboost.py"),
            "--interval",
            "1h",
            "--run-id",
            run_id,
            "--expected-run-id",
            run_id,
            "--strict-artifact-match",
            "--daily-regime-filter-mode",
            str(args.xgb_daily_regime_filter_mode),
        ]
        if xgb_symbols:
            xgb_cmd.extend(["--symbols", xgb_symbols_str])
            if not run_cmd(xgb_cmd):
                all_success = False
    else:
        logger.info("--- Stage 5: XGBoost Meta-Layer (SKIPPED in daily mode) ---")
        xgb_optional_symbols = set()
        
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

        logger.info("--- Stage 7: Artifact Contract Validation ---")
        contract_ok, contract_errors = _validate_artifact_contract(
            symbols,
            run_id,
            interval="1h",
            xgb_optional_symbols=xgb_optional_symbols,
        )
        if not contract_ok:
            all_success = False
            for error in contract_errors:
                logger.error("Artifact contract check failed: %s", error)
        else:
            logger.info("Artifact contract validation passed for run_id=%s", run_id)

        logger.info("--- Stage 8: Retrain Evidence Gate ---")
        evidence_ok, evidence_errors = _validate_retrain_evidence(
            symbols,
            xgb_optional_symbols=xgb_optional_symbols,
        )
        if not evidence_ok:
            all_success = False
            for error in evidence_errors:
                logger.error("Retrain evidence check failed: %s", error)
        else:
            logger.info("Retrain evidence gate passed for all symbols.")
            
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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
    start_time = time.time()
    success = run_training_pipeline(symbols, args.mode, run_id, args)
    duration = time.time() - start_time
    
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"=== Pipeline {status} in {duration:.1f} seconds ===")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
