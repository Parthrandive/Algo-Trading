import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# VOLATILITY PROXY IMPORT - Kept as modular utility for future data-starved symbols
# NOTE: Not applied to TATASTEEL per user override; TATASTEEL uses 2-state HMM.
try:
    from src.agents.regime.volatility_proxy_regime import compute_volatility_proxy_regime
except ImportError:
    pass

from src.agents.regime.data_loader import RegimeDataLoader
from src.agents.regime.regime_agent import RegimeAgent
from config.symbols import EQUITY_SYMBOLS, dedupe_symbols, discover_pipeline_equity_symbols

logger = logging.getLogger("train_regime_model")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Regime Model Training")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Target symbols (defaults to EQUITY_SYMBOLS)",
    )
    parser.add_argument("--limit", type=int, default=2000, help="Max Gold feature rows to load (default: 2000)")
    parser.add_argument(
        "--min-gold-rows",
        type=int,
        default=120,
        help="Minimum acceptable Gold rows before falling back to ohlcv_bars for regime training.",
    )
    parser.add_argument("--interval", type=str, default="1d", help="Interval to load (e.g., 1d, 1h)")
    parser.add_argument("--output-dir", default="data/models/regime", help="Output directory for model artifacts")
    parser.add_argument("--database-url", default=None, help="Database URL")
    parser.add_argument(
        "--hmm-components",
        type=int,
        default=None,
        help="Fixed HMM component count for all symbols. None = auto-select based on data volume.",
    )
    parser.add_argument(
        "--symbol-hmm-overrides",
        type=str,
        default=None,
        help='Per-symbol HMM component overrides, e.g. "TATASTEEL.NS:2,INFY.NS:2,TCS.NS:3".',
    )
    return parser


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def setup_logger(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    fh = logging.FileHandler(output_dir / "train_regime.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _parse_hmm_overrides(raw: str | None) -> dict[str, int]:
    """Parse 'SYM1:N,SYM2:M' into {symbol: n_components} dict."""
    if not raw:
        return {}
    overrides: dict[str, int] = {}
    for part in str(raw).split(","):
        part = part.strip()
        if ":" not in part:
            continue
        sym, n = part.rsplit(":", 1)
        try:
            overrides[sym.strip()] = int(n.strip())
        except ValueError:
            logger.warning("Skipping invalid HMM override: %s", part)
    return overrides


def train_for_symbol(symbol: str, args: argparse.Namespace, hmm_overrides: dict[str, int] | None = None) -> bool:
    logger.info(f"[{symbol}] Starting regime model training")
    out_dir = Path(args.output_dir) / symbol.replace("/", "_").replace("=", "_").replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    loader = RegimeDataLoader(database_url=args.database_url)
    raw = loader.load_features(
        symbol=symbol,
        limit=args.limit,
        interval=args.interval,
        min_gold_rows=int(args.min_gold_rows),
    )
    
    if raw.empty:
        logger.error(f"[{symbol}] No gold features available for training.")
        return False
        
    logger.info(f"[{symbol}] Loaded {len(raw)} rows from Gold tier.")
    
    # Determine HMM component count
    hmm_overrides = hmm_overrides or {}
    if symbol in hmm_overrides:
        n_components = hmm_overrides[symbol]
        logger.info(f"[{symbol}] Using per-symbol HMM override: {n_components} components")
    elif args.hmm_components is not None:
        n_components = int(args.hmm_components)
        logger.info(f"[{symbol}] Using fixed HMM components: {n_components}")
    else:
        n_components = RegimeAgent.auto_select_components(len(raw))
        logger.info(f"[{symbol}] Auto-selected HMM components: {n_components} (based on {len(raw)} rows)")

    # FIX 3: TCS Override Fallback Loop
    n_components_to_try = [n_components]
    if symbol == "TCS.NS" and n_components == 3:
        n_components_to_try = [3, 2]

    agent = None
    prepared = None
    
    for current_n_comp in n_components_to_try:
        logger.info(f"[{symbol}] Attempting HMM fit with {current_n_comp} components...")
        agent = RegimeAgent(loader=loader, persist_predictions=False, n_components=current_n_comp)
        prepared = agent._prepare_features(raw)
        
        if len(prepared) < agent.warmup_rows:
            logger.error(f"[{symbol}] Not enough rows for warmup (need {agent.warmup_rows}, got {len(prepared)}).")
            return False
            
        agent.hmm.fit(prepared)
        
        # Check occupancy early
        predicted_states = agent.hmm.predict_sequence(prepared)
        state_counts = pd.Series(predicted_states).value_counts(normalize=True)
        min_occupancy = float(state_counts.min()) if not state_counts.empty else 0.0
        
        if symbol == "TCS.NS" and current_n_comp == 3 and (len(state_counts) < 3 or min_occupancy < 0.15):
            logger.warning(f"[{symbol}] 3-state HMM has state with < 15% occupancy ({min_occupancy:.1%}). Falling back to 2-state.")
            continue # Try next fallback
            
        break # Success

    logger.info(f"[{symbol}] Fitting Regime Ensemble (HMM, PEARL, OOD)...")
    assert agent is not None and prepared is not None
    agent.pearl.fit(prepared)
    agent.ood.fit(prepared)
    
    # Generate labels for XGBoost meta-layer
    logger.info(f"[{symbol}] Generating regime labels for XGBoost...")
    results = []
    
    for i in range(agent.warmup_rows, len(prepared)):
        sub_df = prepared.iloc[:i+1]
        
        # Predict directly using the fitted models
        hmm_pred = agent.hmm.predict(sub_df)
        pearl_pred = agent.pearl.predict(sub_df)
        ood_result = agent.ood.detect(sub_df.tail(min(120, len(sub_df))))
        
        regime_state = agent._ensemble_state(
            hmm_pred.regime_state, pearl_pred.regime_state, 
            hmm_pred.confidence, pearl_pred.confidence
        )
        if ood_result.is_alien:
            regime_state = "ALIEN"
        else:
            regime_state = regime_state.value
            
        transition_probability = agent._blend_transition(
            hmm_pred.transition_probability, pearl_pred.transition_probability, ood_result.is_warning
        )
        confidence = agent._blend_confidence(
            hmm_pred.confidence, pearl_pred.confidence, ood_result.is_warning, ood_result.is_alien
        )
            
        results.append({
            "timestamp": sub_df.iloc[-1]["timestamp"],
            "hmm_regime": hmm_pred.hidden_state,
            "regime_state": regime_state,
            "confidence": confidence,
            "transition_probability": transition_probability,
            "is_warning": ood_result.is_warning,
            "is_alien": ood_result.is_alien,
        })
        
    results_df = pd.DataFrame(results)
    parquet_path = out_dir / "hmm_regime.parquet"
    results_df.to_parquet(parquet_path, index=False)
    logger.info(f"[{symbol}] Saved regime labels to {parquet_path}")
    
    # Save training metadata
    run_timestamp = datetime.now(timezone.utc).isoformat()
    meta = {
        "timestamp": run_timestamp,
        "symbol": symbol,
        "run_timestamp_utc": run_timestamp,
        "rows_trained": len(prepared),
        "start_timestamp": str(prepared.iloc[0]["timestamp"]),
        "end_timestamp": str(prepared.iloc[-1]["timestamp"]),
        "artifacts_saved": [
            "hmm_regime.parquet",
            "training_meta.json"
        ],
        "hmm_components": agent.hmm.n_components,
        "hyperparameters": {
            "interval": args.interval,
            "limit": int(args.limit),
            "min_gold_rows": int(args.min_gold_rows),
            "warmup_rows": int(agent.warmup_rows),
            "hmm_components": int(agent.hmm.n_components),
        },
        "metrics": {
            "regime_distribution": results_df["hmm_regime"].value_counts(normalize=True).to_dict()
        }
    }
    
    meta_path = out_dir / "training_meta.json"

    # FIX 3: Detailed Regime Summary Logging
    regime_dist = dict(results_df["hmm_regime"].value_counts(normalize=True))
    min_occ = min(regime_dist.values()) if regime_dist else 0.0
    logger.info(
        f"[{symbol}] REGIME SUMMARY | n_states: {agent.hmm.n_components} | "
        f"min_occupancy: {min_occ:.1%} | "
        f"distribution: {', '.join(f'{k}:{v:.1%}' for k,v in regime_dist.items())}"
    )

    for state_id, occupancy in regime_dist.items():
        if occupancy < 0.10:
            logger.warning(
                f"[{symbol}] Regime state {state_id} has only {occupancy:.1%} occupancy "
                f"({int(occupancy * len(results_df))} rows). Consider reducing n_components."
            )

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[{symbol}] Saved metadata to {meta_path}")
    
    return True


def main() -> int:
    args = parse_args()
    setup_logger(Path(args.output_dir))
    
    symbols = args.symbols if args.symbols else EQUITY_SYMBOLS
    symbols = dedupe_symbols(symbols)
    hmm_overrides = _parse_hmm_overrides(args.symbol_hmm_overrides)
    
    if not symbols:
        logger.info("No explicit symbols provided. Auto-discovering from database...")
        try:
            symbols = discover_pipeline_equity_symbols(interval="1d", database_url=args.database_url)
        except Exception as e:
            logger.error(f"Failed to discover symbols: {e}")
            
    if not symbols:
        logger.error("No valid symbols discovered or provided for training.")
        return 1
        
    logger.info(f"Starting regime training for symbols: {symbols}")
    
    success_count = 0
    for symbol in symbols:
        if train_for_symbol(symbol, args, hmm_overrides=hmm_overrides):
            success_count += 1
            
    logger.info(f"Finished training. Success: {success_count}/{len(symbols)}")
    return 0 if success_count == len(symbols) else 1


if __name__ == "__main__":
    sys.exit(main())
