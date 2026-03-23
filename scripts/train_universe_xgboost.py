import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import (
    EQUITY_SYMBOLS,
    FOREX_SYMBOLS,
    MIN_ROWS,
    dedupe_symbols,
)


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def canonicalize_symbol(symbol: str) -> str:
    value = str(symbol).strip()
    if value.endswith("_NS") and "." not in value:
        return value[:-3] + ".NS"
    return value


from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_FREQ = "1h"
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2

def parse_args():
    parser = argparse.ArgumentParser(description="Train Universe XGBoost Meta-Layer")
    parser.add_argument("--cnn-dir", type=str, default="data/models/cnn_pattern", help="CNN models directory")
    parser.add_argument("--lstm-dir", type=str, default="data/models/arima_lstm", help="LSTM models directory")
    parser.add_argument("--output-dir", type=str, default="data/models/xgboost", help="XGBoost output directory")
    parser.add_argument("--interval", type=str, default=PRIMARY_FREQ, help="Candle interval")
    parser.add_argument("--limit", type=int, default=4000, help="Rows to load per symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

def maybe_load_regime_labels(symbol: str, base_path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    default_regime = np.full(len(frame), -1, dtype=np.int64)
    shift_flag = np.zeros(len(frame), dtype=np.int64)
    candidate_paths = [
        base_path / "regime_agent" / "hmm_regime.parquet",
        base_path / "hmm_regime.parquet",
        PROJECT_ROOT / "data" / "models" / "regime" / f"{sanitize_symbol(symbol)}_hmm_regime.parquet",
    ]
    regime_path = first_existing(candidate_paths)
    if regime_path is None:
        return default_regime, shift_flag

    try:
        regime_df = pd.read_parquet(regime_path)
    except Exception as exc:
        return default_regime, shift_flag

    ts_col = "timestamp" if "timestamp" in regime_df.columns else regime_df.columns[0]
    label_col = "hmm_regime" if "hmm_regime" in regime_df.columns else regime_df.columns[1]

    regime_data = regime_df[[ts_col, label_col]].copy()
    regime_data[ts_col] = pd.to_datetime(regime_data[ts_col], utc=True, errors="coerce")
    regime_data = regime_data.dropna().sort_values(ts_col)
    
    if regime_data.empty:
        return default_regime, shift_flag

    merged = pd.merge_asof(
        frame[["timestamp"]].sort_values("timestamp"),
        regime_data.rename(columns={ts_col: "timestamp", label_col: "hmm_regime"}).sort_values("timestamp"),
        on="timestamp", direction="backward"
    )
    hmm_regime = pd.to_numeric(merged["hmm_regime"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    
    dominant = pd.Series(hmm_regime[hmm_regime >= 0]).mode()
    if not dominant.empty:
        shift_flag = (hmm_regime != int(dominant.iloc[0])).astype(np.int64)

    return hmm_regime, shift_flag

def build_symbol_data(symbol: str, cnn_dir: Path, lstm_dir: Path, loader: DataLoader, args) -> dict[str, pd.DataFrame] | None:
    """
    Loads CNN and LSTM predictions, engineers technical features, 
    and merges them into a single dataset.
    """
    cnn_path = first_existing(
        [
            cnn_dir / sanitize_symbol(symbol) / "predictions.parquet",
            cnn_dir / symbol / "predictions.parquet",
        ]
    )
    lstm_path = first_existing(
        [
            lstm_dir / sanitize_symbol(symbol) / "predictions.parquet",
            lstm_dir / symbol / "predictions.parquet",
        ]
    )
    
    if cnn_path is None:
        logger.warning("[%s] Missing CNN predictions: %s", symbol, cnn_path)
        return None
    if lstm_path is None:
        logger.warning("[%s] Missing LSTM predictions: %s", symbol, lstm_path)
        return None
        
    cnn_df = pd.read_parquet(cnn_path)
    lstm_df = pd.read_parquet(lstm_path)
    
    frame = loader.load_historical_bars(symbol, limit=args.limit, interval=args.interval)
    if len(frame) < 100:
        logger.warning("[%s] Insufficient data.", symbol)
        return None
        
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame = engineer_features(frame, is_forex=False)
    
    tech_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in ("timestamp", "symbol", "close", "log_return")]
    
    next_returns = frame["log_return"].shift(-1).to_numpy()
    hmm_regime, regime_shift_flag = maybe_load_regime_labels(symbol, PROJECT_ROOT / "data" / "models" / "regime", frame)
    
    meta_df = pd.DataFrame({"timestamp": frame["timestamp"]})
    for c in tech_cols:
        meta_df[f"tech_{c}"] = frame[c]
    
    meta_df["hmm_regime"] = hmm_regime
    meta_df["regime_shift_flag"] = regime_shift_flag
    meta_df["actual_next_log_return"] = next_returns
    
    meta_df["y_label"] = -1
    threshold = 0.005
    valid_mask = np.isfinite(next_returns)
    ret = next_returns[valid_mask]
    lbls = np.full(len(ret), CLASS_NEUTRAL, dtype=np.int64)
    lbls[ret > threshold] = CLASS_UP
    lbls[ret < -threshold] = CLASS_DOWN
    meta_df.loc[valid_mask, "y_label"] = lbls
    
    aligned = pd.merge(cnn_df, lstm_df, on=["timestamp", "split"], how="inner")
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
    meta_df["timestamp"] = pd.to_datetime(meta_df["timestamp"], utc=True)

    final_df = pd.merge(aligned, meta_df, on="timestamp", how="inner")
    
    final_df["cnn_up_reliable"] = (final_df["cnn_prob_up"] > 0.45).astype(np.int64)
    final_df["cnn_confidence"] = final_df[["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].max(axis=1)
    final_df["lstm_confidence"] = final_df[["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].max(axis=1)
    cnn_argmax = np.argmax(final_df[["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(), axis=1)
    lstm_argmax = np.argmax(final_df[["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(), axis=1)
    final_df["model_agreement"] = (cnn_argmax == lstm_argmax).astype(np.int64)

    return {
        "train": final_df[final_df["split"] == "train"].copy(),
        "val": final_df[final_df["split"] == "val"].copy(),
        "test": final_df[final_df["split"] == "test"].copy(),
    }

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    
    cnn_dir = Path(args.cnn_dir)
    lstm_dir = Path(args.lstm_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    discovered_symbols = [canonicalize_symbol(d.name) for d in cnn_dir.iterdir() if d.is_dir() and d.name != "GLOBAL"]
    symbols = dedupe_symbols([*EQUITY_SYMBOLS, *discovered_symbols])
    
    for symbol in symbols:
        if symbol in FOREX_SYMBOLS: continue
        logger.info("========== Processing XGBoost for %s ==========", symbol)
        
        splits = build_symbol_data(symbol, cnn_dir, lstm_dir, loader, args)
        if not splits: continue
        
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        valid_train = train_df[train_df["y_label"] >= 0]
        valid_val = val_df[val_df["y_label"] >= 0]
        
        feature_cols = [c for c in train_df.columns if c not in ("timestamp", "split", "y_label", "actual_next_log_return")]
        
        X_train = np.ascontiguousarray(valid_train[feature_cols].to_numpy(dtype=np.float32))
        y_train = np.ascontiguousarray(valid_train["y_label"].to_numpy(dtype=np.int64))
        X_val = np.ascontiguousarray(valid_val[feature_cols].to_numpy(dtype=np.float32))
        y_val = np.ascontiguousarray(valid_val["y_label"].to_numpy(dtype=np.int64))
        
        sample_weight = np.ascontiguousarray(compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32))
        
        xgb = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss", early_stopping_rounds=30,
            random_state=args.seed, n_jobs=-1
        )
        xgb.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        
        sym_out = output_root / sanitize_symbol(symbol)
        sym_out.mkdir(parents=True, exist_ok=True)
        xgb.save_model(sym_out / "model.json")
        
        logger.info("[%s] XGBoost trained. Best iteration: %s", symbol, getattr(xgb, "best_iteration", 0))

if __name__ == "__main__":
    main()
