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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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
ANNUALIZATION_1H = int(252.0 * 6.0)
COST_PER_SIDE = 0.0003

def compute_backtest_metrics(
    probs: np.ndarray,
    actual_next_log_returns: np.ndarray,
    threshold: float | None = None,
    min_hold_bars: int = 3,
) -> dict[str, float]:
    probabilities = np.asarray(probs, dtype=np.float64)
    confidence_threshold = float(threshold if threshold is not None else 0.0)
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 0.9999))
    denominator = max(1.0 - confidence_threshold, 1e-12)
    min_hold = max(int(min_hold_bars), 0)

    positions = np.zeros(len(probabilities), dtype=np.float64)
    last_change_idx = -10_000_000
    eps = 1e-9

    for idx in range(len(probabilities)):
        row = probabilities[idx]
        raw_confidence = float(np.max(row))
        signal = int(np.argmax(row))

        if raw_confidence < confidence_threshold:
            desired_direction = 0.0
            desired_size = 0.0
        else:
            excess = raw_confidence - confidence_threshold
            position_size = float(np.clip(excess / denominator, 0.0, 1.0))
            if signal == CLASS_UP:
                desired_direction = 1.0
            elif signal == CLASS_DOWN:
                desired_direction = -1.0
            else:
                desired_direction = 0.0
            desired_size = position_size if desired_direction != 0.0 else 0.0

        desired_position = float(desired_direction * desired_size)

        if idx == 0:
            positions[idx] = desired_position
            if abs(desired_position) > eps:
                last_change_idx = idx
            continue

        prev_position = positions[idx - 1]
        prev_direction = 0.0 if abs(prev_position) <= eps else float(np.sign(prev_position))
        next_direction = 0.0 if abs(desired_position) <= eps else float(np.sign(desired_position))

        if prev_direction == next_direction and prev_direction != 0.0:
            desired_position = prev_position

        has_change = abs(desired_position - prev_position) > eps
        can_change = (idx - last_change_idx) >= min_hold

        if has_change and not can_change:
            positions[idx] = prev_position
        else:
            positions[idx] = desired_position
            if has_change:
                last_change_idx = idx

    returns = np.expm1(np.asarray(actual_next_log_returns, dtype=np.float64))
    returns = np.where(np.isfinite(returns), returns, 0.0)

    prev_positions = np.roll(positions, 1)
    prev_positions[0] = 0
    turnover = np.abs(positions - prev_positions)
    transaction_cost = turnover * COST_PER_SIDE

    gross_returns = positions * returns
    net_returns = gross_returns - transaction_cost

    mean_return = float(np.mean(net_returns)) if len(net_returns) else 0.0
    std_return = float(np.std(net_returns, ddof=0)) if len(net_returns) else 0.0
    sharpe = float(mean_return / std_return * ANNUALIZATION_1H) if std_return > 1e-12 else 0.0

    downside = net_returns[net_returns < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    sortino = float(mean_return / downside_std * ANNUALIZATION_1H) if downside_std > 1e-12 else 0.0

    equity_curve = np.cumprod(1.0 + net_returns)
    if len(equity_curve):
        peaks = np.maximum.accumulate(equity_curve)
        drawdown = 1.0 - (equity_curve / np.clip(peaks, 1e-12, None))
        max_drawdown = float(np.max(drawdown))
    else:
        max_drawdown = 0.0

    active_mask = np.abs(positions) > eps
    active_returns = net_returns[active_mask]
    win_rate = float(np.mean(active_returns > 0.0)) if len(active_returns) else 0.0
    pos_sum = float(active_returns[active_returns > 0.0].sum()) if len(active_returns) else 0.0
    neg_sum = float(np.abs(active_returns[active_returns < 0.0].sum())) if len(active_returns) else 0.0
    profit_factor = float("inf") if neg_sum <= 1e-12 and pos_sum > 0 else pos_sum / max(neg_sum, 1e-12)

    total_trades = int(np.sum(turnover > eps))
    average_trade_return = float(np.mean(active_returns)) if len(active_returns) else 0.0
    coverage = float(np.mean(active_mask)) if len(active_mask) else 0.0

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "total_trades": total_trades,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Universe XGBoost Meta-Layer")
    parser.add_argument("--symbol", type=str, default=None, help="Process a single symbol")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols to process")
    parser.add_argument("--cnn-dir", type=str, default="data/models/cnn_pattern", help="CNN models directory")
    parser.add_argument("--lstm-dir", type=str, default="data/models/arima_lstm", help="LSTM models directory")
    parser.add_argument("--output-dir", type=str, default="data/models/xgboost", help="XGBoost output directory")
    parser.add_argument("--interval", type=str, default=PRIMARY_FREQ, help="Candle interval")
    parser.add_argument("--limit", type=int, default=4000, help="Rows to load per symbol")
    parser.add_argument("--class-threshold", type=float, default=0.005, help="Return threshold used to derive down/neutral/up labels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[CLASS_UP, CLASS_DOWN],
        average=None,
        zero_division=0,
    )
    directional_mask = np.isin(y_true, [CLASS_UP, CLASS_DOWN])
    directional_accuracy = (
        float(accuracy_score(y_true[directional_mask], y_pred[directional_mask]))
        if directional_mask.any()
        else 0.0
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy,
        "up_precision": float(precision[0]),
        "up_recall": float(recall[0]),
        "up_f1": float(f1[0]),
        "up_support": int(support[0]),
        "down_precision": float(precision[1]),
        "down_recall": float(recall[1]),
        "down_f1": float(f1[1]),
        "down_support": int(support[1]),
        "test_confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP],
        ).tolist(),
    }

def maybe_load_regime_labels(symbol: str, base_path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    default_regime = np.full(len(frame), -1, dtype=np.int64)
    shift_flag = np.zeros(len(frame), dtype=np.int64)
    candidate_paths = [
        base_path / "regime_agent" / "hmm_regime.parquet",
        base_path / "hmm_regime.parquet",
        PROJECT_ROOT / "data" / "models" / "regime" / f"{sanitize_symbol(symbol)}_hmm_regime.parquet",
        PROJECT_ROOT / "data" / "models" / "regime" / sanitize_symbol(symbol) / "hmm_regime.parquet",
        PROJECT_ROOT / "data" / "models" / "regime" / symbol / "hmm_regime.parquet",
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
        on="timestamp", direction="nearest"
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
    label_threshold = float(args.class_threshold)
    if "class_threshold" in lstm_df.columns:
        lstm_thresh = lstm_df["class_threshold"].dropna()
        if not lstm_thresh.empty:
            label_threshold = float(lstm_thresh.iloc[0])
            logger.info("[%s] Using ARIMA label threshold: %.6f (eval static conf: %.6f)", symbol, label_threshold, float(args.class_threshold))

    valid_mask = np.isfinite(next_returns)
    ret = next_returns[valid_mask]
    lbls = np.full(len(ret), CLASS_NEUTRAL, dtype=np.int64)
    lbls[ret > label_threshold] = CLASS_UP
    lbls[ret < -label_threshold] = CLASS_DOWN
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
    
    requested = []
    if args.symbol:
        requested.append(canonicalize_symbol(args.symbol.strip()))
    if args.symbols:
        requested.extend([canonicalize_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()])
    requested = dedupe_symbols(requested)
    if requested:
        symbols = [s for s in symbols if s in requested]
    
    for symbol in symbols:
        if symbol in FOREX_SYMBOLS: continue
        logger.info("========== Processing XGBoost for %s ==========", symbol)
        
        splits = build_symbol_data(symbol, cnn_dir, lstm_dir, loader, args)
        if not splits: continue
        
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        valid_train = train_df[train_df["y_label"] >= 0]
        valid_val = val_df[val_df["y_label"] >= 0]
        valid_test = test_df[test_df["y_label"] >= 0]
        
        feature_cols = [c for c in train_df.columns if c not in ("timestamp", "split", "y_label", "actual_next_log_return")]
        
        X_train = np.ascontiguousarray(valid_train[feature_cols].to_numpy(dtype=np.float32))
        y_train = np.ascontiguousarray(valid_train["y_label"].to_numpy(dtype=np.int64))
        X_val = np.ascontiguousarray(valid_val[feature_cols].to_numpy(dtype=np.float32))
        y_val = np.ascontiguousarray(valid_val["y_label"].to_numpy(dtype=np.int64))
        X_test = np.ascontiguousarray(valid_test[feature_cols].to_numpy(dtype=np.float32))
        y_test = np.ascontiguousarray(valid_test["y_label"].to_numpy(dtype=np.int64))
        
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

        if len(X_test) > 0:
            probs_test = xgb.predict_proba(X_test)
            y_pred_test = xgb.predict(X_test).astype(np.int64)
            cls_metrics = directional_metrics(y_true=y_test, y_pred=y_pred_test)
            trade_stats = compute_backtest_metrics(
                probs=probs_test,
                actual_next_log_returns=valid_test["actual_next_log_return"].to_numpy(dtype=np.float64),
                threshold=args.class_threshold
            )
            logger.info(
                "[%s] XGBoost Test Acc: %.4f | Sharpe: %.4f | Drawdown: %.4f | WinRate: %.4f",
                symbol,
                cls_metrics["test_accuracy"],
                trade_stats["sharpe"],
                trade_stats["max_drawdown"],
                trade_stats["win_rate"]
            )
        else:
            probs_test = np.empty((0, 3), dtype=np.float32)
            y_pred_test = np.empty((0,), dtype=np.int64)
            cls_metrics = {
                "test_accuracy": 0.0,
                "directional_accuracy": 0.0,
                "up_precision": 0.0,
                "up_recall": 0.0,
                "up_f1": 0.0,
                "up_support": 0,
                "down_precision": 0.0,
                "down_recall": 0.0,
                "down_f1": 0.0,
                "down_support": 0,
                "test_confusion_matrix": [],
            }

        test_pred_df = pd.DataFrame(
            {
                "timestamp": valid_test["timestamp"].to_numpy() if len(valid_test) else np.array([], dtype="datetime64[ns]"),
                "split": "test",
                "xgb_prob_down": probs_test[:, 0] if len(probs_test) else np.array([], dtype=np.float32),
                "xgb_prob_neutral": probs_test[:, 1] if len(probs_test) else np.array([], dtype=np.float32),
                "xgb_prob_up": probs_test[:, 2] if len(probs_test) else np.array([], dtype=np.float32),
                "predicted_label": y_pred_test,
                "actual_label": y_test if len(y_test) else np.array([], dtype=np.int64),
            }
        )
        test_pred_df.to_parquet(sym_out / "predictions.parquet", index=False)

        split_counts = {
            "train": int(len(valid_train)),
            "val": int(len(valid_val)),
            "test": int(len(valid_test)),
        }
        hyperparams = {
            "symbol": symbol,
            "class_threshold": float(args.class_threshold),
            "seed": int(args.seed),
            "features": feature_cols,
            "model": {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "early_stopping_rounds": 30,
            },
        }
        safe_write_json(sym_out / "hyperparams.json", hyperparams)
        safe_write_json(
            sym_out / "training_meta.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "split_counts": split_counts,
                "hyperparameters": hyperparams,
                "metrics": {
                    **cls_metrics,
                    "best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
                },
            },
        )

        logger.info("[%s] XGBoost trained. Best iteration: %s", symbol, getattr(xgb, "best_iteration", 0))

if __name__ == "__main__":
    main()
