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
COST_PER_SIDE = 0.0010  # 10 bps per side (20 bps round trip)
SOURCE_SCRIPT = str(Path(__file__).name)
DEFAULT_FEATURE_SCHEMA_VERSION = "technical_features_v1"
REQUIRED_ARTIFACT_META_FIELDS = (
    "run_id",
    "interval",
    "symbol_canonical",
    "source_script",
    "feature_schema_version",
)


def build_daily_regime_block_mask(frame: pd.DataFrame, mode: str) -> np.ndarray:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized == "off":
        return np.zeros(len(frame), dtype=bool)
    if mode_normalized != "long_only_above_ma200":
        raise ValueError(f"Unsupported daily regime filter mode: {mode}")

    if "close" not in frame.columns or "daily_ma_200" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)

    close = pd.to_numeric(frame["close"], errors="coerce")
    ma200 = pd.to_numeric(frame["daily_ma_200"], errors="coerce")
    return (close.notna() & ma200.notna() & (close < ma200)).to_numpy(dtype=bool)


def apply_daily_regime_filter_to_labels(preds: np.ndarray, block_up_mask: np.ndarray) -> np.ndarray:
    if len(preds) == 0:
        return preds
    adjusted = np.asarray(preds, dtype=np.int64).copy()
    mask = np.asarray(block_up_mask, dtype=bool)
    if len(mask) != len(adjusted):
        return adjusted
    adjusted[(adjusted == CLASS_UP) & mask] = CLASS_NEUTRAL
    return adjusted


def apply_daily_regime_filter_to_probs(probs: np.ndarray, block_up_mask: np.ndarray) -> np.ndarray:
    if len(probs) == 0:
        return probs
    adjusted = np.asarray(probs, dtype=np.float64).copy()
    mask = np.asarray(block_up_mask, dtype=bool)
    if len(mask) != len(adjusted):
        return adjusted
    if adjusted.shape[1] <= CLASS_UP:
        return adjusted
    blocked = np.where(mask)[0]
    if len(blocked) == 0:
        return adjusted

    up_prob = adjusted[blocked, CLASS_UP].copy()
    adjusted[blocked, CLASS_UP] = 0.0
    adjusted[blocked, CLASS_NEUTRAL] = np.clip(adjusted[blocked, CLASS_NEUTRAL] + up_prob, 0.0, 1.0)
    row_sum = adjusted[blocked].sum(axis=1, keepdims=True)
    adjusted[blocked] = np.divide(adjusted[blocked], np.where(row_sum == 0.0, 1.0, row_sum))
    return adjusted

def compute_backtest_metrics(
    probs: np.ndarray,
    actual_next_log_returns: np.ndarray,
    threshold: float | None = None,
    min_hold_bars: int = 0,
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
    annualization_scale = float(np.sqrt(ANNUALIZATION_1H))
    sharpe = float(mean_return / std_return * annualization_scale) if std_return > 1e-12 else 0.0

    downside = net_returns[net_returns < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    sortino = float(mean_return / downside_std * annualization_scale) if downside_std > 1e-12 else 0.0

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
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "total_trades": total_trades,
        "average_trade_return": average_trade_return,
        "coverage": coverage,
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
    parser.add_argument(
        "--strict-artifact-match",
        action="store_true",
        help="Require canonical SYMBOL_NS artifact paths and fail on mixed SYMBOL.NS/SYMBOL_NS collisions.",
    )
    parser.add_argument(
        "--expected-run-id",
        type=str,
        default=None,
        help="Expected upstream run_id from ARIMA/CNN artifacts; strict mode enforces exact match.",
    )
    parser.add_argument(
        "--min-directional-coverage",
        type=float,
        default=0.12,
        help="Minimum base-model directional coverage required before XGBoost training is allowed for a symbol.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional output run identifier; defaults to expected-run-id or generated timestamp.",
    )
    parser.add_argument(
        "--feature-schema-version",
        type=str,
        default=DEFAULT_FEATURE_SCHEMA_VERSION,
        help="Feature schema version tag persisted in XGBoost artifacts.",
    )
    parser.add_argument(
        "--daily-regime-filter-mode",
        choices=["off", "long_only_above_ma200"],
        default="off",
        help="Optional regime filter applied to UP predictions before metrics/backtest.",
    )
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


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_symbol_artifact_dir(base_dir: Path, symbol: str, strict: bool) -> Path | None:
    canonical_dir = base_dir / sanitize_symbol(symbol)
    legacy_dir = base_dir / symbol
    canonical_exists = canonical_dir.exists()
    legacy_exists = legacy_dir.exists() and legacy_dir.resolve() != canonical_dir.resolve()

    if strict and canonical_exists and legacy_exists:
        raise ValueError(
            f"Mixed artifact collision for {symbol}: both {canonical_dir} and {legacy_dir} exist."
        )

    if strict:
        if not canonical_exists:
            raise FileNotFoundError(f"Missing canonical artifact directory for {symbol}: {canonical_dir}")
        return canonical_dir

    if canonical_exists:
        return canonical_dir
    if legacy_exists:
        return legacy_dir
    return None


def validate_artifact_contract(
    *,
    symbol: str,
    meta: dict[str, Any],
    strict: bool,
    expected_interval: str,
    expected_run_id: str | None,
) -> tuple[bool, str]:
    missing = [field for field in REQUIRED_ARTIFACT_META_FIELDS if field not in meta]
    if missing:
        return False, f"missing metadata fields: {missing}"

    symbol_canonical = canonicalize_symbol(str(meta.get("symbol_canonical", "")).strip())
    if symbol_canonical != canonicalize_symbol(symbol):
        return False, f"symbol_canonical mismatch: {symbol_canonical} != {symbol}"

    artifact_interval = str(meta.get("interval", "")).strip()
    if artifact_interval != str(expected_interval).strip():
        return False, f"interval mismatch: {artifact_interval} != {expected_interval}"

    artifact_run_id = str(meta.get("run_id", "")).strip()
    if expected_run_id and artifact_run_id != expected_run_id:
        return False, f"run_id mismatch: {artifact_run_id} != {expected_run_id}"

    if strict and not artifact_run_id:
        return False, "run_id is empty under strict artifact match mode"

    return True, ""


def tune_confidence_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    min_directional_coverage: float,
    regime_block_up_mask: np.ndarray | None = None,
) -> tuple[float, float, float, dict[str, Any]]:
    best_threshold = 0.35
    best_score = float("-inf")
    best_coverage = 0.0
    best_metrics: dict[str, Any] = {
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
    for threshold in np.arange(0.15, 0.85, 0.05):
        preds = apply_confidence_threshold(probs_val, threshold=float(threshold))
        if regime_block_up_mask is not None:
            preds = apply_daily_regime_filter_to_labels(preds, np.asarray(regime_block_up_mask, dtype=bool))
        cls_metrics = directional_metrics(y_true=y_val, y_pred=preds)
        directional_coverage = float(np.mean(np.isin(preds, [CLASS_UP, CLASS_DOWN])))
        directional_recall = float((cls_metrics["up_recall"] + cls_metrics["down_recall"]) / 2.0)
        shortfall = max(0.0, float(min_directional_coverage) - directional_coverage)
        score = (
            (0.65 * float(cls_metrics["directional_accuracy"]))
            + (0.25 * directional_recall)
            + (0.10 * directional_coverage)
            - (1.5 * shortfall)
        )
        if score > best_score or (np.isclose(score, best_score) and directional_coverage > best_coverage):
            best_score = score
            best_threshold = float(threshold)
            best_coverage = directional_coverage
            best_metrics = cls_metrics
    return best_threshold, best_score, best_coverage, best_metrics


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


def apply_confidence_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    if len(probs) == 0:
        return np.empty((0,), dtype=np.int64)
    preds = np.argmax(probs, axis=1).astype(np.int64)
    confidence = np.max(probs, axis=1)
    preds[confidence < float(threshold)] = CLASS_NEUTRAL
    return preds

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

def build_symbol_data(symbol: str, cnn_dir: Path, lstm_dir: Path, loader: DataLoader, args) -> dict[str, Any] | None:
    """
    Loads CNN and LSTM predictions, engineers technical features, 
    and merges them into a single dataset.
    """
    try:
        cnn_symbol_dir = resolve_symbol_artifact_dir(cnn_dir, symbol, strict=bool(args.strict_artifact_match))
        lstm_symbol_dir = resolve_symbol_artifact_dir(lstm_dir, symbol, strict=bool(args.strict_artifact_match))
    except (ValueError, FileNotFoundError) as exc:
        if bool(args.strict_artifact_match):
            raise
        logger.warning("[%s] Artifact resolution failed: %s", symbol, exc)
        return None

    if cnn_symbol_dir is None:
        logger.warning("[%s] Missing CNN artifact directory under %s", symbol, cnn_dir)
        return None
    if lstm_symbol_dir is None:
        logger.warning("[%s] Missing ARIMA-LSTM artifact directory under %s", symbol, lstm_dir)
        return None

    cnn_path = cnn_symbol_dir / "predictions.parquet"
    lstm_path = lstm_symbol_dir / "predictions.parquet"
    cnn_meta_path = cnn_symbol_dir / "training_meta.json"
    lstm_meta_path = lstm_symbol_dir / "training_meta.json"

    if not cnn_path.exists():
        logger.warning("[%s] Missing CNN predictions: %s", symbol, cnn_path)
        return None
    if not lstm_path.exists():
        logger.warning("[%s] Missing LSTM predictions: %s", symbol, lstm_path)
        return None
    if not cnn_meta_path.exists():
        logger.warning("[%s] Missing CNN metadata: %s", symbol, cnn_meta_path)
        return None
    if not lstm_meta_path.exists():
        logger.warning("[%s] Missing ARIMA-LSTM metadata: %s", symbol, lstm_meta_path)
        return None

    cnn_meta = load_json(cnn_meta_path)
    lstm_meta = load_json(lstm_meta_path)
    cnn_ok, cnn_reason = validate_artifact_contract(
        symbol=symbol,
        meta=cnn_meta,
        strict=bool(args.strict_artifact_match),
        expected_interval=args.interval,
        expected_run_id=args.expected_run_id,
    )
    lstm_ok, lstm_reason = validate_artifact_contract(
        symbol=symbol,
        meta=lstm_meta,
        strict=bool(args.strict_artifact_match),
        expected_interval=args.interval,
        expected_run_id=args.expected_run_id,
    )
    if not cnn_ok:
        if bool(args.strict_artifact_match):
            raise ValueError(f"[{symbol}] CNN artifact contract rejected: {cnn_reason}")
        logger.warning("[%s] CNN artifact contract rejected: %s", symbol, cnn_reason)
        return None
    if not lstm_ok:
        if bool(args.strict_artifact_match):
            raise ValueError(f"[{symbol}] LSTM artifact contract rejected: {lstm_reason}")
        logger.warning("[%s] LSTM artifact contract rejected: %s", symbol, lstm_reason)
        return None

    cnn_run_id = str(cnn_meta.get("run_id", "")).strip()
    lstm_run_id = str(lstm_meta.get("run_id", "")).strip()
    if cnn_run_id != lstm_run_id:
        message = f"Artifact run_id mismatch: cnn={cnn_run_id} vs lstm={lstm_run_id}"
        if bool(args.strict_artifact_match):
            raise ValueError(f"[{symbol}] {message}")
        logger.warning("[%s] %s. Symbol skipped.", symbol, message)
        return None

    if args.expected_run_id and cnn_run_id != str(args.expected_run_id).strip():
        message = f"Expected run_id={args.expected_run_id} but artifacts have run_id={cnn_run_id}"
        if bool(args.strict_artifact_match):
            raise ValueError(f"[{symbol}] {message}")
        logger.warning("[%s] %s. Symbol skipped.", symbol, message)
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

    train_split = final_df[final_df["split"] == "train"].copy()
    if len(train_split) == 0:
        logger.warning("[%s] Empty aligned train split after merge; symbol skipped.", symbol)
        return None

    cnn_train_argmax = np.argmax(
        train_split[["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    lstm_train_argmax = np.argmax(
        train_split[["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    cnn_directional_coverage = float(np.mean(np.isin(cnn_train_argmax, [CLASS_DOWN, CLASS_UP])))
    lstm_directional_coverage = float(np.mean(np.isin(lstm_train_argmax, [CLASS_DOWN, CLASS_UP])))
    base_directional_coverage = float((cnn_directional_coverage + lstm_directional_coverage) / 2.0)

    return {
        "train": train_split,
        "val": final_df[final_df["split"] == "val"].copy(),
        "test": final_df[final_df["split"] == "test"].copy(),
        "artifact_contract": {
            "run_id": cnn_run_id,
            "interval": str(cnn_meta.get("interval", "")),
            "feature_schema_version": str(cnn_meta.get("feature_schema_version", "")),
            "cnn_source_script": str(cnn_meta.get("source_script", "")),
            "lstm_source_script": str(lstm_meta.get("source_script", "")),
        },
        "readiness": {
            "cnn_directional_coverage": cnn_directional_coverage,
            "lstm_directional_coverage": lstm_directional_coverage,
            "base_directional_coverage": base_directional_coverage,
        },
    }

def main():
    args = parse_args()
    np.random.seed(args.seed)
    run_id = str(args.run_id).strip() if args.run_id else ""
    expected_run_id = str(args.expected_run_id).strip() if args.expected_run_id else ""
    if not run_id:
        run_id = expected_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_schema_version = str(args.feature_schema_version).strip()
    
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    
    cnn_dir = Path(args.cnn_dir)
    lstm_dir = Path(args.lstm_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "XGBoost run_id=%s | interval=%s | strict_artifact_match=%s | expected_run_id=%s",
        run_id,
        args.interval,
        bool(args.strict_artifact_match),
        expected_run_id or "none",
    )
    
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

    run_report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "expected_run_id": expected_run_id or None,
        "interval": args.interval,
        "source_script": SOURCE_SCRIPT,
        "feature_schema_version": feature_schema_version,
        "strict_artifact_match": bool(args.strict_artifact_match),
        "daily_regime_filter_mode": str(args.daily_regime_filter_mode).strip().lower(),
        "symbols_requested": symbols,
        "symbols_processed": [],
        "symbols_skipped": [],
    }
    
    for symbol in symbols:
        if symbol in FOREX_SYMBOLS: continue
        logger.info("========== Processing XGBoost for %s ==========", symbol)
        
        splits = build_symbol_data(symbol, cnn_dir, lstm_dir, loader, args)
        if not splits:
            run_report["symbols_skipped"].append(
                {"symbol": symbol, "reason": "artifact_or_alignment_validation_failed"}
            )
            continue
        
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        readiness = dict(splits.get("readiness", {}))
        artifact_contract = dict(splits.get("artifact_contract", {}))
        base_directional_coverage = float(readiness.get("base_directional_coverage", 0.0))
        if base_directional_coverage < float(args.min_directional_coverage):
            reason = (
                f"base directional coverage {base_directional_coverage:.4f} below "
                f"min-directional-coverage={float(args.min_directional_coverage):.4f}"
            )
            logger.warning("[%s] %s. Skipping XGBoost training.", symbol, reason)
            run_report["symbols_skipped"].append({"symbol": symbol, "reason": reason, "readiness": readiness})
            continue
        valid_train = train_df[train_df["y_label"] >= 0]
        valid_val = val_df[val_df["y_label"] >= 0]
        valid_test = test_df[test_df["y_label"] >= 0]
        if min(len(valid_train), len(valid_val), len(valid_test)) == 0:
            reason = (
                f"empty aligned split(s): train={len(valid_train)}, val={len(valid_val)}, test={len(valid_test)}"
            )
            logger.warning("[%s] %s", symbol, reason)
            run_report["symbols_skipped"].append({"symbol": symbol, "reason": reason})
            continue
        
        feature_cols = [
            c for c in train_df.columns 
            if c not in (
                "timestamp", "split", "y_label", "actual_next_log_return", 
                "actual_return", "class_threshold", "predicted_label", "lstm_final_score"
            )
        ]
        
        X_train = np.ascontiguousarray(valid_train[feature_cols].to_numpy(dtype=np.float32))
        y_train = np.ascontiguousarray(valid_train["y_label"].to_numpy(dtype=np.int64))
        X_val = np.ascontiguousarray(valid_val[feature_cols].to_numpy(dtype=np.float32))
        y_val = np.ascontiguousarray(valid_val["y_label"].to_numpy(dtype=np.int64))
        X_test = np.ascontiguousarray(valid_test[feature_cols].to_numpy(dtype=np.float32))
        y_test = np.ascontiguousarray(valid_test["y_label"].to_numpy(dtype=np.int64))
        
        # Compute balanced weights, then boost directional classes to fight neutral bias
        base_sw = compute_sample_weight(class_weight="balanced", y=y_train)
        sw = np.copy(base_sw)
        sw[y_train == CLASS_UP] *= 3.0
        sw[y_train == CLASS_DOWN] *= 2.0
        sample_weight = np.ascontiguousarray(sw.astype(np.float32))
        
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
            probs_val = xgb.predict_proba(X_val)
            probs_test = xgb.predict_proba(X_test)
            regime_filter_mode = str(args.daily_regime_filter_mode).strip().lower()
            val_block_up_mask = build_daily_regime_block_mask(valid_val, regime_filter_mode)
            test_block_up_mask = build_daily_regime_block_mask(valid_test, regime_filter_mode)
            probs_val_filtered = apply_daily_regime_filter_to_probs(probs_val, val_block_up_mask)
            probs_test_filtered = apply_daily_regime_filter_to_probs(probs_test, test_block_up_mask)
            best_conf_threshold, threshold_score, val_directional_coverage, val_cls_metrics = tune_confidence_threshold(
                y_val=y_val,
                probs_val=probs_val_filtered,
                min_directional_coverage=float(args.min_directional_coverage),
                regime_block_up_mask=val_block_up_mask,
            )
            y_pred_test = apply_confidence_threshold(probs_test_filtered, threshold=best_conf_threshold)
            y_pred_test = apply_daily_regime_filter_to_labels(y_pred_test, test_block_up_mask)
            cls_metrics = directional_metrics(y_true=y_test, y_pred=y_pred_test)
            test_directional_coverage = float(np.mean(np.isin(y_pred_test, [CLASS_UP, CLASS_DOWN]))) if len(y_pred_test) else 0.0
            trade_stats = compute_backtest_metrics(
                probs=probs_test_filtered,
                actual_next_log_returns=valid_test["actual_next_log_return"].to_numpy(dtype=np.float64),
                threshold=best_conf_threshold,
                min_hold_bars=0,
            )
            regime_filter_stats = {
                "mode": regime_filter_mode,
                "val_up_blocked_rows": int(np.sum(val_block_up_mask)),
                "test_up_blocked_rows": int(np.sum(test_block_up_mask)),
            }

            gate_failed = False
            gate_reasons = []
            if cls_metrics["up_recall"] < 0.15 or cls_metrics["down_recall"] < 0.15:
                gate_failed = True
                gate_reasons.append("Low directional recall (< 15%)")
            if cls_metrics["directional_accuracy"] <= 0.40:
                gate_failed = True
                gate_reasons.append("Directional accuracy <= 40%")
            if test_directional_coverage < float(args.min_directional_coverage):
                gate_failed = True
                gate_reasons.append(
                    f"Directional coverage {test_directional_coverage:.3f} < min {float(args.min_directional_coverage):.3f}"
                )
            if getattr(xgb, "best_iteration", 0) < 20:
                gate_failed = True
                gate_reasons.append("Early stopping too early (best_iteration < 20)")

            gate_status = "FAILED" if gate_failed else "PASSED"
            logger.info(
                "[%s] XGBoost Test Acc: %.4f | Sharpe: %.4f | Drawdown: %.4f | WinRate: %.4f | Coverage: %.2f%% | Gates: %s",
                symbol,
                cls_metrics["test_accuracy"],
                trade_stats["sharpe"],
                trade_stats["max_drawdown"],
                trade_stats["win_rate"],
                test_directional_coverage * 100.0,
                gate_status,
            )
            if gate_failed:
                logger.warning("[%s] Gate failure reasons: %s", symbol, ", ".join(gate_reasons))
        else:
            probs_val = np.empty((0, 3), dtype=np.float32)
            probs_test = np.empty((0, 3), dtype=np.float32)
            y_pred_test = np.empty((0,), dtype=np.int64)
            gate_failed = True
            gate_reasons = ["No test data"]
            best_conf_threshold = float(args.class_threshold)
            threshold_score = 0.0
            val_directional_coverage = 0.0
            test_directional_coverage = 0.0
            regime_filter_stats = {
                "mode": str(args.daily_regime_filter_mode).strip().lower(),
                "val_up_blocked_rows": 0,
                "test_up_blocked_rows": 0,
            }
            val_cls_metrics = {
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
            cls_metrics = dict(val_cls_metrics)
            trade_stats = {
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "average_trade_return": 0.0,
                "coverage": 0.0,
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
                "confidence_threshold": best_conf_threshold,
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
            "confidence_threshold": float(best_conf_threshold),
            "seed": int(args.seed),
            "strict_artifact_match": bool(args.strict_artifact_match),
            "expected_run_id": expected_run_id or None,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": SOURCE_SCRIPT,
            "feature_schema_version": feature_schema_version,
            "min_directional_coverage": float(args.min_directional_coverage),
            "daily_regime_filter_mode": str(args.daily_regime_filter_mode).strip().lower(),
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
        evaluation_bundle = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "symbol_canonical": symbol,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": SOURCE_SCRIPT,
            "feature_schema_version": feature_schema_version,
            "artifact_contract": artifact_contract,
            "readiness": readiness,
            "split_counts": split_counts,
            "classification": {
                "validation": {
                    **val_cls_metrics,
                    "directional_coverage": val_directional_coverage,
                    "threshold_score": float(threshold_score),
                },
                "test": {
                    **cls_metrics,
                    "directional_coverage": test_directional_coverage,
                },
            },
            "backtest": trade_stats,
            "gates": {
                "passed": not gate_failed,
                "status": "PASSED" if not gate_failed else "FAILED",
                "reasons": gate_reasons,
                "best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
                "confidence_threshold": float(best_conf_threshold),
                "min_directional_coverage": float(args.min_directional_coverage),
            },
            "daily_regime_filter": regime_filter_stats,
        }
        safe_write_json(sym_out / "evaluation_bundle.json", evaluation_bundle)
        safe_write_json(
            sym_out / "training_meta.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "symbol_canonical": symbol,
                "run_id": run_id,
                "interval": args.interval,
                "source_script": SOURCE_SCRIPT,
                "feature_schema_version": feature_schema_version,
                "split_counts": split_counts,
                "hyperparameters": hyperparams,
                "metrics": {
                    **cls_metrics,
                    "best_iteration": int(getattr(xgb, "best_iteration", 0) or 0),
                    "gate_passed": not gate_failed,
                    "gate_reasons": gate_reasons,
                    "confidence_threshold": float(best_conf_threshold),
                    "directional_coverage": test_directional_coverage,
                    "backtest": trade_stats,
                    "daily_regime_filter": regime_filter_stats,
                },
            },
        )

        run_report["symbols_processed"].append(
            {
                "symbol": symbol,
                "run_id": run_id,
                "artifact_run_id": artifact_contract.get("run_id"),
                "confidence_threshold": float(best_conf_threshold),
                "gate_passed": not gate_failed,
                "gate_reasons": gate_reasons,
                "directional_coverage": test_directional_coverage,
                "backtest": trade_stats,
                "readiness": readiness,
                "daily_regime_filter": regime_filter_stats,
            }
        )

        logger.info("[%s] XGBoost trained. Best iteration: %s", symbol, getattr(xgb, "best_iteration", 0))

    run_report_path = output_root / f"xgboost_run_report_{run_id}.json"
    safe_write_json(run_report_path, run_report)
    logger.info("Saved XGBoost run report: %s", run_report_path)

if __name__ == "__main__":
    main()
