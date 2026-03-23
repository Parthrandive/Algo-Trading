from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.symbols import (  # noqa: E402
    EQUITY_SYMBOLS,
    FOREX_SYMBOLS,
    FX_RESULTS_NOTE,
    MIN_ROWS,
    dedupe_symbols,
)
from scripts.train_regime_aware import (  # noqa: E402
    DirectionCNNClassifier,
    ResidualLSTMRegressor,
    add_volatility_regime_features,
    apply_labels,
    build_cnn_sequences,
    build_feature_table,
    build_regression_sequences,
    impute_and_scale_features,
    load_symbol_frame,
    make_loader,
    sanitize_symbol,
    tensor_predict,
)


PRIMARY_FREQ = "1h"
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2
CLASS_ORDER = ["down", "neutral", "up"]
PROB_COLS = ["prob_down", "prob_neutral", "prob_up"]
ANNUALIZATION_1H = math.sqrt(252.0 * 6.5)
COST_PER_SIDE = 0.001


@dataclass
class SkipEntry:
    reason: str
    fix: str


@dataclass
class ValidArtifact:
    symbol: str
    base_path: Path
    summary_path: Path
    cnn_path: Path
    lstm_path: Path
    scaler_path: Path
    summary: dict[str, Any]
    frequency: str
    row_count: int


@dataclass
class SymbolResult:
    symbol: str
    xgb_val_bal: float
    xgb_test_bal: float
    ens_val_bal: float
    ens_test_bal: float
    sharpe: float
    drawdown: float
    win_rate: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-symbol XGBoost stacking meta-layer on saved CNN + ARIMA-LSTM artifacts."
    )
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Regime-aware run directory containing symbol artifacts.")
    parser.add_argument("--database-url", type=str, default=None, help="Optional override for data loader.")
    parser.add_argument("--gold-dir", type=str, default="data/gold", help="Gold parquet fallback directory.")
    parser.add_argument("--interval", type=str, default=PRIMARY_FREQ, help="Expected artifact/data interval.")
    parser.add_argument("--limit", type=int, default=4000, help="Rows to load per symbol when rebuilding features.")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS, help="Minimum rows gate.")
    parser.add_argument("--fx-context-symbol", type=str, default=FOREX_SYMBOLS[0], help="FX context symbol (external feature only).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def format_bytes(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def infer_frequency(summary: dict[str, Any], manifest: dict[str, Any] | None) -> str:
    candidates = [
        summary.get("primary_freq"),
        summary.get("frequency"),
        summary.get("interval"),
        summary.get("config", {}).get("interval") if isinstance(summary.get("config"), dict) else None,
        manifest.get("config", {}).get("interval") if manifest else None,
    ]
    for candidate in candidates:
        text = str(candidate).strip() if candidate is not None else ""
        if text:
            return text
    return ""


def infer_row_count(summary: dict[str, Any]) -> int:
    for key in ("rows", "rows_after_feature_engineering", "raw_rows_loaded", "train_rows"):
        value = summary.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def warn_skip(symbol: str, reason: str, fix: str) -> None:
    print(f"WARNING: [{symbol}] skipped for XGBoost")
    print(f"Reason: {reason}")
    print(f"Fix: {fix}")


def resolve_artifacts_dir(explicit_path: str | None, expected_interval: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    reports_root = PROJECT_ROOT / "data" / "reports"
    if not reports_root.exists():
        raise FileNotFoundError("data/reports directory not found. Provide --artifacts-dir.")

    best_item: tuple[pd.Timestamp, Path] | None = None
    for manifest_path in reports_root.rglob("run_manifest.json"):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        interval = str(manifest.get("config", {}).get("interval", "")).strip()
        run_dir_raw = manifest.get("run_dir")
        if not run_dir_raw:
            continue
        run_dir = Path(run_dir_raw)
        if not run_dir.is_absolute():
            run_dir = (PROJECT_ROOT / run_dir).resolve()
        if interval != expected_interval:
            continue
        ts = parse_timestamp(manifest.get("timestamp_utc"))
        if pd.isna(ts):
            ts = pd.to_datetime(datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc))
        if best_item is None or ts > best_item[0]:
            best_item = (ts, run_dir)

    if best_item is None:
        raise FileNotFoundError(
            f"No run_manifest.json with interval={expected_interval} found under data/reports. "
            "Provide --artifacts-dir explicitly."
        )
    return best_item[1]


def resolve_candidate_symbols(artifacts_dir: Path) -> list[str]:
    configured = dedupe_symbols(EQUITY_SYMBOLS)
    discovered: list[str] = []
    for summary_path in sorted(artifacts_dir.glob("*/summary.json")):
        try:
            payload = load_json(summary_path)
        except Exception:
            continue
        symbol = str(payload.get("symbol", "")).strip()
        if symbol and symbol not in FOREX_SYMBOLS:
            discovered.append(symbol)
    discovered = dedupe_symbols(discovered)

    # Keep explicit config symbols first, then include newly discovered symbols automatically.
    merged = dedupe_symbols([*configured, *discovered])
    if not configured and discovered:
        print("EQUITY_SYMBOLS is empty; using artifact-discovered equity symbols for this run.")
    return merged


def validate_symbol_artifact(
    symbol: str,
    artifacts_dir: Path,
    manifest: dict[str, Any] | None,
    min_rows: int,
    expected_freq: str,
) -> tuple[ValidArtifact | None, SkipEntry | None]:
    try:
        assert symbol not in FOREX_SYMBOLS, "forex symbol is external feature only"
    except AssertionError:
        return None, SkipEntry(
            reason=f"Check 6 failed: {symbol} is forex (external feature only)",
            fix=f"remove {symbol} from target symbols; keep it only in FOREX_SYMBOLS context features",
        )

    base_path = artifacts_dir / sanitize_symbol(symbol)
    if not base_path.exists():
        return None, SkipEntry(
            reason=f"symbol artifact folder missing: {base_path}",
            fix=f"run regime-aware training for {symbol} so artifacts are generated under {base_path}",
        )

    cnn_expected = base_path / "cnn_pattern" / "model.keras"
    cnn_path = first_existing(
        [
            cnn_expected,
            base_path / "cnn_pattern" / "best_cnn_checkpoint.pt",
            base_path / "cnn_pattern" / "model.pt",
        ]
    )
    if cnn_path is None:
        return None, SkipEntry(
            reason=f"Check 1 failed: CNN artifact missing at {cnn_expected}",
            fix=f"train/export CNN artifact for {symbol} under {base_path / 'cnn_pattern'}",
        )

    lstm_expected = base_path / "arima_lstm" / "model"
    lstm_path = first_existing(
        [
            lstm_expected,
            base_path / "arima_lstm" / "best_lstm_checkpoint.pt",
            base_path / "arima_lstm" / "model.pt",
        ]
    )
    if lstm_path is None:
        return None, SkipEntry(
            reason=f"Check 2 failed: ARIMA-LSTM artifact missing at {lstm_expected}",
            fix=f"train/export ARIMA-LSTM artifact for {symbol} under {base_path / 'arima_lstm'}",
        )

    scaler_expected = base_path / "scaler.pkl"
    scaler_path = first_existing(
        [
            scaler_expected,
            base_path / "arima_lstm" / "feature_scaler.pkl",
        ]
    )
    if scaler_path is None:
        return None, SkipEntry(
            reason=f"Check 3 failed: scaler artifact missing at {scaler_expected}",
            fix=f"save fitted scaler for {symbol} to {scaler_expected} (or feature_scaler.pkl fallback)",
        )

    summary_path = base_path / "summary.json"
    if not summary_path.exists():
        return None, SkipEntry(
            reason="Check 4 failed: summary metadata missing (summary.json not found)",
            fix=f"write summary.json for {symbol} including interval/frequency and row count metadata",
        )
    summary = load_json(summary_path)

    frequency = infer_frequency(summary, manifest)
    if str(frequency).strip() != expected_freq:
        return None, SkipEntry(
            reason=f"Check 4 failed: artifact frequency={frequency!r} (expected {expected_freq})",
            fix=f"backfill hourly data and retrain {symbol} on {expected_freq} artifacts only",
        )

    row_count = infer_row_count(summary)
    if row_count < min_rows:
        return None, SkipEntry(
            reason=f"Check 5 failed: row_count={row_count} < MIN_ROWS={min_rows}",
            fix=f"backfill more {expected_freq} data for {symbol} and retrain so row_count >= {min_rows}",
        )

    return (
        ValidArtifact(
            symbol=symbol,
            base_path=base_path,
            summary_path=summary_path,
            cnn_path=cnn_path,
            lstm_path=lstm_path,
            scaler_path=scaler_path,
            summary=summary,
            frequency=frequency,
            row_count=row_count,
        ),
        None,
    )


def internal_label_to_user(labels_internal: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_internal, dtype=np.int64)
    out = np.full(labels.shape, -1, dtype=np.int64)
    out[labels == 2] = CLASS_DOWN
    out[labels == 1] = CLASS_NEUTRAL
    out[labels == 0] = CLASS_UP
    return out


def internal_probs_to_user(probs_internal: np.ndarray) -> np.ndarray:
    # Internal order: up, neutral, down. User order: down, neutral, up.
    probs = np.asarray(probs_internal, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected (n,3) internal probability matrix, got {probs.shape}")
    return probs[:, [2, 1, 0]]


def scores_to_soft_probs(scores: np.ndarray, threshold: float, scale_factor: float = 2.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    effective_thresh = max(float(threshold), 1e-6)
    scale = scale_factor / effective_thresh
    
    logits_down = (-scores - effective_thresh) * scale
    logits_neutral = (effective_thresh - np.abs(scores)) * scale
    logits_up = (scores - effective_thresh) * scale
    
    logits = np.column_stack([logits_down, logits_neutral, logits_up])
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def validate_prob_rows(probs: np.ndarray, tolerance: float = 0.001) -> None:
    row_sums = np.asarray(probs, dtype=np.float64).sum(axis=1)
    if not np.all(np.isfinite(row_sums)):
        raise ValueError("Probability rows contain non-finite sums.")
    if not np.all(np.abs(row_sums - 1.0) <= tolerance):
        raise ValueError("Probability rows do not sum to 1.0 within tolerance.")


def split_series(n_rows: int, train_end: int, val_end: int) -> np.ndarray:
    split = np.empty(n_rows, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    return split


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def compute_arima_baseline_from_loaded_model(arima_model: Any, log_returns: np.ndarray, train_end: int) -> np.ndarray:
    values = np.asarray(log_returns, dtype=np.float64).reshape(-1)
    baseline = np.full(values.shape, np.nan, dtype=np.float64)

    rolling_model = arima_model
    if hasattr(rolling_model, "predict_in_sample"):
        train_pred = np.asarray(rolling_model.predict_in_sample(), dtype=np.float64).reshape(-1)
        baseline[: min(len(train_pred), train_end)] = train_pred[:train_end]
    elif hasattr(rolling_model, "predict"):
        try:
            train_pred = np.asarray(rolling_model.predict(start=0, end=train_end - 1), dtype=np.float64).reshape(-1)
            baseline[: min(len(train_pred), train_end)] = train_pred[:train_end]
        except Exception:
            pass

    for idx in range(train_end, len(values)):
        next_value = float(values[idx])
        if hasattr(rolling_model, "predict") and hasattr(rolling_model, "update"):
            forecast = np.asarray(rolling_model.predict(n_periods=1), dtype=np.float64).reshape(-1)
            baseline[idx] = float(forecast[0])
            rolling_model.update(next_value)
            continue
        if hasattr(rolling_model, "forecast"):
            forecast = np.asarray(rolling_model.forecast(steps=1), dtype=np.float64).reshape(-1)
            baseline[idx] = float(forecast[0])
            if hasattr(rolling_model, "append"):
                try:
                    rolling_model = rolling_model.append([next_value], refit=False)
                except Exception:
                    pass
            continue
        if idx > 0 and np.isfinite(baseline[idx - 1]):
            baseline[idx] = baseline[idx - 1]
        else:
            baseline[idx] = 0.0

    baseline = (
        pd.Series(baseline)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    return baseline


def predict_split_outputs(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    y_dummy = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, y_dummy, batch_size=batch_size, shuffle=False)
    preds, probs = tensor_predict(model, loader, device)
    return preds, probs


def infer_threshold(summary: dict[str, Any]) -> float:
    symbol_cfg = summary.get("symbol_config", {})
    cls_metrics = summary.get("classification_metrics", {})
    threshold = symbol_cfg.get("threshold", cls_metrics.get("neutral_threshold", 0.0))
    return float(threshold)


def infer_split_indices(summary: dict[str, Any], n_rows: int) -> tuple[int, int]:
    reg = summary.get("regression_metrics", {})
    train_end = reg.get("train_split_index", summary.get("train_rows"))
    val_end = reg.get("val_split_index")
    if val_end is None and train_end is not None:
        val_rows = summary.get("val_rows")
        if val_rows is not None:
            val_end = int(train_end) + int(val_rows)
    if train_end is None:
        train_end = int(n_rows * 0.70)
    if val_end is None:
        val_end = int(n_rows * 0.85)
    train_end = int(train_end)
    val_end = int(val_end)
    if train_end <= 0 or train_end >= n_rows:
        raise ValueError(f"Invalid TRAIN_END={train_end} for n_rows={n_rows}")
    if val_end <= train_end or val_end >= n_rows:
        raise ValueError(f"Invalid VAL_END={val_end} for n_rows={n_rows}")
    return train_end, val_end


def maybe_load_regime_labels(symbol: str, base_path: Path, frame: pd.DataFrame, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    split = frame["split"].to_numpy()
    default_regime = np.full(len(frame), -1, dtype=np.int64)
    candidate_paths = [
        base_path / "regime_agent" / "hmm_regime.parquet",
        base_path / "regime_agent" / "hmm_regime.csv",
        base_path / "hmm_regime.parquet",
        base_path / "hmm_regime.csv",
        PROJECT_ROOT / "data" / "models" / "hmm_regime" / f"{sanitize_symbol(symbol)}_hmm_regime.parquet",
        PROJECT_ROOT / "data" / "models" / "hmm_regime" / f"{sanitize_symbol(symbol)}_hmm_regime.csv",
    ]
    regime_path = first_existing(candidate_paths)
    if regime_path is None:
        print(f"WARNING: [{symbol}] Regime artifact missing — setting hmm_regime=-1")
        shift_flag = np.zeros(len(frame), dtype=np.int64)
        return default_regime, shift_flag

    try:
        if regime_path.suffix.lower() == ".parquet":
            regime_df = pd.read_parquet(regime_path)
        else:
            regime_df = pd.read_csv(regime_path)
    except Exception as exc:
        print(f"WARNING: [{symbol}] Failed loading regime artifact ({regime_path}): {exc}. Using hmm_regime=-1")
        shift_flag = np.zeros(len(frame), dtype=np.int64)
        return default_regime, shift_flag

    ts_col = None
    for candidate in ("timestamp", "ts", "datetime", "as_of"):
        if candidate in regime_df.columns:
            ts_col = candidate
            break
    label_col = None
    for candidate in ("hmm_regime", "regime", "regime_label", "state", "label"):
        if candidate in regime_df.columns:
            label_col = candidate
            break
    if ts_col is None or label_col is None:
        print(f"WARNING: [{symbol}] Regime artifact schema invalid at {regime_path}. Using hmm_regime=-1")
        shift_flag = np.zeros(len(frame), dtype=np.int64)
        return default_regime, shift_flag

    regime_data = regime_df[[ts_col, label_col]].copy()
    regime_data[ts_col] = pd.to_datetime(regime_data[ts_col], utc=True, errors="coerce")
    regime_data[label_col] = pd.to_numeric(regime_data[label_col], errors="coerce")
    regime_data = regime_data.dropna(subset=[ts_col, label_col]).sort_values(ts_col)
    if regime_data.empty:
        print(f"WARNING: [{symbol}] Regime artifact has no usable rows at {regime_path}. Using hmm_regime=-1")
        shift_flag = np.zeros(len(frame), dtype=np.int64)
        return default_regime, shift_flag

    merged = pd.merge_asof(
        frame[["timestamp"]].sort_values("timestamp"),
        regime_data.rename(columns={ts_col: "timestamp", label_col: "hmm_regime"}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    hmm_regime = (
        pd.to_numeric(merged["hmm_regime"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    train_mask = split == "train"
    train_regimes = hmm_regime[train_mask]
    valid_train_regimes = train_regimes[train_regimes >= 0]
    if len(valid_train_regimes) == 0:
        dominant = -1
    else:
        dominant = int(pd.Series(valid_train_regimes).mode().iloc[0])
    if dominant < 0:
        regime_shift_flag = np.zeros(len(frame), dtype=np.int64)
    else:
        regime_shift_flag = (hmm_regime != dominant).astype(np.int64)
    return hmm_regime, regime_shift_flag


def apply_confidence_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = np.argmax(probs, axis=1).astype(np.int64)
    confidence = np.max(probs, axis=1)
    preds[confidence < threshold] = CLASS_NEUTRAL
    return preds


def generate_weight_grid(min_weight: float = 0.1, max_weight: float = 0.7, step: float = 0.1) -> list[tuple[float, float, float]]:
    grid = np.round(np.arange(min_weight, max_weight + 1e-9, step), 2)
    combos: list[tuple[float, float, float]] = []
    for w_cnn in grid:
        for w_lstm in grid:
            w_xgb = round(1.0 - float(w_cnn) - float(w_lstm), 2)
            if w_xgb < min_weight or w_xgb > max_weight:
                continue
            if abs((w_cnn + w_lstm + w_xgb) - 1.0) > 1e-6:
                continue
            combos.append((float(w_cnn), float(w_lstm), float(w_xgb)))
    return combos


def compute_backtest_metrics(
    probs: np.ndarray,
    actual_next_log_returns: np.ndarray,
    threshold: float | None = None,
    min_hold_bars: int = 3,
) -> dict[str, float]:
    probabilities = np.asarray(probs, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"Expected probs shape (n,3), got {probabilities.shape}")

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

        # Fix 1: confidence threshold is checked before direction assignment.
        if raw_confidence < confidence_threshold:
            desired_direction = 0.0
            desired_size = 0.0
        else:
            # Fix 2: confidence-scaled position size in [0, 1].
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

        # Keep same-direction size unchanged to avoid confidence-only churn.
        if prev_direction == next_direction and prev_direction != 0.0:
            desired_position = prev_position

        has_change = abs(desired_position - prev_position) > eps
        can_change = (idx - last_change_idx) >= min_hold

        # Fix 3: minimum hold period (3 bars by default) for any position change.
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
    if neg_sum <= 1e-12:
        profit_factor = float("inf") if pos_sum > 0 else 0.0
    else:
        profit_factor = pos_sum / neg_sum

    total_trades = int(np.sum(turnover > eps))
    average_trade_return = float(np.mean(active_returns)) if len(active_returns) else 0.0
    coverage = float(np.mean(active_mask)) if len(active_mask) else 0.0
    status = (
        "OK"
        if (
            sharpe > 1.0
            and sortino > 1.2
            and max_drawdown < 0.20
            and win_rate > 0.45
            and profit_factor > 1.3
        )
        else "WARN"
    )

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "total_trades": total_trades,
        "average_trade_return": average_trade_return,
        "coverage": coverage,
        "status": status,
    }


def to_prob_df(timestamps: list[str], split: str, probs: np.ndarray, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce"),
            "split": split,
            f"{prefix}_prob_down": probs[:, 0],
            f"{prefix}_prob_neutral": probs[:, 1],
            f"{prefix}_prob_up": probs[:, 2],
        }
    )
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)


def print_ablation_table(symbol: str, rows: list[tuple[str, float, float, float, str]]) -> None:
    print(f"[{symbol}] ABLATION:")
    print("Config           Val Bal  Test Bal  Sharpe  Status")
    for config_name, val_bal, test_bal, sharpe, status in rows:
        print(f"{config_name:<15} {val_bal:.4f}   {test_bal:.4f}   {sharpe:.4f}  {status}")


def build_symbol_meta(
    symbol_artifact: ValidArtifact,
    args: argparse.Namespace,
    device: torch.device,
    fx_context_frame: pd.DataFrame,
) -> dict[str, Any]:
    symbol = symbol_artifact.symbol
    summary = symbol_artifact.summary

    load_args = argparse.Namespace(
        data_path=None,
        database_url=args.database_url,
        gold_dir=args.gold_dir,
        interval=args.interval,
        limit=args.limit,
        min_rows=args.min_rows,
        fx_context_symbol=args.fx_context_symbol,
    )
    frame, _, _ = load_symbol_frame(symbol, load_args, fx_context_frame=fx_context_frame)
    pre_feature_train_end = int(len(frame) * 0.70)
    frame, _ = add_volatility_regime_features(frame, pre_feature_train_end)
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill()
    required_after_regime = [col for col in ("open", "high", "low", "close", "volume", "log_return") if col in frame.columns]
    if required_after_regime:
        frame = frame.dropna(subset=required_after_regime)
    frame = frame.reset_index(drop=True)

    train_end, val_end = infer_split_indices(summary, len(frame))
    threshold = infer_threshold(summary)
    split = split_series(len(frame), train_end, val_end)
    frame = frame.copy()
    frame["split"] = split

    train_start = frame["timestamp"].iloc[0]
    train_end_ts = frame["timestamp"].iloc[train_end - 1]
    print(f"[{symbol}] Artifacts loaded:")
    print(f"CNN: {symbol_artifact.cnn_path} ({format_bytes(symbol_artifact.cnn_path)})")
    print(f"LSTM: {symbol_artifact.lstm_path} ({format_bytes(symbol_artifact.lstm_path)})")
    print(f"Threshold used in training: {threshold:.6f}")
    print(f"Training date range: {train_start} to {train_end_ts}")

    feature_table = build_feature_table(frame)
    next_log_returns = frame["log_return"].shift(-1).astype(float).to_numpy()

    cnn_ckpt = torch.load(symbol_artifact.cnn_path, map_location=device)
    lstm_ckpt = torch.load(symbol_artifact.lstm_path, map_location=device)
    scaler = load_pickle(symbol_artifact.scaler_path)

    medians_path = symbol_artifact.base_path / "arima_lstm" / "feature_imputation_medians.pkl"
    arima_model_path = symbol_artifact.base_path / "arima_lstm" / "arima_model.pkl"
    if not medians_path.exists():
        raise FileNotFoundError(f"Missing medians artifact: {medians_path}")
    if not arima_model_path.exists():
        raise FileNotFoundError(f"Missing ARIMA artifact: {arima_model_path}")
    medians = load_pickle(medians_path)
    arima_model = load_pickle(arima_model_path)

    cnn_columns = list(cnn_ckpt.get("feature_columns", []))
    lstm_columns = list(lstm_ckpt.get("feature_columns", []))
    if not cnn_columns:
        raise ValueError("CNN checkpoint missing feature_columns.")
    if not lstm_columns:
        raise ValueError("LSTM checkpoint missing feature_columns.")

    cnn_feature_frame = feature_table.reindex(columns=cnn_columns).copy()
    cnn_scaled, _, _ = impute_and_scale_features(cnn_feature_frame, train_end)
    cnn_bundle = build_cnn_sequences(
        feature_values=cnn_scaled,
        close_values=frame["close"].to_numpy(dtype=np.float64),
        timestamps=frame["timestamp"],
        train_end=train_end,
        val_end=val_end,
        window_size=int(cnn_ckpt.get("window_size", 30)),
        neutral_threshold=None,
        next_log_returns=next_log_returns,
        label_threshold=threshold,
    )
    cnn_model = DirectionCNNClassifier(
        time_steps=cnn_bundle.X_train.shape[-2],
        num_features=cnn_bundle.X_train.shape[-1],
        dropout=float(cnn_ckpt.get("dropout", 0.25)),
    ).to(device)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])

    _, cnn_prob_train_internal = predict_split_outputs(cnn_model, cnn_bundle.X_train, device)
    _, cnn_prob_val_internal = predict_split_outputs(cnn_model, cnn_bundle.X_val, device)
    _, cnn_prob_test_internal = predict_split_outputs(cnn_model, cnn_bundle.X_test, device)
    cnn_prob_train = internal_probs_to_user(cnn_prob_train_internal)
    cnn_prob_val = internal_probs_to_user(cnn_prob_val_internal)
    cnn_prob_test = internal_probs_to_user(cnn_prob_test_internal)

    validate_prob_rows(cnn_prob_train)
    validate_prob_rows(cnn_prob_val)
    validate_prob_rows(cnn_prob_test)

    reg_feature_frame = feature_table.reindex(columns=lstm_columns).copy()
    train_medians = reg_feature_frame.iloc[:train_end].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    default_medians = {col: float(train_medians.get(col, 0.0)) for col in reg_feature_frame.columns}
    merged_medians = {**default_medians, **{str(k): float(v) for k, v in medians.items() if k in reg_feature_frame.columns}}
    medians_series = pd.Series(merged_medians)
    reg_filled = (
        reg_feature_frame.replace([np.inf, -np.inf], np.nan)
        .fillna(medians_series)
        .fillna(0.0)
    )
    reg_scaled = np.asarray(scaler.transform(reg_filled.to_numpy()), dtype=np.float32)

    arima_baseline = compute_arima_baseline_from_loaded_model(arima_model, frame["log_return"].to_numpy(dtype=np.float64), train_end)
    residual_target = frame["log_return"].to_numpy(dtype=np.float64) - arima_baseline
    reg_bundle = build_regression_sequences(
        scaled_features=reg_scaled,
        residual_target=residual_target,
        timestamps=frame["timestamp"],
        close_values=frame["close"].to_numpy(dtype=np.float64),
        arima_baseline=arima_baseline,
        train_end=train_end,
        val_end=val_end,
        window_size=int(lstm_ckpt.get("window_size", 20)),
    )
    lstm_model = ResidualLSTMRegressor(
        input_size=reg_bundle.X_train.shape[-1],
        hidden_size=int(lstm_ckpt.get("hidden_size", 64)),
        num_layers=int(lstm_ckpt.get("num_layers", 2)),
        dropout=float(lstm_ckpt.get("dropout", 0.25)),
    ).to(device)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])

    lstm_res_train, _ = predict_split_outputs(lstm_model, reg_bundle.X_train, device)
    lstm_res_val, _ = predict_split_outputs(lstm_model, reg_bundle.X_val, device)
    lstm_res_test, _ = predict_split_outputs(lstm_model, reg_bundle.X_test, device)
    lstm_res_train = np.asarray(lstm_res_train, dtype=np.float64).reshape(-1)
    lstm_res_val = np.asarray(lstm_res_val, dtype=np.float64).reshape(-1)
    lstm_res_test = np.asarray(lstm_res_test, dtype=np.float64).reshape(-1)

    lstm_score_train = arima_baseline[reg_bundle.train_indices] + lstm_res_train
    lstm_score_val = arima_baseline[reg_bundle.val_indices] + lstm_res_val
    lstm_score_test = arima_baseline[reg_bundle.test_indices] + lstm_res_test

    lstm_prob_train = scores_to_soft_probs(lstm_score_train, threshold)
    lstm_prob_val = scores_to_soft_probs(lstm_score_val, threshold)
    lstm_prob_test = scores_to_soft_probs(lstm_score_test, threshold)

    validate_prob_rows(lstm_prob_train)
    validate_prob_rows(lstm_prob_val)
    validate_prob_rows(lstm_prob_test)

    cnn_df = pd.concat(
        [
            to_prob_df(cnn_bundle.train_timestamps, "train", cnn_prob_train, "cnn"),
            to_prob_df(cnn_bundle.val_timestamps, "val", cnn_prob_val, "cnn"),
            to_prob_df(cnn_bundle.test_timestamps, "test", cnn_prob_test, "cnn"),
        ],
        ignore_index=True,
    )
    lstm_df = pd.concat(
        [
            to_prob_df(reg_bundle.train_timestamps, "train", lstm_prob_train, "lstm"),
            to_prob_df(reg_bundle.val_timestamps, "val", lstm_prob_val, "lstm"),
            to_prob_df(reg_bundle.test_timestamps, "test", lstm_prob_test, "lstm"),
        ],
        ignore_index=True,
    )
    aligned = cnn_df.merge(lstm_df, on=["timestamp", "split"], how="inner")
    if aligned.empty:
        raise ValueError("No aligned timestamps between CNN and LSTM outputs.")

    labels_internal = np.full(len(frame), -1, dtype=np.int64)
    finite_mask = np.isfinite(next_log_returns)
    labels_internal[finite_mask] = apply_labels(next_log_returns[finite_mask], threshold)
    labels_user = internal_label_to_user(labels_internal)
    labels_df = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "split": frame["split"],
            "y_label": labels_user,
            "actual_next_log_return": next_log_returns,
        }
    )
    labels_df = labels_df[labels_df["y_label"] >= 0].copy()
    labels_df["y_label"] = labels_df["y_label"].astype(np.int64)

    # Original technical features from persisted scaler output.
    technical_columns = [f"tech_{col}" for col in lstm_columns]
    tech_df = pd.DataFrame(reg_scaled, columns=technical_columns)
    tech_df["timestamp"] = frame["timestamp"]
    tech_df["split"] = frame["split"]

    hmm_regime, regime_shift_flag = maybe_load_regime_labels(symbol, symbol_artifact.base_path, frame, train_end)
    regime_df = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "split": frame["split"],
            "hmm_regime": hmm_regime.astype(np.int64),
            "regime_shift_flag": regime_shift_flag.astype(np.int64),
        }
    )

    meta_df = aligned.merge(labels_df, on=["timestamp", "split"], how="inner")
    meta_df = meta_df.merge(tech_df, on=["timestamp", "split"], how="inner")
    meta_df = meta_df.merge(regime_df, on=["timestamp", "split"], how="inner")
    meta_df = meta_df.sort_values(["split", "timestamp"]).reset_index(drop=True)
    if meta_df.empty:
        raise ValueError("Meta-feature frame is empty after alignment.")

    meta_df["cnn_prob_up"] = meta_df["cnn_prob_up"].astype(np.float64)
    meta_df["cnn_up_reliable"] = (meta_df["cnn_prob_up"] > 0.45).astype(np.int64)
    meta_df["cnn_confidence"] = meta_df[["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].max(axis=1)
    meta_df["lstm_confidence"] = meta_df[["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].max(axis=1)
    cnn_argmax = np.argmax(meta_df[["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(), axis=1)
    lstm_argmax = np.argmax(meta_df[["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(), axis=1)
    meta_df["model_agreement"] = (cnn_argmax == lstm_argmax).astype(np.int64)

    feature_names = [
        "cnn_prob_down",
        "cnn_prob_neutral",
        "cnn_prob_up",
        "cnn_up_reliable",
        "lstm_prob_down",
        "lstm_prob_neutral",
        "lstm_prob_up",
        "cnn_confidence",
        "lstm_confidence",
        "model_agreement",
        "hmm_regime",
        "regime_shift_flag",
        *technical_columns,
    ]

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_frame = meta_df[meta_df["split"] == split_name].copy()
        split_frame = split_frame.sort_values("timestamp").reset_index(drop=True)
        if split_frame.empty:
            raise ValueError(f"No rows in split={split_name} after meta-feature assembly.")
        split_frames[split_name] = split_frame

    print(
        f"[{symbol}] OOS probs: train={len(split_frames['train'])} "
        f"val={len(split_frames['val'])} test={len(split_frames['test'])}"
    )
    print(f"[{symbol}] Meta-feature matrix:")
    print(f"Shape: ({len(meta_df)}, {len(feature_names)})")
    print("CNN features: 4")
    print("LSTM features: 3")
    print("Derived: 3")
    print("Regime: 2")
    print(f"Technical: {len(technical_columns)}")
    print(f"Total: {len(feature_names)}")

    return {
        "summary": summary,
        "threshold": threshold,
        "train_end": train_end,
        "val_end": val_end,
        "feature_names": feature_names,
        "split_frames": split_frames,
        "cnn_probs": {
            "train": split_frames["train"][["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(dtype=np.float64),
            "val": split_frames["val"][["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(dtype=np.float64),
            "test": split_frames["test"][["cnn_prob_down", "cnn_prob_neutral", "cnn_prob_up"]].to_numpy(dtype=np.float64),
        },
        "lstm_probs": {
            "train": split_frames["train"][["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(dtype=np.float64),
            "val": split_frames["val"][["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(dtype=np.float64),
            "test": split_frames["test"][["lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"]].to_numpy(dtype=np.float64),
        },
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir, args.interval)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else None

    candidate_symbols = resolve_candidate_symbols(artifacts_dir)
    if not candidate_symbols:
        raise ValueError("No candidate equity symbols found in EQUITY_SYMBOLS or artifact summaries.")

    valid_artifacts: list[ValidArtifact] = []
    skipped: dict[str, SkipEntry] = {}

    print("STEP 0: ARTIFACT DISCOVERY AND VALIDATION")
    for symbol in candidate_symbols:
        valid, skip = validate_symbol_artifact(
            symbol=symbol,
            artifacts_dir=artifacts_dir,
            manifest=manifest,
            min_rows=args.min_rows,
            expected_freq=PRIMARY_FREQ,
        )
        if valid is not None:
            valid_artifacts.append(valid)
            continue
        if skip is not None:
            skipped[symbol] = skip
            warn_skip(symbol, skip.reason, skip.fix)

    valid_symbols = [item.symbol for item in valid_artifacts]
    print(f"Valid symbols for XGBoost: {valid_symbols}")
    skipped_summary = [f"{sym} ({entry.reason})" for sym, entry in skipped.items()]
    print(f"Skipped symbols: {skipped_summary}")
    print("USDINR=X: external feature only")

    assert len(valid_symbols) >= 2, "Cannot run XGBoost meta-layer with fewer than 2 valid base model outputs"

    try:
        if os.getenv("OMP_NUM_THREADS") is None:
            os.environ["OMP_NUM_THREADS"] = "1"
            print("WARNING: OMP_NUM_THREADS was unset; defaulting to 1 for XGBoost runtime stability on this host.")

        from xgboost import XGBClassifier
    except Exception as exc:
        raise ModuleNotFoundError(
            "xgboost is required for this pipeline. Install it with: pip install xgboost"
        ) from exc

    load_args = argparse.Namespace(
        data_path=None,
        database_url=args.database_url,
        gold_dir=args.gold_dir,
        interval=args.interval,
        limit=args.limit,
        min_rows=args.min_rows,
        fx_context_symbol=args.fx_context_symbol,
    )
    fx_context_frame, _, _ = load_symbol_frame(
        symbol=args.fx_context_symbol,
        args=load_args,
        fx_context_frame=None,
    )
    fx_context_frame = fx_context_frame[["timestamp", "close"]].rename(columns={"close": "usdinr_close"})
    device = torch.device(args.device)

    symbol_results: list[SymbolResult] = []

    for symbol_artifact in valid_artifacts:
        symbol = symbol_artifact.symbol
        print("\n" + "=" * 88)
        print(f"STEP 1-8: {symbol}")
        print("=" * 88)

        try:
            meta_data = build_symbol_meta(
                symbol_artifact=symbol_artifact,
                args=args,
                device=device,
                fx_context_frame=fx_context_frame,
            )
        except Exception as exc:
            reason = f"meta feature build failed: {exc}"
            fix = f"repair saved artifacts/data for {symbol} and rerun"
            skipped[symbol] = SkipEntry(reason=reason, fix=fix)
            warn_skip(symbol, reason, fix)
            continue

        split_frames = meta_data["split_frames"]
        feature_names = meta_data["feature_names"]

        train_df = split_frames["train"]
        val_df = split_frames["val"]
        test_df = split_frames["test"]

        X_train = np.ascontiguousarray(train_df[feature_names].to_numpy(dtype=np.float32))
        y_train_int = np.ascontiguousarray(train_df["y_label"].to_numpy(dtype=np.int64))
        X_val = np.ascontiguousarray(val_df[feature_names].to_numpy(dtype=np.float32))
        y_val_int = np.ascontiguousarray(val_df["y_label"].to_numpy(dtype=np.int64))
        X_test = np.ascontiguousarray(test_df[feature_names].to_numpy(dtype=np.float32))
        y_test_int = np.ascontiguousarray(test_df["y_label"].to_numpy(dtype=np.int64))

        # XGBoost 3.2.0 on this environment can segfault on int64 labels.
        # Use float32 labels for fit while retaining integer labels for metrics.
        y_train_fit = np.ascontiguousarray(y_train_int.astype(np.float32))
        y_val_fit = np.ascontiguousarray(y_val_int.astype(np.float32))

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        sample_weight = np.ascontiguousarray(
            compute_sample_weight(class_weight="balanced", y=y_train_int).astype(np.float32)
        )
        xgb.fit(
            X_train,
            y_train_fit,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_fit)],
            verbose=False,
        )
        best_iteration = int(getattr(xgb, "best_iteration", xgb.n_estimators - 1))
        evals_result = xgb.evals_result()
        val_mlogloss = float(evals_result["validation_0"]["mlogloss"][best_iteration])
        print(f"[{symbol}] XGBoost best_iteration={best_iteration} val_mlogloss={val_mlogloss:.6f}")

        xgb_probs_train = xgb.predict_proba(X_train)
        xgb_probs_val = xgb.predict_proba(X_val)
        xgb_probs_test = xgb.predict_proba(X_test)
        validate_prob_rows(xgb_probs_train)
        validate_prob_rows(xgb_probs_val)
        validate_prob_rows(xgb_probs_test)

        xgb_dir = symbol_artifact.base_path / "xgboost"
        xgb_dir.mkdir(parents=True, exist_ok=True)
        xgb_model_path = xgb_dir / "model.json"
        xgb.save_model(str(xgb_model_path))
        print(f"[{symbol}] XGBoost model saved: {xgb_model_path} ({format_bytes(xgb_model_path)})")

        importances = np.asarray(xgb.feature_importances_, dtype=np.float64)
        sorted_idx = np.argsort(importances)[::-1]
        print(f"[{symbol}] Top 10 feature importances:")
        for rank, idx in enumerate(sorted_idx[:10], start=1):
            print(f"  {rank:>2}. {feature_names[int(idx)]}: {importances[int(idx)]:.6f}")
        top_feature = feature_names[int(sorted_idx[0])] if len(sorted_idx) else ""
        if top_feature not in {"cnn_prob_up", "lstm_prob_down", "lstm_prob_neutral", "lstm_prob_up"}:
            print(f"WARNING: [{symbol}] top feature '{top_feature}' is not a base-model probability output")

        cnn_probs = meta_data["cnn_probs"]
        lstm_probs = meta_data["lstm_probs"]
        weight_grid = generate_weight_grid(min_weight=0.1, max_weight=0.7, step=0.1)
        best_weight_score = -math.inf
        best_weights = (0.3, 0.3, 0.4)
        for w_cnn, w_lstm, w_xgb in weight_grid:
            combined = (w_cnn * cnn_probs["val"]) + (w_lstm * lstm_probs["val"]) + (w_xgb * xgb_probs_val)
            preds = np.argmax(combined, axis=1)
            score = float(balanced_accuracy_score(y_val_int, preds))
            if score > best_weight_score:
                best_weight_score = score
                best_weights = (w_cnn, w_lstm, w_xgb)
        weights_payload = {
            "symbol": symbol,
            "weights": {
                "cnn": best_weights[0],
                "lstm": best_weights[1],
                "xgb": best_weights[2],
            },
            "val_balanced_accuracy": best_weight_score,
        }
        save_json(xgb_dir / "ensemble_weights.json", weights_payload)
        print(
            f"[{symbol}] Best ensemble weights:\n"
            f"CNN={best_weights[0]:.1f} LSTM={best_weights[1]:.1f} XGB={best_weights[2]:.1f}\n"
            f"Val Balanced Acc: {best_weight_score:.4f}"
        )

        ensemble_probs_val = (
            best_weights[0] * cnn_probs["val"] + best_weights[1] * lstm_probs["val"] + best_weights[2] * xgb_probs_val
        )
        ensemble_probs_test = (
            best_weights[0] * cnn_probs["test"] + best_weights[1] * lstm_probs["test"] + best_weights[2] * xgb_probs_test
        )

        best_thresh = 0.5
        best_f1 = -math.inf
        best_coverage = 0.0
        for conf_thresh in np.round(np.arange(0.35, 0.751, 0.05), 2):
            preds = apply_confidence_threshold(ensemble_probs_val, conf_thresh)
            score = float(f1_score(y_val_int, preds, average="weighted", zero_division=0))
            coverage = float(np.mean(np.max(ensemble_probs_val, axis=1) >= conf_thresh))
            if score > best_f1:
                best_f1 = score
                best_thresh = float(conf_thresh)
                best_coverage = coverage
        threshold_payload = {
            "symbol": symbol,
            "confidence_threshold": best_thresh,
            "val_weighted_f1": best_f1,
            "val_coverage": best_coverage,
        }
        save_json(xgb_dir / "confidence_threshold.json", threshold_payload)
        print(
            f"[{symbol}] Best confidence threshold: {best_thresh:.2f}\n"
            f"Val F1 at threshold: {best_f1:.4f}\n"
            f"Coverage: {best_coverage * 100.0:.2f}%"
        )
        if best_coverage < 0.30:
            print(f"WARNING: [{symbol}] coverage < 30% — threshold may be too high")

        test_actual_log_returns = test_df["actual_next_log_return"].to_numpy(dtype=np.float64)
        backtest = compute_backtest_metrics(ensemble_probs_test, test_actual_log_returns, threshold=best_thresh)
        print(f"[{symbol}] Backtest metrics:")
        print(f"Sharpe ratio: {backtest['sharpe']:.4f} | Total trades: {backtest['total_trades']} (target > 1.0)")
        print(f"Sortino ratio: {backtest['sortino']:.4f} (target > 1.2)")
        print(f"Max drawdown: {backtest['max_drawdown'] * 100.0:.2f}% (target < 20%)")
        print(f"Win rate: {backtest['win_rate'] * 100.0:.2f}% (target > 45%)")
        print(f"Profit factor: {backtest['profit_factor']:.4f} (target > 1.3)")
        print(f"Average trade return: {backtest['average_trade_return']:.6f}")
        print(f"Coverage: {backtest['coverage'] * 100.0:.2f}%")
        print(f"Status: {backtest['status']}")

        cnn_only_val = cnn_probs["val"]
        cnn_only_test = cnn_probs["test"]
        lstm_only_val = lstm_probs["val"]
        lstm_only_test = lstm_probs["test"]
        simple_val = (cnn_probs["val"] + lstm_probs["val"] + xgb_probs_val) / 3.0
        simple_test = (cnn_probs["test"] + lstm_probs["test"] + xgb_probs_test) / 3.0
        tuned_val = ensemble_probs_val
        tuned_test = ensemble_probs_test

        def evaluate_config(name: str, val_probs: np.ndarray, test_probs: np.ndarray, threshold_override: float | None = None) -> tuple[float, float, float, str]:
            val_preds = apply_confidence_threshold(val_probs, threshold_override) if threshold_override is not None else np.argmax(val_probs, axis=1)
            test_preds = apply_confidence_threshold(test_probs, threshold_override) if threshold_override is not None else np.argmax(test_probs, axis=1)
            val_bal = float(balanced_accuracy_score(y_val_int, val_preds))
            test_bal = float(balanced_accuracy_score(y_test_int, test_preds))
            bt = compute_backtest_metrics(test_probs, test_actual_log_returns, threshold=threshold_override)
            status = "OK" if (test_bal > 0.42 and bt["sharpe"] > 1.0) else "WARN"
            _ = name
            return val_bal, test_bal, bt["sharpe"], status

        ablation_rows = [
            ("CNN only", *evaluate_config("CNN only", cnn_only_val, cnn_only_test, None)),
            ("LSTM only", *evaluate_config("LSTM only", lstm_only_val, lstm_only_test, None)),
            ("Simple average", *evaluate_config("Simple average", simple_val, simple_test, None)),
            ("Tuned ensemble", *evaluate_config("Tuned ensemble", tuned_val, tuned_test, best_thresh)),
        ]
        print_ablation_table(symbol, ablation_rows)
        simple_test_bal = next(row[2] for row in ablation_rows if row[0] == "Simple average")
        tuned_test_bal = next(row[2] for row in ablation_rows if row[0] == "Tuned ensemble")
        if tuned_test_bal <= simple_test_bal:
            print(
                f"WARNING: [{symbol}] Tuned ensemble did not beat simple average — "
                "XGBoost may not be adding value\n"
                f"Consider using simple average for {symbol}"
            )

        xgb_val_bal = float(balanced_accuracy_score(y_val_int, np.argmax(xgb_probs_val, axis=1)))
        xgb_test_bal = float(balanced_accuracy_score(y_test_int, np.argmax(xgb_probs_test, axis=1)))
        ens_val_bal = float(
            balanced_accuracy_score(y_val_int, apply_confidence_threshold(ensemble_probs_val, best_thresh))
        )
        ens_test_bal = float(
            balanced_accuracy_score(y_test_int, apply_confidence_threshold(ensemble_probs_test, best_thresh))
        )
        symbol_results.append(
            SymbolResult(
                symbol=symbol,
                xgb_val_bal=xgb_val_bal,
                xgb_test_bal=xgb_test_bal,
                ens_val_bal=ens_val_bal,
                ens_test_bal=ens_test_bal,
                sharpe=backtest["sharpe"],
                drawdown=backtest["max_drawdown"],
                win_rate=backtest["win_rate"],
                status=backtest["status"],
            )
        )

    final_results = [row for row in symbol_results if row.symbol not in skipped]
    if len(final_results) == 0:
        raise RuntimeError("All valid symbols failed during XGBoost processing.")

    print("\n" + "=" * 100)
    print("STEP 9: CROSS-SYMBOL SUMMARY")
    print("=" * 100)
    print("Symbol          XGB Val  XGB Test  Ens Val  Ens Test  Sharpe  Drawdown  WinRate  Status")
    print("-" * 100)
    for item in final_results:
        print(
            f"{item.symbol:<15} {item.xgb_val_bal:>7.4f}  {item.xgb_test_bal:>8.4f}  "
            f"{item.ens_val_bal:>7.4f}  {item.ens_test_bal:>8.4f}  {item.sharpe:>6.3f}  "
            f"{item.drawdown * 100.0:>7.2f}%  {item.win_rate * 100.0:>6.2f}%  {item.status}"
        )

    avg_xgb_val = float(np.mean([r.xgb_val_bal for r in final_results]))
    avg_xgb_test = float(np.mean([r.xgb_test_bal for r in final_results]))
    avg_ens_val = float(np.mean([r.ens_val_bal for r in final_results]))
    avg_ens_test = float(np.mean([r.ens_test_bal for r in final_results]))
    avg_sharpe = float(np.mean([r.sharpe for r in final_results]))
    avg_mdd = float(np.mean([r.drawdown for r in final_results]))
    avg_win = float(np.mean([r.win_rate for r in final_results]))

    print("-" * 100)
    print(f"Avg XGB Val Bal Acc:       {avg_xgb_val:.4f}  target > 0.44")
    print(f"Avg XGB Test Bal Acc:      {avg_xgb_test:.4f}  target > 0.42")
    print(f"Avg Ensemble Val Bal Acc:  {avg_ens_val:.4f}  target > 0.48")
    print(f"Avg Ensemble Test Bal Acc: {avg_ens_test:.4f}  target > 0.45")
    print(f"Avg Sharpe:                {avg_sharpe:.4f}  target > 1.0")
    print(f"Avg Max Drawdown:          {avg_mdd * 100.0:.2f}%  target < 20%")
    print(f"Avg Win Rate:              {avg_win * 100.0:.2f}%  target > 45%")

    print("\nSkipped symbols:")
    if skipped:
        for symbol, entry in skipped.items():
            print(f"{symbol}: skipped — {entry.reason}")
            print(f"Fix: {entry.fix}")
    else:
        print("None")

    print("\nFooter:")
    print(FX_RESULTS_NOTE)
    print(f"Valid symbols this run: {[item.symbol for item in final_results]}")
    print(f"Skipped symbols: {list(skipped.keys())}")
    print("To add a new symbol: add hourly data to pipeline and train — XGBoost includes it automatically on next run.")


if __name__ == "__main__":
    main()
