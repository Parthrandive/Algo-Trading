import argparse
import json
import logging
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from src.agents.technical.label_utils import (
    build_labels as build_labels_unified,
    choose_neutral_threshold as choose_neutral_threshold_unified,
    atr_effective_threshold,
    recall_balance as compute_recall_balance,
    directional_coverage as compute_directional_coverage,
    class_balance_report,
    LABEL_MODES,
)
from src.agents.technical.validation_metrics import (
    post_cost_sharpe,
    expected_calibration_error,
)
try:
    from config.symbols import EQUITY_SYMBOLS, FOREX_SYMBOLS, dedupe_symbols, MIN_ROWS, SplitCounts, SymbolValidationResult, discover_training_symbols
except ImportError:
    from config.symbols import EQUITY_SYMBOLS, FOREX_SYMBOLS, dedupe_symbols
    MIN_ROWS = 300

from config.symbols import is_forex, validate_equity_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
EPS = 1e-8
UP_WEIGHT_MULTIPLIER = 3.0
BEARISH_DOMINANCE_RATIO = 0.70
REGIME_SHIFT_THRESHOLD = 0.10


@dataclass
class SymbolTrainingResult:
    symbol: str
    neutral_threshold: float
    best_threshold: float
    epochs_run: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    test_acc: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    class_weight_dict: Dict[int, float]


class HybridCnnLstmClassifier(nn.Module):
    """
    ARIMA-LSTM-CNN style classifier:
    - ARIMA features are injected in the input feature set.
    - A CNN encoder extracts local temporal patterns.
    - LSTM captures sequential dependencies before classification.
    """

    def __init__(self, num_features: int, num_classes: int, lstm_hidden_size: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window_size, features)
        x = x.transpose(1, 2)  # (batch, features, window_size)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq_len, channels=64)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_symbols(symbol: Optional[str], symbols: Optional[str]) -> List[str]:
    if symbol:
        return [symbol.strip()]
    if not symbols:
        return []
    return [s.strip() for s in symbols.split(",") if s.strip()]


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")

def parse_atr_overrides(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return {}
    overrides = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            sym, val = part.rsplit(":", 1)
            try:
                overrides[sym.strip()] = float(val.strip())
            except ValueError:
                pass
    return overrides

def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def safe_link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source artifact not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.symlink(str(source), str(target))
    except OSError:
        shutil.copy2(source, target)


def safe_torch_save(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    torch.save(model.state_dict(), path)


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def directional_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, use_binary: bool) -> Dict[str, Dict[str, float]]:
    """
    Return precision/recall/F1/support for actionable directions.
    In multiclass mode, actionable classes are up/down (neutral excluded).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "up": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "down": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "directional_accuracy": 0.0,
            "recall_balance": 0.0,
            "directional_coverage": 0.0,
        }

    labels = [0, 1] if use_binary else [0, 2]  # binary: 0=up,1=down; multiclass: 0=up,2=down
    names = ["up", "down"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    if use_binary:
        directional_mask = np.ones(len(y_true), dtype=bool)
    else:
        directional_mask = np.isin(y_true, labels)

    directional_accuracy = (
        float(accuracy_score(y_true[directional_mask], y_pred[directional_mask]))
        if directional_mask.any()
        else 0.0
    )

    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(names):
        out[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    out["directional_accuracy"] = float(directional_accuracy)
    out["recall_balance"] = compute_recall_balance(y_true, y_pred)
    out["directional_coverage"] = compute_directional_coverage(y_pred)
    return out


def validate_data(df: pd.DataFrame, symbol: str = "UNKNOWN", min_rows: int = 40) -> None:
    if len(df) < min_rows:
        raise ValueError(f"{symbol}: Need at least {min_rows} rows for stable split/windows. Got {len(df)}.")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: Missing columns: {sorted(missing)}")
    for col in ["open", "high", "low", "close", "volume"]:
        nan_pct = float(df[col].isna().mean())
        if nan_pct > 0.05:
            raise ValueError(f"{symbol}: Column '{col}' has {nan_pct:.1%} NaNs (max 5% allowed).")
    if df["close"].nunique() <= 1:
        raise ValueError(f"{symbol}: Close series is constant; model cannot learn.")


def load_symbol_from_local_silver(symbol: str, base_dir: str) -> pd.DataFrame:
    root = Path(base_dir)
    symbol_dir = root / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"{symbol}: local directory not found at {symbol_dir}")

    frames: List[pd.DataFrame] = []
    for parquet_file in sorted(symbol_dir.rglob("*.parquet")):
        part = pd.read_parquet(parquet_file)
        if part is None or part.empty:
            continue
        frames.append(part)

    if not frames:
        raise ValueError(f"{symbol}: no parquet rows found under {symbol_dir}")

    df = pd.concat(frames, ignore_index=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def prepare_usdinr_features(df_usdinr: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df_usdinr.columns or "close" not in df_usdinr.columns:
        raise ValueError("USDINR context data must include timestamp and close columns.")

    frame = df_usdinr.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    if frame.empty:
        raise ValueError("USDINR context data is empty after cleaning.")

    frame["usdinr_close"] = frame["close"]
    frame["usdinr_return"] = np.log(frame["usdinr_close"] / frame["usdinr_close"].shift(1))
    rolling_mean = frame["usdinr_return"].rolling(20, min_periods=20).mean()
    rolling_std = frame["usdinr_return"].rolling(20, min_periods=20).std()
    frame["usdinr_zscore"] = (frame["usdinr_return"] - rolling_mean) / (rolling_std + EPS)
    frame["usdinr_trend"] = frame["usdinr_close"] / frame["usdinr_close"].rolling(20, min_periods=20).mean() - 1.0
    frame["usdinr_vol"] = frame["usdinr_return"].rolling(20, min_periods=20).std()

    return frame[
        [
            "timestamp",
            "usdinr_close",
            "usdinr_return",
            "usdinr_zscore",
            "usdinr_trend",
            "usdinr_vol",
        ]
    ].dropna().reset_index(drop=True)


def load_fx_context_frame(
    loader: DataLoader,
    *,
    local_silver_dir: Optional[str],
    limit: Optional[int],
    use_nse: bool,
    interval: str,
) -> pd.DataFrame:
    fx_symbol = FOREX_SYMBOLS[0]
    if local_silver_dir:
        frame = load_symbol_from_local_silver(symbol=fx_symbol, base_dir=local_silver_dir)
        if limit is not None:
            frame = frame.tail(limit).copy()
    else:
        frame = loader.load_historical_bars(
            symbol=fx_symbol,
            limit=limit,
            use_nse_fallback=use_nse,
            min_fallback_rows=180,
            interval=interval,
        )
    return prepare_usdinr_features(frame)


def merge_usdinr_features(df: pd.DataFrame, fx_context: pd.DataFrame) -> pd.DataFrame:
    if fx_context.empty:
        raise ValueError("USDINR context features are required before training any equity symbol.")

    target = df.copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"], utc=True, errors="coerce")
    target = target.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    merged = pd.merge_asof(
        target,
        fx_context.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    for column in ["usdinr_close", "usdinr_return", "usdinr_zscore", "usdinr_trend", "usdinr_vol"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").ffill()
    return merged


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def quality_gate_split_counts(df: pd.DataFrame) -> SplitCounts:
    train_df, val_df, test_df = chronological_split(df)
    return SplitCounts(
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )


def add_symbol_features(df: pd.DataFrame, train_len: int) -> pd.DataFrame:
    out = df.copy()
    out["returns"] = out["close"].pct_change()

    rolling_vol = out["close"].rolling(20, min_periods=20).std()
    train_vol = rolling_vol.iloc[:train_len].dropna()
    if train_vol.empty:
        q1, q2 = 0.0, 0.0
    else:
        q1 = float(train_vol.quantile(1 / 3))
        q2 = float(train_vol.quantile(2 / 3))
        if q2 <= q1:
            q2 = q1 + EPS

    vol_regime = np.where(rolling_vol <= q1, 0.0, np.where(rolling_vol <= q2, 1.0, 2.0))
    out["vol_regime"] = vol_regime
    out.loc[rolling_vol.isna(), "vol_regime"] = np.nan

    ret_roll_mean = out["returns"].rolling(20, min_periods=20).mean()
    ret_roll_std = out["returns"].rolling(20, min_periods=20).std().replace(0, np.nan)
    out["return_zscore"] = (out["returns"] - ret_roll_mean) / ret_roll_std

    vol_roll_mean = out["volume"].rolling(20, min_periods=20).mean()
    vol_roll_std = out["volume"].rolling(20, min_periods=20).std().replace(0, np.nan)
    out["volume_zscore"] = (out["volume"] - vol_roll_mean) / vol_roll_std
    return out


def build_arima_features(close: pd.Series, train_len: int, arima_order: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    values = close.astype(float).values
    n = len(values)
    forecasts = np.full(n, np.nan, dtype=np.float64)

    try:
        model = sm.tsa.ARIMA(
            values[:train_len],
            order=arima_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit()
    except Exception as exc:
        logger.warning(f"ARIMA warm fit failed ({exc}). Falling back to close lag feature.")
        fallback = pd.Series(values).shift(1).bfill().values
        residuals = values - fallback
        return fallback, residuals

    try:
        train_preds = np.asarray(results.predict(start=0, end=train_len - 1, typ="levels"), dtype=np.float64)
        forecasts[:train_len] = train_preds
    except Exception:
        forecasts[:train_len] = pd.Series(values[:train_len]).shift(1).bfill().values

    for idx in range(train_len, n):
        try:
            forecasts[idx] = float(results.forecast(steps=1)[0])
        except Exception:
            forecasts[idx] = values[idx - 1] if idx > 0 else values[idx]

        try:
            results = results.append([values[idx]], refit=False)
        except Exception:
            # Robust fallback if append is unavailable for the active statsmodels backend.
            try:
                results = sm.tsa.ARIMA(
                    values[: idx + 1],
                    order=arima_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            except Exception:
                continue

    fallback = pd.Series(values).shift(1).bfill().values
    forecasts = np.where(np.isnan(forecasts), fallback, forecasts)
    residuals = values - forecasts
    return forecasts, residuals

def prepare_split_for_supervised(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["future_return"] = out["close"].pct_change().shift(-1)
    out = out.dropna(subset=feature_columns + ["future_return"]).reset_index(drop=True)
    return out


def make_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    labels: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = df[feature_columns].values.astype(np.float32)
    forward_returns = df["future_return"].values.astype(np.float32)

    X, y, realized = [], [], []
    for end_idx in range(window_size - 1, len(df)):
        start_idx = end_idx - window_size + 1
        X.append(features[start_idx : end_idx + 1])
        y.append(int(labels[end_idx]))
        realized.append(float(forward_returns[end_idx]))

    if not X:
        return np.empty((0, window_size, len(feature_columns)), dtype=np.float32), np.empty(0, dtype=int), np.empty(
            0, dtype=np.float32
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), np.asarray(realized, dtype=np.float32)


def is_forex_target(symbol: str) -> bool:
    return is_forex(symbol)


def emit_regime_shift_warning(symbol: str, class_name: str, train_pct: float, test_pct: float, diff: float) -> None:
    logger.warning(
        "WARNING: [%s] Regime shift detected\n"
        "Train %s: %.1f%%\n"
        "Test %s:  %.1f%%\n"
        "Difference: %.1f%% exceeds 10%% threshold\n"
        "XGBoost regime feature will be critical\n"
        "for this symbol",
        symbol,
        class_name,
        train_pct * 100.0,
        class_name,
        test_pct * 100.0,
        diff * 100.0,
    )


def compute_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    symbol: str,
    y_test: np.ndarray,
    use_binary: bool,
) -> Tuple[Dict[int, float], torch.Tensor]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    up_idx = 0
    neutral_idx = None if use_binary else 1
    down_idx = 1 if use_binary else 2
    counts_train = np.bincount(y_train.astype(int), minlength=num_classes)
    up_count = int(counts_train[up_idx])
    neutral_count = int(counts_train[neutral_idx]) if neutral_idx is not None else 0
    down_count = int(counts_train[down_idx])
    bearish_dominant = (not is_forex_target(symbol)) and up_count < down_count * BEARISH_DOMINANCE_RATIO

    if bearish_dominant:
        class_weight_dict[up_idx] = float(class_weight_dict.get(up_idx, 1.0) * UP_WEIGHT_MULTIPLIER)
        logger.warning(
            "WARNING: [%s] Bearish-dominant train split detected: up=%s down=%s. Applying up class weight multiplier x%.1f",
            symbol,
            up_count,
            down_count,
            UP_WEIGHT_MULTIPLIER,
        )

    full_weights = torch.ones(num_classes, dtype=torch.float32)
    for cls, weight in class_weight_dict.items():
        full_weights[cls] = float(weight)

    regime_shift_warning = False
    if not is_forex_target(symbol) and len(y_test) > 0:
        counts_test = np.bincount(y_test.astype(int), minlength=num_classes)
        total_train = int(counts_train.sum())
        total_test = int(counts_test.sum())
        if total_train > 0 and total_test > 0:
            train_up_pct = up_count / total_train
            test_up_pct = int(counts_test[up_idx]) / total_test
            up_diff = abs(train_up_pct - test_up_pct)
            if up_diff > REGIME_SHIFT_THRESHOLD:
                regime_shift_warning = True
                emit_regime_shift_warning(symbol, "up", train_up_pct, test_up_pct, up_diff)

            train_down_pct = down_count / total_train
            test_down_pct = int(counts_test[down_idx]) / total_test
            down_diff = abs(train_down_pct - test_down_pct)
            if down_diff > REGIME_SHIFT_THRESHOLD:
                regime_shift_warning = True
                emit_regime_shift_warning(symbol, "down", train_down_pct, test_down_pct, down_diff)

    down_weight = float(full_weights[down_idx].item())
    neutral_weight = float(full_weights[neutral_idx].item()) if neutral_idx is not None else 0.0
    up_weight = float(full_weights[up_idx].item())
    logger.info("[%s] Raw class counts (train):", symbol)
    logger.info("           up=%s  neutral=%s  down=%s", up_count, neutral_count, down_count)
    logger.info("[%s] Bearish dominant: %s", symbol, "Yes" if bearish_dominant else "No")
    logger.info("[%s] Up multiplier applied: %s", symbol, "Yes" if bearish_dominant else "No")
    logger.info("[%s] Final class weights:", symbol)
    logger.info("           down=%.6f  neutral=%.6f  up=%.6f", down_weight, neutral_weight, up_weight)
    logger.info("[%s] Regime shift warning: %s", symbol, "Yes" if regime_shift_warning else "No")
    return class_weight_dict, full_weights


def predict_probabilities(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0, 0), dtype=np.float32)
    model.eval()
    probs: List[np.ndarray] = []
    loader = TorchDataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    return np.vstack(probs)


def apply_confidence_threshold(probs: np.ndarray, threshold: float, use_binary: bool) -> np.ndarray:
    if probs.size == 0:
        return np.empty(0, dtype=int)
    if use_binary:
        # 0=up, 1=down in binary mode. Threshold acts as up/down decision boundary.
        return np.where(probs[:, 0] >= threshold, 0, 1).astype(int)

    # 0=up, 1=neutral, 2=down in multi-class mode.
    argmax = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return np.where(confidence >= threshold, argmax, 1).astype(int)


def evaluate_loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    if len(X) == 0:
        return float("nan")
    loader = TorchDataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * batch_X.size(0)
            total_count += int(batch_X.size(0))
    return total_loss / max(total_count, 1)

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, float]:
    # Keras callback equivalent:
    # EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    # ModelCheckpoint(filepath='best_model_{symbol}.keras', save_best_only=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )

    train_loader = TorchDataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = TorchDataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_acc = 0.0
    best_train_acc = 0.0
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * batch_X.size(0)
            preds = torch.argmax(logits, dim=1)
            train_total += int(batch_y.size(0))
            train_correct += int((preds == batch_y).sum().item())

        avg_train_loss = train_loss / max(train_total, 1)
        avg_train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += float(loss.item()) * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                val_total += int(batch_y.size(0))
                val_correct += int((preds == batch_y).sum().item())

        avg_val_loss = val_loss / max(val_total, 1)
        avg_val_acc = val_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_train_loss = avg_train_loss
            best_train_acc = avg_train_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        epochs_run = epoch
        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Epoch [%s/%s] train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%% lr=%.6f",
                epoch,
                epochs,
                avg_train_loss,
                avg_train_acc * 100,
                avg_val_loss,
                avg_val_acc * 100,
                current_lr,
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %s (no val_loss improvement in %s epochs).",
                epoch,
                early_stopping_patience,
            )
            break

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return {
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "epochs_run": epochs_run,
    }


def tune_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    use_binary: bool,
    min_directional_pred_rate: float,
) -> Tuple[float, float, float]:
    best_threshold = 0.5
    best_score = float("-inf")
    best_directional_rate = 0.0
    for threshold in np.arange(0.15, 0.85, 0.05):
        preds = apply_confidence_threshold(probs_val, float(threshold), use_binary=use_binary)
        weighted_f1 = float(f1_score(y_val, preds, average="weighted", zero_division=0))
        if use_binary:
            directional_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
            directional_pred_rate = 1.0
            score = directional_f1
        else:
            directional_mask = np.isin(y_val, [0, 2])
            if directional_mask.any():
                directional_f1 = float(
                    f1_score(
                        y_val[directional_mask],
                        preds[directional_mask],
                        labels=[0, 2],
                        average="macro",
                        zero_division=0,
                    )
                )
            else:
                directional_f1 = 0.0

            directional_pred_rate = float(np.mean(np.isin(preds, [0, 2])))
            shortfall = max(0.0, float(min_directional_pred_rate) - directional_pred_rate)
            # Prioritize directional quality while retaining some overall class-balance signal.
            score = (0.85 * directional_f1) + (0.15 * weighted_f1) - (1.5 * shortfall)

        if score > best_score or (np.isclose(score, best_score) and directional_pred_rate > best_directional_rate):
            best_threshold = float(threshold)
            best_score = score
            best_directional_rate = directional_pred_rate
    return best_threshold, best_score, best_directional_rate


def label_to_signal(labels: np.ndarray, use_binary: bool) -> np.ndarray:
    if use_binary:
        # 0=up, 1=down
        return np.where(labels == 0, 1.0, -1.0).astype(np.float32)
    # 0=up, 1=neutral, 2=down
    signals = np.zeros(len(labels), dtype=np.float32)
    signals[labels == 0] = 1.0
    signals[labels == 2] = -1.0
    return signals


def _annualization_periods(interval: str) -> float:
    value = str(interval or "").strip().lower()
    if value.endswith("h"):
        hours = pd.Timedelta(value).total_seconds() / 3600.0
        return float((252.0 * 6.0) / max(hours, EPS))
    if value.endswith("d"):
        days = pd.Timedelta(value).total_seconds() / 86400.0
        return float(252.0 / max(days, EPS))
    return 252.0


def trading_metrics(pred_labels: np.ndarray, realized_returns: np.ndarray, use_binary: bool, interval: str) -> Dict[str, float]:
    if len(pred_labels) == 0:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "win_rate": float("nan")}

    signals = label_to_signal(pred_labels, use_binary=use_binary)
    strategy_returns = signals * realized_returns
    mean_ret = float(np.mean(strategy_returns))
    vol = float(np.std(strategy_returns))
    annualization_scale = float(np.sqrt(_annualization_periods(interval)))
    sharpe = float((mean_ret / vol) * annualization_scale) if vol > 0 else float("nan")

    equity = np.cumprod(1.0 + strategy_returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peaks, EPS) - 1.0
    max_drawdown = float(abs(np.min(drawdown)))

    active = strategy_returns[np.abs(signals) > 0]
    if len(active) == 0:
        win_rate = float("nan")
    else:
        win_rate = float(np.mean(active > 0))

    return {"sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate}


def format_distribution(dist: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.2f}%" for k, v in dist.items())

def extract_symbol_features(
    symbol: str,
    df: pd.DataFrame,
    window_size: int,
    arima_order: Tuple[int, int, int],
    min_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    validate_data(df, symbol=symbol, min_rows=min_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_df_raw, val_df_raw, test_df_raw = chronological_split(df)
    train_len = len(train_df_raw)

    feat_df = engineer_features(df)
    feat_df = add_symbol_features(feat_df, train_len=train_len)
    arima_forecast, arima_residual = build_arima_features(feat_df["close"], train_len=train_len, arima_order=arima_order)
    feat_df["arima_forecast"] = arima_forecast
    feat_df["arima_residual"] = arima_residual

    split_1 = len(train_df_raw)
    split_2 = split_1 + len(val_df_raw)
    train_feat = feat_df.iloc[:split_1].copy()
    val_feat = feat_df.iloc[split_1:split_2].copy()
    test_feat = feat_df.iloc[split_2:].copy()

    exclude_cols = {"symbol", "timestamp", "interval"}
    feature_columns = [
        c
        for c in train_feat.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_feat[c]) and not train_feat[c].isna().all()
    ]
    if not feature_columns:
        raise ValueError(f"{symbol}: no numeric feature columns available after preprocessing.")

    # Drop extremely sparse features to avoid collapsing all rows when building supervised windows.
    # Macro z-score columns can be sparse on shorter histories.
    nan_ratio = train_feat[feature_columns].isna().mean()
    feature_columns = [c for c in feature_columns if float(nan_ratio[c]) <= 0.40]
    if not feature_columns:
        raise ValueError(f"{symbol}: all candidate features are too sparse after filtering.")

    train_feat_filled = train_feat.copy()
    base_medians = train_feat_filled[feature_columns].median(numeric_only=True)
    train_feat_filled[feature_columns] = train_feat_filled[feature_columns].fillna(base_medians)

    train_supervised = prepare_split_for_supervised(train_feat_filled, feature_columns=feature_columns)
    medians = train_supervised[feature_columns].median()
    
    val_feat_filled = val_feat.copy()
    val_feat_filled[feature_columns] = val_feat_filled[feature_columns].fillna(medians)
    val_supervised = prepare_split_for_supervised(val_feat_filled, feature_columns=feature_columns)
    
    test_feat_filled = test_feat.copy()
    test_feat_filled[feature_columns] = test_feat_filled[feature_columns].fillna(medians)
    test_supervised = prepare_split_for_supervised(test_feat_filled, feature_columns=feature_columns)

    if min(len(train_supervised), len(val_supervised), len(test_supervised)) < window_size:
        raise ValueError(
            f"{symbol}: insufficient supervised rows after feature prep for window_size={window_size}. "
            f"Got train={len(train_supervised)}, val={len(val_supervised)}, test={len(test_supervised)}."
        )

    # Return pure untransformed, unscaled supervised feature blocks
    return train_supervised, val_supervised, test_supervised, feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GLOBAL Universe CNN-LSTM Hybrid Classifier")
    parser.add_argument("--symbol", default=None, help="Ignored in Universe mode")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs (default: 150).")
    parser.add_argument("--finetune-epochs", type=int, default=25, help="Max epochs for per-symbol fine-tuning.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (larger for global).")
    parser.add_argument("--window-size", type=int, default=20, help="Sequence window size.")
    parser.add_argument("--neutral-threshold", type=float, default=0.005, help="Neutral band threshold.")
    parser.add_argument("--use-binary", action="store_true", help="Binary mode fallback.")
    parser.add_argument("--min-neutral-ratio", type=float, default=0.15, help="Minimum neutral class ratio.")
    parser.add_argument("--max-neutral-ratio", type=float, default=0.45, help="Maximum neutral class ratio.")
    parser.add_argument("--target-neutral-ratio", type=float, default=0.25, help="Target neutral class ratio for threshold selection.")
    parser.add_argument("--min-directional-pred-rate", type=float, default=0.12, help="Minimum predicted non-neutral rate during threshold tuning.")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order, format p,d,q.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="data/models/cnn_pattern", help="Output root directory.")
    parser.add_argument("--use-nse", action="store_true", help="Use NSE fallback.")
    parser.add_argument("--interval", default="1h", help="Candle interval (e.g., 1d, 1h).")
    parser.add_argument("--local-silver-dir", default=None, help="Optional local OHLCV parquet root.")
    parser.add_argument("--min-rows", type=int, default=180, help="Minimum rows required per symbol.")
    parser.add_argument("--finetune-patience", type=int, default=8, help="Early-stopping patience for per-symbol fine-tuning.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional orchestration run identifier for artifact alignment.")
    parser.add_argument(
        "--feature-schema-version",
        type=str,
        default="technical_features_v1",
        help="Feature schema version tag persisted in training artifacts.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["fixed", "atr", "percentile"],
        default="atr",
        help="Label construction mode: fixed (legacy), atr (volatility-adjusted), percentile (distribution-balanced).",
    )
    parser.add_argument(
        "--atr-k",
        type=float,
        default=0.5,
        help="ATR multiplier k for threshold = k × ATR/close. Only used with --label-mode atr.",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR lookback period. Only used with --label-mode atr.",
    )
    parser.add_argument(
        "--symbol-atr-k-overrides",
        type=str,
        default=None,
        help='Per-symbol ATR multiplier overrides, e.g. "TATASTEEL.NS:1.0,INFY.NS:0.5".',
    )
    args = parser.parse_args()

    set_seed(args.seed)
    run_id = str(args.run_id).strip() if args.run_id else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_script = str(Path(__file__).name)
    feature_schema_version = str(args.feature_schema_version).strip()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    arima_order = tuple(int(x.strip()) for x in args.arima_order.split(","))

    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    fx_context = load_fx_context_frame(loader, local_silver_dir=args.local_silver_dir, limit=args.limit, use_nse=args.use_nse, interval=args.interval)
    logger.info("USDINR context rows available: %s", len(fx_context))
    logger.info("Universe CNN run_id=%s | interval=%s", run_id, args.interval)

    requested_symbols = dedupe_symbols(parse_symbols(args.symbol, args.symbols))

    def validate_symbol(symbol: str):
        try:
            if args.local_silver_dir:
                frame = load_symbol_from_local_silver(symbol=symbol, base_dir=args.local_silver_dir)
                if args.limit is not None:
                    frame = frame.tail(args.limit).copy()
            else:
                frame = loader.load_historical_bars(symbol=symbol, limit=args.limit, use_nse_fallback=args.use_nse, min_fallback_rows=180, interval=args.interval)
            frame = frame.sort_values("timestamp").dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
            frame = merge_usdinr_features(frame, fx_context)
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        return validate_equity_symbol(symbol=symbol, frame=frame, interval=args.interval, split_counts=quality_gate_split_counts(frame))

    discovery = discover_training_symbols(interval=args.interval, requested_symbols=requested_symbols or None, database_url=db_url, validator=validate_symbol, print_fn=lambda message: logger.info(message))
    training_symbols = list(discovery.active_symbols)
    logger.info("Symbols to train globally: %s", training_symbols)
    if not training_symbols:
        logger.error("No active equity symbols passed.")
        raise SystemExit(1)

    # --- Phase 1: Extract features per symbol ---
    symbol_data = {}
    all_train_sup = []
    global_feature_columns = None
    
    for symbol in training_symbols:
        if is_forex(symbol):
            continue
        try:
            logger.info("Extracting features for %s...", symbol)
            train_sup, val_sup, test_sup, f_cols = extract_symbol_features(symbol, discovery.frames[symbol].copy(), args.window_size, arima_order, args.min_rows)
            symbol_data[symbol] = {"train": train_sup, "val": val_sup, "test": test_sup}
            all_train_sup.append(train_sup[f_cols].copy())
            if not global_feature_columns:
                global_feature_columns = f_cols
        except Exception as exc:
            logger.error("Skipping %s due to extraction error: %s", symbol, exc)
            
    if not symbol_data:
        logger.error("No valid symbol data extracted.")
        sys.exit(1)

    # --- Phase 2: Fit Global Scaler & Thresholds ---
    global_train_df = pd.concat(all_train_sup, ignore_index=True).astype("float64")
    scaler = StandardScaler()
    scaler.fit(global_train_df.values)
    
    global_train_returns = pd.concat([symbol_data[s]["train"]["future_return"] for s in symbol_data], ignore_index=True).values
    effective_neutral_threshold, neutral_ratio = choose_neutral_threshold_unified(
        train_forward_returns=global_train_returns,
        requested_threshold=args.neutral_threshold,
        min_neutral_ratio=args.min_neutral_ratio,
        max_neutral_ratio=args.max_neutral_ratio,
        target_neutral_ratio=args.target_neutral_ratio,
    )
    logger.info("GLOBAL Neutral Threshold baseline: %.4f (train neutral share %.2f%%)", effective_neutral_threshold, neutral_ratio*100)

    # --- Phase 3: Build Windows ---
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    for symbol, splits in symbol_data.items():
        # Per-symbol thresholding
        if getattr(args, "label_mode", "fixed") == "atr":
            train_raw_df = discovery.frames[symbol].copy()
            
            k_overrides = parse_atr_overrides(getattr(args, "symbol_atr_k_overrides", None))
            symbol_k = k_overrides.get(symbol, getattr(args, "atr_k", 0.5))
            
            symbol_threshold = atr_effective_threshold(
                high=train_raw_df["high"].values,
                low=train_raw_df["low"].values,
                close=train_raw_df["close"].values,
                k=symbol_k,
                atr_period=int(getattr(args, "atr_period", 14)),
            )
            logger.info("[%s] Using ATR-scaled threshold: %.6f", symbol, symbol_threshold)
        else:
            symbol_threshold = effective_neutral_threshold

        for s_name in ["train", "val", "test"]:
            sup = splits[s_name].copy().astype({col: "float64" for col in global_feature_columns})
            sup.loc[:, global_feature_columns] = scaler.transform(sup[global_feature_columns].values)
            
            y_raw, _ = build_labels_unified(
                sup["future_return"].values, 
                threshold=symbol_threshold, 
                use_binary=args.use_binary,
                mode="fixed"
            )
            X, y, ret = make_windows(sup, global_feature_columns, y_raw, window_size=args.window_size)
            
            timestamps = sup["timestamp"].iloc[args.window_size - 1:].values if len(sup) >= args.window_size else np.empty(0)
            
            splits[f"X_{s_name}"] = X
            splits[f"y_{s_name}"] = y
            splits[f"ret_{s_name}"] = ret
            splits[f"timestamps_{s_name}"] = timestamps
            
            if s_name == "train":
                X_train_list.append(X)
                y_train_list.append(y)
            elif s_name == "val":
                X_val_list.append(X)
                y_val_list.append(y)
            else:
                X_test_list.append(X)
                y_test_list.append(y)

    X_train_global = np.concatenate(X_train_list, axis=0) if X_train_list else np.empty((0,))
    y_train_global = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,))
    X_val_global = np.concatenate(X_val_list, axis=0) if X_val_list else np.empty((0,))
    y_val_global = np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,))
    
    logger.info("GLOBAL Train shape: %s | Val shape: %s", X_train_global.shape, X_val_global.shape)

    # --- Phase 4: Train Global Model ---
    num_classes = 2 if args.use_binary else 3
    class_weight_dict, class_weight_tensor = compute_class_weights(y_train_global, num_classes=num_classes, symbol="GLOBAL", y_test=np.empty(0), use_binary=args.use_binary)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCnnLstmClassifier(num_features=len(global_feature_columns), num_classes=num_classes).to(device)
    
    global_dir = output_root / "GLOBAL"
    global_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(global_dir / "best_model_global.keras")

    logger.info("========== Training GLOBAL Base Model ==========")
    train_stats = train_model(
        model=model, X_train=X_train_global, y_train=y_train_global, X_val=X_val_global, y_val=y_val_global,
        class_weights=class_weight_tensor, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        early_stopping_patience=15, checkpoint_path=checkpoint_path, device=device
    )

    probs_val_global = predict_probabilities(model, X_val_global, device=device, batch_size=args.batch_size)
    best_threshold, best_val_tune_score, best_val_directional_rate = tune_threshold(
        y_val=y_val_global,
        probs_val=probs_val_global,
        use_binary=args.use_binary,
        min_directional_pred_rate=args.min_directional_pred_rate,
    )
    logger.info(
        "GLOBAL Best Threshold: %.2f | Val tune score: %.4f | Val predicted directional rate: %.2f%%",
        best_threshold,
        best_val_tune_score,
        best_val_directional_rate * 100.0,
    )

    safe_torch_save(model, global_dir / "hybrid_cnn_lstm_weights.pt")
    with open(global_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    hyperparams = {
        "symbol": "GLOBAL",
        "symbol_canonical": "GLOBAL",
        "run_id": run_id,
        "interval": args.interval,
        "source_script": source_script,
        "feature_schema_version": feature_schema_version,
        "trained_on": list(symbol_data.keys()),
        "window_size": args.window_size,
        "learning_rate": args.lr,
        "feature_columns": global_feature_columns,
        "neutral_threshold": effective_neutral_threshold,
        "use_binary": args.use_binary,
        "confidence_threshold": best_threshold,
        "min_directional_pred_rate": args.min_directional_pred_rate,
        "target_neutral_ratio": args.target_neutral_ratio,
        "selected_neutral_ratio": neutral_ratio,
        "threshold_tune_score": best_val_tune_score,
        "val_predicted_directional_rate": best_val_directional_rate,
        "class_weight_dict": class_weight_dict,
    }
    safe_write_json(global_dir / "hyperparams.json", hyperparams)
    global_state_dict = clone_state_dict(model)
    global_scaler_path = global_dir / "feature_scaler.pkl"

    # --- Phase 5: Evaluate Per Symbol ---
    logger.info("========== Evaluating Per Symbol ==========")
    for symbol, splits in symbol_data.items():
        sym_dir = output_root / sanitize_symbol(symbol)
        sym_dir.mkdir(parents=True, exist_ok=True)

        symbol_model = HybridCnnLstmClassifier(num_features=len(global_feature_columns), num_classes=num_classes).to(device)
        symbol_model.load_state_dict(global_state_dict)
        symbol_checkpoint_path = str(sym_dir / f"best_model_{sanitize_symbol(symbol)}.keras")
        symbol_threshold = float(best_threshold)
        symbol_val_tune_score = float("nan")
        symbol_val_directional_rate = 0.0
        fine_tune_applied = False
        symbol_train_stats: Dict[str, float] = {}
        symbol_class_weight_dict = {int(k): float(v) for k, v in class_weight_dict.items()}

        X_train_sym = splits.get("X_train")
        y_train_sym = splits.get("y_train")
        X_val_sym = splits.get("X_val")
        y_val_sym = splits.get("y_val")
        X_test_sym = splits.get("X_test")
        y_test_sym = splits.get("y_test")

        can_finetune = (
            isinstance(X_train_sym, np.ndarray)
            and isinstance(y_train_sym, np.ndarray)
            and isinstance(X_val_sym, np.ndarray)
            and isinstance(y_val_sym, np.ndarray)
            and len(X_train_sym) > 0
            and len(y_train_sym) > 0
            and len(X_val_sym) > 0
            and len(y_val_sym) > 0
            and len(np.unique(y_train_sym)) >= 2
        )

        if can_finetune:
            symbol_class_weight_dict, symbol_class_weight_tensor = compute_class_weights(
                y_train_sym,
                num_classes=num_classes,
                symbol=symbol,
                y_test=y_test_sym if isinstance(y_test_sym, np.ndarray) else np.empty(0),
                use_binary=args.use_binary,
            )
            symbol_train_stats = train_model(
                model=symbol_model,
                X_train=X_train_sym,
                y_train=y_train_sym,
                X_val=X_val_sym,
                y_val=y_val_sym,
                class_weights=symbol_class_weight_tensor,
                lr=args.lr,
                epochs=max(1, int(args.finetune_epochs)),
                batch_size=args.batch_size,
                early_stopping_patience=max(1, int(args.finetune_patience)),
                checkpoint_path=symbol_checkpoint_path,
                device=device,
            )
            probs_val_symbol = predict_probabilities(symbol_model, X_val_sym, device=device, batch_size=args.batch_size)
            if len(probs_val_symbol) > 0:
                symbol_threshold, symbol_val_tune_score, symbol_val_directional_rate = tune_threshold(
                    y_val=y_val_sym,
                    probs_val=probs_val_symbol,
                    use_binary=args.use_binary,
                    min_directional_pred_rate=args.min_directional_pred_rate,
                )
            fine_tune_applied = True
        else:
            logger.warning(
                "[%s] Skipping fine-tune (insufficient train/val windows or single-class train labels). Using GLOBAL base model.",
                symbol,
            )

        sym_dfs = []
        test_acc = 0.0
        trade_stats = {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
        direction_stats = {
            "up": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "down": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            "directional_accuracy": 0.0,
        }
        test_confusion_matrix: List[List[int]] = []
        
        for s_name in ["train", "val", "test"]:
            X_data = splits.get(f"X_{s_name}")
            y_data = splits.get(f"y_{s_name}")
            ret_data = splits.get(f"ret_{s_name}")
            timestamps = splits.get(f"timestamps_{s_name}")
            
            if X_data is None or len(X_data) == 0:
                continue
                
            probs = predict_probabilities(symbol_model, X_data, device=device, batch_size=args.batch_size)
            
            if s_name == "test":
                preds_test = apply_confidence_threshold(probs, threshold=symbol_threshold, use_binary=args.use_binary)
                test_acc = float(accuracy_score(y_data, preds_test))
                trade_stats = trading_metrics(preds_test, ret_data, use_binary=args.use_binary, interval=args.interval)
                direction_stats = directional_class_metrics(y_true=y_data, y_pred=preds_test, use_binary=args.use_binary)
                test_labels = [0, 1] if args.use_binary else [0, 1, 2]
                test_confusion_matrix = confusion_matrix(y_data, preds_test, labels=test_labels).tolist()
                logger.info("[%s] Test Acc: %.4f | Sharpe: %.4f | WinRate: %.4f", symbol, test_acc, trade_stats["sharpe"], trade_stats["win_rate"])
                logger.info(
                    "[%s] Directional metrics | up(P/R): %.4f / %.4f | down(P/R): %.4f / %.4f",
                    symbol,
                    direction_stats["up"]["precision"],
                    direction_stats["up"]["recall"],
                    direction_stats["down"]["precision"],
                    direction_stats["down"]["recall"],
                )
                
            prob_df = pd.DataFrame({
                "timestamp": timestamps,
                "split": s_name,
                "cnn_prob_up": probs[:, 0],
                "cnn_prob_neutral": probs[:, 1] if not args.use_binary else 0.0,
                "cnn_prob_down": probs[:, 2] if not args.use_binary else probs[:, 1],
            })
            sym_dfs.append(prob_df)
            
        if sym_dfs:
            sym_prob_df = pd.concat(sym_dfs, ignore_index=True)
            sym_dir.mkdir(parents=True, exist_ok=True)
            sym_prob_df.to_parquet(sym_dir / "predictions.parquet", index=False)

        sym_dir.mkdir(parents=True, exist_ok=True)
        safe_torch_save(symbol_model, sym_dir / "hybrid_cnn_lstm_weights.pt")
        safe_link_or_copy(global_scaler_path, sym_dir / "feature_scaler.pkl")

        symbol_hyperparams = {
            "symbol": symbol,
            "symbol_canonical": symbol,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": source_script,
            "feature_schema_version": feature_schema_version,
            "trained_as_part_of_universe": True,
            "base_model_symbol": "GLOBAL",
            "window_size": args.window_size,
            "learning_rate": args.lr,
            "feature_columns": global_feature_columns,
            "neutral_threshold": effective_neutral_threshold,
            "use_binary": args.use_binary,
            "confidence_threshold": symbol_threshold,
            "class_weight_dict": symbol_class_weight_dict,
                "finetune": {
                    "applied": fine_tune_applied,
                    "epochs_requested": int(args.finetune_epochs),
                    "patience": int(args.finetune_patience),
                    "best_val_tune_score": symbol_val_tune_score,
                    "val_predicted_directional_rate": symbol_val_directional_rate,
                },
            }
        safe_write_json(sym_dir / "hyperparams.json", symbol_hyperparams)

        split_counts = {
            "train": int(len(X_train_sym)) if isinstance(X_train_sym, np.ndarray) else 0,
            "val": int(len(X_val_sym)) if isinstance(X_val_sym, np.ndarray) else 0,
            "test": int(len(X_test_sym)) if isinstance(X_test_sym, np.ndarray) else 0,
        }
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "symbol_canonical": symbol,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": source_script,
            "feature_schema_version": feature_schema_version,
            "trained_as_part_of_universe": True,
            "hyperparameters": {
                "window_size": args.window_size,
                "learning_rate": args.lr,
                "epochs_requested": int(args.finetune_epochs),
                "confidence_threshold": symbol_threshold,
                "use_binary": args.use_binary,
            },
            "split_counts": split_counts,
            "metrics": {
                "test_accuracy": test_acc,
                "sharpe": trade_stats["sharpe"],
                "max_drawdown": trade_stats["max_drawdown"],
                "win_rate": trade_stats["win_rate"],
                "post_cost_sharpe": post_cost_sharpe(y_data, preds_test, ret_data) if not args.use_binary else trade_stats["sharpe"],
                "recall_balance": direction_stats["recall_balance"],
                "directional_coverage": direction_stats["directional_coverage"],
                "up_precision": direction_stats["up"]["precision"],
                "up_recall": direction_stats["up"]["recall"],
                "up_f1": direction_stats["up"]["f1"],
                "up_support": direction_stats["up"]["support"],
                "down_precision": direction_stats["down"]["precision"],
                "down_recall": direction_stats["down"]["recall"],
                "down_f1": direction_stats["down"]["f1"],
                "down_support": direction_stats["down"]["support"],
                "directional_accuracy": direction_stats["directional_accuracy"],
                "test_confusion_matrix": test_confusion_matrix,
                "finetune_best_train_loss": symbol_train_stats.get("best_train_loss"),
                "finetune_best_val_loss": symbol_train_stats.get("best_val_loss"),
                "finetune_best_train_acc": symbol_train_stats.get("best_train_acc"),
                "finetune_best_val_acc": symbol_train_stats.get("best_val_acc"),
                "finetune_epochs_run": symbol_train_stats.get("epochs_run"),
                "finetune_best_val_tune_score": symbol_val_tune_score,
                "finetune_val_predicted_directional_rate": symbol_val_directional_rate,
            },
        }
        safe_write_json(sym_dir / "training_meta.json", meta)

    logger.info("Universe CNN training pipeline complete.")

if __name__ == "__main__":
    main()
