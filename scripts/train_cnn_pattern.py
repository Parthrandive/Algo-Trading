import argparse
import json
import logging
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
from config.symbols import (
    FX_RESULTS_NOTE,
    FOREX_SYMBOLS,
    SplitCounts,
    SymbolValidationResult,
    dedupe_symbols,
    discover_training_symbols,
    is_forex,
    validate_equity_symbol,
)

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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)


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


def choose_neutral_threshold(
    train_forward_returns: np.ndarray,
    requested_threshold: float,
    use_binary: bool,
    min_neutral_ratio: float,
) -> Tuple[float, float]:
    if use_binary:
        return requested_threshold, 0.0

    candidates = sorted({requested_threshold, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02})
    best_threshold = requested_threshold
    best_ratio = 0.0
    target_ratio = 0.175

    for threshold in candidates:
        neutral_ratio = float((np.abs(train_forward_returns) <= threshold).mean())
        if abs(neutral_ratio - target_ratio) < abs(best_ratio - target_ratio):
            best_threshold = threshold
            best_ratio = neutral_ratio
        if neutral_ratio >= min_neutral_ratio:
            return threshold, neutral_ratio

    return best_threshold, best_ratio


def build_labels(forward_returns: np.ndarray, threshold: float, use_binary: bool) -> np.ndarray:
    if use_binary:
        # Binary fallback mode: 0=up, 1=down
        return np.where(forward_returns >= 0.0, 0, 1).astype(int)

    # Multi-class mode: 0=up, 1=neutral, 2=down
    labels = np.ones(len(forward_returns), dtype=int)
    labels[forward_returns > threshold] = 0
    labels[forward_returns < -threshold] = 2
    return labels


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


def class_distribution(y: np.ndarray, use_binary: bool) -> Dict[str, float]:
    if len(y) == 0:
        if use_binary:
            return {"up": 0.0, "down": 0.0}
        return {"up": 0.0, "neutral": 0.0, "down": 0.0}

    if use_binary:
        counts = np.bincount(y, minlength=2)
        total = counts.sum()
        return {"up": float(counts[0] / total * 100), "down": float(counts[1] / total * 100)}

    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    return {
        "up": float(counts[0] / total * 100),
        "neutral": float(counts[1] / total * 100),
        "down": float(counts[2] / total * 100),
    }


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

    # Force directional stance by boosting UP and DOWN, and suppressing NEUTRAL
    class_weight_dict[up_idx] = float(class_weight_dict.get(up_idx, 1.0) * 3.0)
    class_weight_dict[down_idx] = float(class_weight_dict.get(down_idx, 1.0) * 2.0)
    if neutral_idx is not None:
        class_weight_dict[neutral_idx] = float(class_weight_dict.get(neutral_idx, 1.0) * 0.5)

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


def tune_threshold(y_val: np.ndarray, probs_val: np.ndarray, use_binary: bool) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.3, 0.8, 0.05):
        preds = apply_confidence_threshold(probs_val, float(threshold), use_binary=use_binary)
        score = float(f1_score(y_val, preds, average="weighted"))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def label_to_signal(labels: np.ndarray, use_binary: bool) -> np.ndarray:
    if use_binary:
        # 0=up, 1=down
        return np.where(labels == 0, 1.0, -1.0).astype(np.float32)
    # 0=up, 1=neutral, 2=down
    signals = np.zeros(len(labels), dtype=np.float32)
    signals[labels == 0] = 1.0
    signals[labels == 2] = -1.0
    return signals


def trading_metrics(pred_labels: np.ndarray, realized_returns: np.ndarray, use_binary: bool) -> Dict[str, float]:
    if len(pred_labels) == 0:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "win_rate": float("nan")}

    signals = label_to_signal(pred_labels, use_binary=use_binary)
    strategy_returns = signals * realized_returns
    mean_ret = float(np.mean(strategy_returns))
    vol = float(np.std(strategy_returns))
    sharpe = float((mean_ret / vol) * np.sqrt(252.0)) if vol > 0 else float("nan")

    equity = np.cumprod(1.0 + strategy_returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peaks, EPS) - 1.0
    max_drawdown = float(np.min(drawdown))

    active = strategy_returns[np.abs(signals) > 0]
    if len(active) == 0:
        win_rate = float("nan")
    else:
        win_rate = float(np.mean(active > 0))

    return {"sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate}


def format_distribution(dist: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.2f}%" for k, v in dist.items())

def train_single_symbol(
    symbol: str,
    df: pd.DataFrame,
    output_root: Path,
    window_size: int,
    neutral_threshold: float,
    use_binary: bool,
    min_neutral_ratio: float,
    arima_order: Tuple[int, int, int],
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    min_rows: int,
    include_daily_features: bool,
) -> SymbolTrainingResult:
    set_seed(seed)
    validate_data(df, symbol=symbol, min_rows=min_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_df_raw, val_df_raw, test_df_raw = chronological_split(df)
    train_len = len(train_df_raw)

    feat_df = engineer_features(df, include_daily_features=include_daily_features)
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

    train_supervised = prepare_split_for_supervised(train_feat, feature_columns=feature_columns)
    medians = train_supervised[feature_columns].median()
    
    val_feat_filled = val_feat.copy()
    val_feat_filled[feature_columns] = val_feat_filled[feature_columns].fillna(medians)
    val_supervised = prepare_split_for_supervised(val_feat_filled, feature_columns=feature_columns)
    
    test_feat_filled = test_feat.copy()
    test_feat_filled[feature_columns] = test_feat_filled[feature_columns].fillna(medians)
    test_supervised = prepare_split_for_supervised(test_feat_filled, feature_columns=feature_columns)

    if min(len(train_supervised), len(val_supervised), len(test_supervised)) < window_size:
        print("DEBUG: val_feat_filled NaNs:", val_feat_filled[feature_columns + ["close"]].isna().sum().to_dict())
        out_test = val_feat_filled.copy()
        out_test["future_return"] = out_test["close"].pct_change().shift(-1)
        print("DEBUG: future_return NaNs:", out_test["future_return"].isna().sum())
        raise ValueError(
            f"{symbol}: insufficient supervised rows after feature prep for window_size={window_size}. "
            f"Got train={len(train_supervised)}, val={len(val_supervised)}, test={len(test_supervised)}."
        )

    effective_neutral_threshold, neutral_ratio = choose_neutral_threshold(
        train_forward_returns=train_supervised["future_return"].values,
        requested_threshold=neutral_threshold,
        use_binary=use_binary,
        min_neutral_ratio=min_neutral_ratio,
    )
    if not use_binary:
        logger.info(
            "%s neutral threshold selected=%.4f (train neutral share %.2f%%)",
            symbol,
            effective_neutral_threshold,
            neutral_ratio * 100,
        )

    y_train_raw = build_labels(
        train_supervised["future_return"].values, threshold=effective_neutral_threshold, use_binary=use_binary
    )
    y_val_raw = build_labels(val_supervised["future_return"].values, threshold=effective_neutral_threshold, use_binary=use_binary)
    y_test_raw = build_labels(
        test_supervised["future_return"].values, threshold=effective_neutral_threshold, use_binary=use_binary
    )

    # Ensure numeric feature block is float before scaling to avoid dtype assignment issues.
    train_supervised = train_supervised.copy().astype({col: "float64" for col in feature_columns})
    val_supervised = val_supervised.copy().astype({col: "float64" for col in feature_columns})
    test_supervised = test_supervised.copy().astype({col: "float64" for col in feature_columns})

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_supervised[feature_columns].values)
    val_scaled = scaler.transform(val_supervised[feature_columns].values)
    test_scaled = scaler.transform(test_supervised[feature_columns].values)
    train_supervised.loc[:, feature_columns] = train_scaled
    val_supervised.loc[:, feature_columns] = val_scaled
    test_supervised.loc[:, feature_columns] = test_scaled

    X_train, y_train, _ = make_windows(train_supervised, feature_columns, y_train_raw, window_size=window_size)
    X_val, y_val, _ = make_windows(val_supervised, feature_columns, y_val_raw, window_size=window_size)
    X_test, y_test, ret_test = make_windows(test_supervised, feature_columns, y_test_raw, window_size=window_size)

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            f"{symbol}: no windows generated after preprocessing. "
            f"window_size={window_size}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

    num_classes = 2 if use_binary else 3
    class_weight_dict, class_weight_tensor = compute_class_weights(
        y_train,
        num_classes=num_classes,
        symbol=symbol,
        y_test=y_test,
        use_binary=use_binary,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCnnLstmClassifier(num_features=len(feature_columns), num_classes=num_classes).to(device)

    symbol_dir = output_root / sanitize_symbol(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(symbol_dir / f"best_model_{sanitize_symbol(symbol)}.keras")

    train_stats = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weight_tensor,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=15,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    probs_val = predict_probabilities(model, X_val, device=device, batch_size=batch_size)
    best_threshold, best_val_f1 = tune_threshold(y_val=y_val, probs_val=probs_val, use_binary=use_binary)
    logger.info("%s Best threshold: %.2f, Val F1: %.4f", symbol, best_threshold, best_val_f1)

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    train_loss = evaluate_loss(model, X_train, y_train, criterion, device=device, batch_size=batch_size)
    val_loss = evaluate_loss(model, X_val, y_val, criterion, device=device, batch_size=batch_size)

    probs_train = predict_probabilities(model, X_train, device=device, batch_size=batch_size)
    probs_test = predict_probabilities(model, X_test, device=device, batch_size=batch_size)
    preds_train = apply_confidence_threshold(probs_train, threshold=best_threshold, use_binary=use_binary)
    preds_val = apply_confidence_threshold(probs_val, threshold=best_threshold, use_binary=use_binary)
    preds_test = apply_confidence_threshold(probs_test, threshold=best_threshold, use_binary=use_binary)

    train_acc = float(accuracy_score(y_train, preds_train))
    val_acc = float(accuracy_score(y_val, preds_val))
    test_acc = float(accuracy_score(y_test, preds_test))

    trade_stats = trading_metrics(pred_labels=preds_test, realized_returns=ret_test, use_binary=use_binary)
    labels_range = [0, 1] if use_binary else [0, 1, 2]
    cm = confusion_matrix(y_test, preds_test, labels=labels_range)

    train_dist = class_distribution(y_train, use_binary=use_binary)
    val_dist = class_distribution(y_val, use_binary=use_binary)
    test_dist = class_distribution(y_test, use_binary=use_binary)

    logger.info("===== %s Validation Checklist =====", symbol)
    logger.info("[OK] Train/Val/Test sample counts: %s / %s / %s", len(X_train), len(X_val), len(X_test))
    logger.info("[OK] Class distribution train: %s", format_distribution(train_dist))
    logger.info("[OK] Class distribution val: %s", format_distribution(val_dist))
    logger.info("[OK] Class distribution test: %s", format_distribution(test_dist))
    logger.info(
        "[OK] Train/Val/Test accuracy: %.4f / %.4f / %.4f",
        train_acc,
        val_acc,
        test_acc,
    )
    logger.info("[OK] Train loss / Val loss: %.4f / %.4f", train_loss, val_loss)
    logger.info(
        "[OK] Sharpe / Max Drawdown / Win Rate: %.4f / %.4f / %.4f",
        trade_stats["sharpe"],
        trade_stats["max_drawdown"],
        trade_stats["win_rate"],
    )
    logger.info("[OK] Total epochs trained: %s", train_stats["epochs_run"])
    logger.info("[OK] Confusion matrix (test):\n%s", cm)

    if train_acc < val_acc:
        logger.warning(
            "%s train accuracy (%.4f) is below val accuracy (%.4f). Re-check regularization/data variance.",
            symbol,
            train_acc,
            val_acc,
        )

    torch.save(model.state_dict(), symbol_dir / "hybrid_cnn_lstm_weights.pt")
    with open(symbol_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    hyperparams = {
        "symbol": symbol,
        "window_size": window_size,
        "learning_rate": lr,
        "feature_columns": feature_columns,
        "include_daily_features": include_daily_features,
        "neutral_threshold": effective_neutral_threshold,
        "use_binary": use_binary,
        "confidence_threshold": best_threshold,
        "class_weight_dict": class_weight_dict,
    }
    with open(symbol_dir / "hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)

    training_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "hyperparameters": hyperparams,
        "epochs_requested": epochs,
        "epochs_run": train_stats["epochs_run"],
        "split_counts": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "class_distribution_pct": {"train": train_dist, "val": val_dist, "test": test_dist},
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "best_val_f1": best_val_f1,
            "sharpe": trade_stats["sharpe"],
            "max_drawdown": trade_stats["max_drawdown"],
            "win_rate": trade_stats["win_rate"],
            "confusion_matrix": cm.tolist(),
        },
    }
    # Save training_meta.json to persistent lightweight location
    persistent_meta_dir = Path("data/models/cnn_pattern") / sanitize_symbol(symbol)
    persistent_meta_dir.mkdir(parents=True, exist_ok=True)
    with open(persistent_meta_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    return SymbolTrainingResult(
        symbol=symbol,
        neutral_threshold=effective_neutral_threshold,
        best_threshold=best_threshold,
        epochs_run=int(train_stats["epochs_run"]),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        train_acc=float(train_acc),
        val_acc=float(val_acc),
        test_acc=float(test_acc),
        sharpe=float(trade_stats["sharpe"]),
        max_drawdown=float(trade_stats["max_drawdown"]),
        win_rate=float(trade_stats["win_rate"]),
        class_weight_dict=class_weight_dict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-symbol ARIMA-LSTM-CNN hybrid regime classifier with strict chronological splits and leakage-safe preprocessing."
        )
    )
    parser.add_argument("--symbol", default=None, help="Single symbol override (e.g., TATASTEEL.NS).")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to train (ignored if --symbol is set).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to fetch per symbol.")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs (default: 150).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--window-size", type=int, default=20, help="Sequence window size.")
    parser.add_argument("--neutral-threshold", type=float, default=0.005, help="Neutral band threshold (e.g., 0.005 or 0.01).")
    parser.add_argument("--use-binary", action="store_true", help="Binary mode fallback: up/down only.")
    parser.add_argument("--min-neutral-ratio", type=float, default=0.15, help="Minimum neutral class ratio target in multiclass mode.")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order for hybrid features, format p,d,q.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="/tmp/cnn_pattern/", help="Output root directory (defaults to /tmp to avoid disk bloat).")
    parser.add_argument("--use-nse", action="store_true", help="Use NSE fallback when DB data is sparse.")
    parser.add_argument("--interval", default="1d", help="Candle interval (e.g., 1d, 1h).")
    parser.add_argument(
        "--disable-daily-features",
        action="store_true",
        help="Disable daily timeframe feature fusion onto intraday rows.",
    )
    parser.add_argument(
        "--local-silver-dir",
        default=None,
        help="Optional local OHLCV parquet root (e.g., data/silver/ohlcv) to bypass DB/NSE.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=180,
        help="Minimum rows required per symbol before training.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    include_daily_features = not bool(args.disable_daily_features)
    logger.info(
        "Daily feature fusion: %s",
        "enabled" if include_daily_features else "disabled",
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    arima_order = tuple(int(x.strip()) for x in args.arima_order.split(","))

    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    fx_context = load_fx_context_frame(
        loader,
        local_silver_dir=args.local_silver_dir,
        limit=args.limit,
        use_nse=args.use_nse,
        interval=args.interval,
    )
    logger.info("USDINR context rows available: %s", len(fx_context))

    requested_symbols = dedupe_symbols(parse_symbols(args.symbol, args.symbols))

    def validate_symbol(symbol: str):
        try:
            if args.local_silver_dir:
                frame = load_symbol_from_local_silver(symbol=symbol, base_dir=args.local_silver_dir)
                if args.limit is not None:
                    frame = frame.tail(args.limit).copy()
            else:
                frame = loader.load_historical_bars(
                    symbol=symbol,
                    limit=args.limit,
                    use_nse_fallback=args.use_nse,
                    min_fallback_rows=180,
                    interval=args.interval,
                )
            frame = frame.sort_values("timestamp").dropna(
                subset=["timestamp", "open", "high", "low", "close", "volume"]
            ).reset_index(drop=True)
            frame = merge_usdinr_features(frame, fx_context)
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        return validate_equity_symbol(
            symbol=symbol,
            frame=frame,
            interval=args.interval,
            split_counts=quality_gate_split_counts(frame),
        )

    discovery = discover_training_symbols(
        interval=args.interval,
        requested_symbols=requested_symbols or None,
        database_url=db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    training_symbols = list(discovery.active_symbols)
    logger.info("Symbols to train: %s", training_symbols)
    logger.info("Mode: %s", "binary(up/down)" if args.use_binary else "multiclass(up/neutral/down)")
    if not training_symbols:
        logger.error("No active equity symbols passed the training quality gate.")
        raise SystemExit(1)

    results: List[SymbolTrainingResult] = []
    failures: List[Tuple[str, str]] = []

    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
        logger.info("========== Training %s ==========", symbol)
        try:
            df = discovery.frames[symbol].copy()
            result = train_single_symbol(
                symbol=symbol,
                df=df,
                output_root=output_root,
                window_size=args.window_size,
                neutral_threshold=args.neutral_threshold,
                use_binary=args.use_binary,
                min_neutral_ratio=args.min_neutral_ratio,
                arima_order=arima_order,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                min_rows=args.min_rows,
                include_daily_features=include_daily_features,
            )
            results.append(result)
        except Exception as exc:
            failures.append((symbol, str(exc)))
            logger.exception("Training failed for %s: %s", symbol, exc)

    if results:
        mean_test_acc = float(np.nanmean([r.test_acc for r in results]))
        mean_sharpe = float(np.nanmean([r.sharpe for r in results]))
        logger.info("===== Aggregate Summary =====")
        logger.info("Mean Test Accuracy: %.4f", mean_test_acc)
        logger.info("Mean Sharpe: %.4f", mean_sharpe)
        for r in results:
            logger.info(
                "%s -> test_acc=%.4f sharpe=%.4f win_rate=%.4f neutral_th=%.4f conf_th=%.2f epochs=%s",
                r.symbol,
                r.test_acc,
                r.sharpe,
                r.win_rate,
                r.neutral_threshold,
                r.best_threshold,
                r.epochs_run,
            )

    if failures:
        for symbol, message in failures:
            logger.error("%s failed: %s", symbol, message)
        sys.exit(1)

    logger.info(FX_RESULTS_NOTE)

    logger.info("Hybrid regime training pipeline complete.")


if __name__ == "__main__":
    main()
