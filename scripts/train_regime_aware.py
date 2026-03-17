from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.regime.data_loader import RegimeDataLoader
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features


REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}
LOGGER = logging.getLogger("train_regime_aware")
CLASS_NAMES = {0: "up", 1: "neutral", 2: "down"}
DEFAULT_META_COLUMNS = {
    "symbol",
    "timestamp",
    "snapshot_id",
    "interval",
    "exchange",
    "source_type",
    "quality_status",
    "schema_version",
    "ingestion_timestamp_utc",
    "ingestion_timestamp_ist",
    "item_type",
    "author",
    "publisher",
    "platform",
    "url",
    "headline",
    "content",
    "source_id",
    "source_name",
}


@dataclass
class SequenceDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_timestamps: list[str]
    val_timestamps: list[str]


@dataclass
class RegressionArtifacts:
    metrics: dict[str, Any]
    history: list[dict[str, float]]
    prediction_frame: pd.DataFrame


@dataclass
class ClassificationArtifacts:
    metrics: dict[str, Any]
    history: list[dict[str, float]]
    prediction_frame: pd.DataFrame
    confusion: np.ndarray


class ResidualLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden)


class DirectionCNNClassifier(nn.Module):
    def __init__(self, time_steps: int, num_features: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, time_steps, num_features)
            flattened = self.features(dummy).reshape(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified regime-aware trainer for ARIMA-LSTM regression and CNN direction classification."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["USDINR=X"],
        help="One or more symbols to train, e.g. USDINR=X POWERGRID.NS LT.NS",
    )
    parser.add_argument("--data-path", default=None, help="Optional CSV/Parquet file containing the training dataset.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="Optional override for DATABASE_URL.")
    parser.add_argument("--gold-dir", default="data/gold", help="Gold parquet fallback directory.")
    parser.add_argument("--interval", default="1d", help="Candle interval to use when falling back to OHLCV DB loader.")
    parser.add_argument("--limit", type=int, default=4000, help="Maximum rows to load per symbol.")
    parser.add_argument("--output-root", default="data/reports/training_runs", help="Directory for run artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu.")

    parser.add_argument("--epochs", type=int, default=150, help="Maximum epochs. Recommended 100-200.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=10, help="ReduceLROnPlateau patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization via AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate between 0.2 and 0.3.")

    parser.add_argument("--reg-window", type=int, default=20, help="Sequence length for ARIMA-LSTM residual model.")
    parser.add_argument("--cnn-window", type=int, default=30, help="Sequence length for CNN classifier.")
    parser.add_argument("--neutral-threshold", type=float, default=0.0045, help="Neutral band threshold in decimal returns.")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order as p,d,q.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train ratio.")
    parser.add_argument("--min-rows", type=int, default=300, help="Minimum usable rows after cleaning.")
    return parser.parse_args()


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_").replace(".", "_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def save_json(path: Path, payload: Any) -> None:
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


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def plot_history(history: list[dict[str, float]], metric_names: Iterable[str], path: Path, title: str) -> None:
    if plt is None or not history:
        return

    metric_names = list(metric_names)
    if not metric_names:
        return

    plt.figure(figsize=(10, 6))
    epochs = [int(row["epoch"]) for row in history]
    for metric_name in metric_names:
        values = [row.get(metric_name) for row in history]
        if all(value is None for value in values):
            continue
        plt.plot(epochs, values, label=metric_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(confusion: np.ndarray, path: Path) -> None:
    if plt is None:
        return

    labels = [CLASS_NAMES[idx] for idx in sorted(CLASS_NAMES)]
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(
                j,
                i,
                str(confusion[i, j]),
                horizontalalignment="center",
                color="white" if confusion[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def load_symbol_frame(symbol: str, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.data_path:
        source_path = Path(args.data_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Input data path does not exist: {source_path}")
        if source_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source_path)
        else:
            frame = pd.read_csv(source_path)
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] == symbol].copy()
        source_name = f"file:{source_path}"
    else:
        gold_loader = RegimeDataLoader(database_url=args.database_url, gold_dir=args.gold_dir)
        frame = gold_loader.load_features(symbol=symbol, limit=args.limit)
        source_name = "gold_features"

        if frame.empty or not REQUIRED_OHLCV.issubset(frame.columns):
            raw_loader = DataLoader(args.database_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"))
            frame = raw_loader.load_historical_bars(
                symbol=symbol,
                limit=args.limit,
                use_nse_fallback=False,
                interval=args.interval,
                include_macro=True,
            )
            source_name = "ohlcv_bars"

    if frame.empty:
        raise ValueError(f"No data found for {symbol}.")

    frame = frame.copy()
    if "timestamp" not in frame.columns:
        raise ValueError(f"Dataset for {symbol} must include a timestamp column.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

    missing_ohlcv = REQUIRED_OHLCV - set(frame.columns)
    if missing_ohlcv:
        raise ValueError(f"Dataset for {symbol} is missing required columns: {sorted(missing_ohlcv)}")

    for column in REQUIRED_OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    if len(frame) < args.min_rows:
        raise ValueError(f"Only {len(frame)} rows available for {symbol}; need at least {args.min_rows}.")

    is_forex = symbol.endswith("=X")
    frame = engineer_features(frame, is_forex=is_forex)
    frame["symbol"] = symbol
    frame["daily_return"] = frame["close"].pct_change()
    frame["log_close"] = np.log(frame["close"].clip(lower=1e-8))
    return frame.reset_index(drop=True), source_name


def build_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    feature_base = frame.drop(columns=[column for column in ["log_close", "daily_return"] if column in frame.columns], errors="ignore")

    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for column in feature_base.columns:
        if column in DEFAULT_META_COLUMNS:
            continue
        if column == "close":
            numeric_columns.append(column)
            continue

        series = feature_base[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        if pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column)
            feature_base[column] = series.astype(float)
            continue

        distinct = series.dropna().astype(str).nunique()
        if 1 < distinct <= 12:
            categorical_columns.append(column)

    encoded = pd.DataFrame(index=feature_base.index)
    if categorical_columns:
        encoded = pd.get_dummies(
            feature_base[categorical_columns].fillna("missing").astype(str),
            prefix=categorical_columns,
            dtype=float,
        )

    numeric_frame = feature_base[numeric_columns].copy()
    numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")

    all_features = pd.concat([numeric_frame, encoded], axis=1)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.loc[:, all_features.nunique(dropna=False) > 1]
    return all_features


def select_regression_columns(feature_table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in feature_table.columns:
        lowered = column.lower()
        if lowered in {"log_close", "daily_return"}:
            continue
        if any(token in lowered for token in ("target", "future", "forecast", "predicted", "prediction")):
            continue
        columns.append(column)
    return columns


def select_cnn_columns(feature_table: pd.DataFrame) -> list[str]:
    preferred_prefixes = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "macd",
        "macro",
        "cpi_",
        "wpi_",
        "iip_",
        "fii_",
        "dii_",
        "fx_reserves",
        "india_us_10y_spread",
        "repo_rate",
        "us_10y",
        "rbi_bulletin",
        "close_lag",
        "close_roll",
        "rolling_vol",
        "volatility",
        "garch",
        "var_",
        "es_",
        "quantile",
        "regime",
        "pearl",
        "risk",
        "ood",
    )

    selected = [
        column
        for column in feature_table.columns
        if column.lower().startswith(preferred_prefixes)
        and not any(token in column.lower() for token in ("target", "future", "forecast", "predicted", "prediction"))
    ]

    if "close" not in selected and "close" in feature_table.columns:
        selected.append("close")
    if "volume" not in selected and "volume" in feature_table.columns:
        selected.append("volume")
    return sorted(dict.fromkeys(selected))


def impute_and_scale_features(feature_frame: pd.DataFrame, split_idx: int) -> tuple[np.ndarray, StandardScaler, dict[str, float]]:
    train_features = feature_frame.iloc[:split_idx].copy()
    medians = train_features.median(numeric_only=True)
    medians = medians.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    filled = feature_frame.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(filled.iloc[:split_idx].values)
    scaled = scaler.transform(filled.values).astype(np.float32)
    return scaled, scaler, {column: float(value) for column, value in medians.items()}


def compute_arima_predictions(
    log_close: pd.Series,
    split_idx: int,
    arima_order: tuple[int, int, int],
) -> tuple[Any, np.ndarray]:
    train_series = log_close.iloc[:split_idx].astype(float).values
    if len(train_series) < 30:
        raise ValueError("Need at least 30 training points to fit ARIMA.")

    model = sm.tsa.ARIMA(
        train_series,
        order=arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit()

    baseline = np.full(len(log_close), np.nan, dtype=np.float64)
    train_pred = np.asarray(result.predict(start=0, end=split_idx - 1), dtype=np.float64)
    baseline[:split_idx] = train_pred

    rolling_result = result
    for idx in range(split_idx, len(log_close)):
        forecast = np.asarray(rolling_result.forecast(steps=1), dtype=np.float64).reshape(-1)
        baseline[idx] = float(forecast[0])
        next_value = float(log_close.iloc[idx])
        try:
            rolling_result = rolling_result.append([next_value], refit=False)
        except Exception:
            history = log_close.iloc[: idx + 1].astype(float).values
            rolling_result = sm.tsa.ARIMA(
                history,
                order=arima_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()

    return result, baseline


def build_regression_sequences(
    scaled_features: np.ndarray,
    residual_target: np.ndarray,
    timestamps: pd.Series,
    close_values: np.ndarray,
    arima_baseline: np.ndarray,
    split_idx: int,
    window_size: int,
) -> SequenceDatasetBundle:
    train_X: list[np.ndarray] = []
    train_y: list[float] = []
    val_X: list[np.ndarray] = []
    val_y: list[float] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    train_timestamps: list[str] = []
    val_timestamps: list[str] = []

    for target_idx in range(window_size, len(scaled_features)):
        window = scaled_features[target_idx - window_size : target_idx]
        if not np.isfinite(window).all():
            continue
        if not np.isfinite(residual_target[target_idx]):
            continue
        if not np.isfinite(arima_baseline[target_idx]):
            continue
        if not np.isfinite(close_values[target_idx]) or not np.isfinite(close_values[target_idx - 1]):
            continue

        timestamp = timestamps.iloc[target_idx].isoformat()
        if target_idx < split_idx:
            train_X.append(window)
            train_y.append(float(residual_target[target_idx]))
            train_indices.append(target_idx)
            train_timestamps.append(timestamp)
        else:
            val_X.append(window)
            val_y.append(float(residual_target[target_idx]))
            val_indices.append(target_idx)
            val_timestamps.append(timestamp)

    if not train_X or not val_X:
        raise ValueError("Unable to build non-empty train/val regression sequence sets.")

    return SequenceDatasetBundle(
        X_train=np.asarray(train_X, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.float32),
        X_val=np.asarray(val_X, dtype=np.float32),
        y_val=np.asarray(val_y, dtype=np.float32),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
    )


def build_cnn_sequences(
    feature_values: np.ndarray,
    close_values: np.ndarray,
    timestamps: pd.Series,
    split_idx: int,
    window_size: int,
    neutral_threshold: float,
) -> SequenceDatasetBundle:
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    val_X: list[np.ndarray] = []
    val_y: list[int] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    train_timestamps: list[str] = []
    val_timestamps: list[str] = []

    for target_idx in range(window_size, len(feature_values)):
        window = feature_values[target_idx - window_size : target_idx].copy()
        if not np.isfinite(window).all():
            continue
        reference_close = float(close_values[target_idx - 1])
        target_close = float(close_values[target_idx])
        if not np.isfinite(reference_close) or not np.isfinite(target_close) or reference_close == 0.0:
            continue

        window_mean = window.mean(axis=0, keepdims=True)
        window_std = window.std(axis=0, keepdims=True)
        window_std[window_std < 1e-6] = 1.0
        window = (window - window_mean) / window_std

        next_return = (target_close / reference_close) - 1.0
        if next_return > neutral_threshold:
            label = 0
        elif next_return < -neutral_threshold:
            label = 2
        else:
            label = 1

        timestamp = timestamps.iloc[target_idx].isoformat()
        window = np.expand_dims(window.astype(np.float32), axis=0)
        if target_idx < split_idx:
            train_X.append(window)
            train_y.append(label)
            train_indices.append(target_idx)
            train_timestamps.append(timestamp)
        else:
            val_X.append(window)
            val_y.append(label)
            val_indices.append(target_idx)
            val_timestamps.append(timestamp)

    if not train_X or not val_X:
        raise ValueError("Unable to build non-empty train/val CNN sequence sets.")

    return SequenceDatasetBundle(
        X_train=np.asarray(train_X, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.int64),
        X_val=np.asarray(val_X, dtype=np.float32),
        y_val=np.asarray(val_y, dtype=np.int64),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> TorchDataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32 if y.dtype.kind == "f" else torch.long),
    )
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)


def tensor_predict(model: nn.Module, loader: TorchDataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            if output.ndim == 2 and output.shape[1] == 3:
                prob = torch.softmax(output, dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                preds.append(pred)
                probs.append(prob)
            else:
                pred = output.squeeze(-1).cpu().numpy()
                preds.append(pred)
    if probs:
        return np.concatenate(preds), np.concatenate(probs)
    return np.concatenate(preds), np.empty((0, 0), dtype=np.float32)


def train_regression_model(
    bundle: SequenceDatasetBundle,
    baseline_predictions: np.ndarray,
    close_values: np.ndarray,
    log_close_values: np.ndarray,
    feature_columns: list[str],
    timestamps: pd.Series,
    split_idx: int,
    args: argparse.Namespace,
    model_dir: Path,
    imputation_medians: dict[str, float],
    feature_scaler: StandardScaler,
    arima_result: Any,
) -> RegressionArtifacts:
    device = torch.device(args.device)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(bundle.y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_val_scaled = target_scaler.transform(bundle.y_val.reshape(-1, 1)).astype(np.float32).ravel()

    train_loader = make_loader(bundle.X_train, y_train_scaled, args.batch_size)
    val_loader = make_loader(bundle.X_val, y_val_scaled, args.batch_size)

    model = ResidualLSTMRegressor(
        input_size=bundle.X_train.shape[-1],
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = model_dir / "best_lstm_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                batch_size = X_batch.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        LOGGER.info(
            "[ARIMA-LSTM] epoch=%s train_loss=%.6f val_loss=%.6f lr=%.6g",
            epoch,
            train_loss,
            val_loss,
            current_lr,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_columns,
                    "window_size": args.reg_window,
                    "hidden_size": args.lstm_hidden_size,
                    "num_layers": args.lstm_layers,
                    "dropout": args.dropout,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("[ARIMA-LSTM] early stopping at epoch=%s", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pred_scaled, _ = tensor_predict(model, val_loader, device)
    val_pred_residual = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    actual_residual = bundle.y_val

    val_baseline = baseline_predictions[bundle.val_indices]
    predicted_log_close = val_baseline + val_pred_residual
    actual_log_close = log_close_values[bundle.val_indices]
    predicted_close = np.exp(predicted_log_close)
    actual_close = close_values[bundle.val_indices]
    reference_close = close_values[bundle.val_indices - 1]

    rmse = float(np.sqrt(np.mean((predicted_close - actual_close) ** 2)))
    mae = float(np.mean(np.abs(predicted_close - actual_close)))
    mape = float(np.mean(np.abs((predicted_close - actual_close) / np.clip(actual_close, 1e-8, None))))
    predicted_direction = np.sign((predicted_close / np.clip(reference_close, 1e-8, None)) - 1.0)
    actual_direction = np.sign((actual_close / np.clip(reference_close, 1e-8, None)) - 1.0)
    directional_accuracy = float(np.mean(predicted_direction == actual_direction))

    prediction_frame = pd.DataFrame(
        {
            "timestamp": bundle.val_timestamps,
            "actual_close_inr": actual_close,
            "predicted_close_inr": predicted_close,
            "arima_only_close_inr": np.exp(val_baseline),
            "actual_log_close": actual_log_close,
            "predicted_log_close": predicted_log_close,
            "actual_residual": actual_residual,
            "predicted_residual": val_pred_residual,
            "reference_close_inr": reference_close,
            "actual_return": (actual_close / np.clip(reference_close, 1e-8, None)) - 1.0,
            "predicted_return": (predicted_close / np.clip(reference_close, 1e-8, None)) - 1.0,
            "absolute_error_inr": np.abs(predicted_close - actual_close),
        }
    )

    save_pickle(model_dir / "feature_scaler.pkl", feature_scaler)
    save_pickle(model_dir / "target_scaler.pkl", target_scaler)
    save_pickle(model_dir / "feature_imputation_medians.pkl", imputation_medians)
    save_pickle(model_dir / "arima_model.pkl", arima_result)

    prediction_frame.to_csv(model_dir / "val_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    save_json(model_dir / "training_history.json", history)
    plot_history(history, ["train_loss", "val_loss"], model_dir / "loss_curves.png", "ARIMA-LSTM Loss Curves")

    metrics = {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_val_loss_scaled": best_val_loss,
        "val_rmse_inr": rmse,
        "val_mae_inr": mae,
        "val_mape": mape,
        "val_directional_accuracy": directional_accuracy,
        "train_sequences": int(len(bundle.X_train)),
        "val_sequences": int(len(bundle.X_val)),
        "train_split_index": int(split_idx),
    }
    LOGGER.info("[ARIMA-LSTM] best_epoch=%s metrics=%s", best_epoch, metrics)
    save_json(model_dir / "metrics.json", metrics)
    save_json(
        model_dir / "config.json",
        {
            "feature_columns": feature_columns,
            "window_size": args.reg_window,
            "arima_order": list(map(int, args.arima_order.split(","))),
            "hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        },
    )
    return RegressionArtifacts(metrics=metrics, history=history, prediction_frame=prediction_frame)


def build_class_weight_vector(y_train: np.ndarray) -> np.ndarray:
    present_classes = np.unique(y_train)
    present_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    full_weights = np.ones(3, dtype=np.float32)
    for klass, weight in zip(present_classes, present_weights):
        full_weights[int(klass)] = float(weight)
    return full_weights


def class_distribution(y: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[idx]: int((y == idx).sum()) for idx in sorted(CLASS_NAMES)}


def per_class_accuracy(confusion: np.ndarray) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for idx, class_name in CLASS_NAMES.items():
        total = int(confusion[idx].sum())
        correct = int(confusion[idx, idx])
        accuracy = float(correct / total) if total else 0.0
        output[class_name] = {"correct": correct, "total": total, "accuracy": accuracy}
    return output


def train_cnn_model(
    bundle: SequenceDatasetBundle,
    feature_columns: list[str],
    args: argparse.Namespace,
    model_dir: Path,
) -> ClassificationArtifacts:
    device = torch.device(args.device)
    class_weights = build_class_weight_vector(bundle.y_train)
    LOGGER.info("[CNN] class distribution train=%s", class_distribution(bundle.y_train))
    LOGGER.info("[CNN] class distribution val=%s", class_distribution(bundle.y_val))
    LOGGER.info("[CNN] balanced class weights=%s", class_weights.tolist())

    train_loader = make_loader(bundle.X_train, bundle.y_train, args.batch_size)
    val_loader = make_loader(bundle.X_val, bundle.y_val, args.batch_size)

    model = DirectionCNNClassifier(
        time_steps=bundle.X_train.shape[-2],
        num_features=bundle.X_train.shape[-1],
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = model_dir / "best_cnn_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size
            train_true.append(y_batch.cpu().numpy())
            train_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

        train_loss = train_loss_sum / max(train_count, 1)
        train_true_np = np.concatenate(train_true)
        train_pred_np = np.concatenate(train_pred)
        train_acc = float(accuracy_score(train_true_np, train_pred_np))

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                batch_size = X_batch.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size
                val_true.append(y_batch.cpu().numpy())
                val_pred.append(logits.argmax(dim=1).cpu().numpy())

        val_loss = val_loss_sum / max(val_count, 1)
        val_true_np = np.concatenate(val_true)
        val_pred_np = np.concatenate(val_pred)
        val_acc = float(accuracy_score(val_true_np, val_pred_np))
        val_bal_acc = float(balanced_accuracy_score(val_true_np, val_pred_np))
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal_acc,
                "lr": current_lr,
            }
        )

        LOGGER.info(
            "[CNN] epoch=%s train_loss=%.6f val_loss=%.6f train_acc=%.4f val_acc=%.4f val_bal_acc=%.4f lr=%.6g",
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            val_bal_acc,
            current_lr,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_columns,
                    "window_size": args.cnn_window,
                    "dropout": args.dropout,
                    "neutral_threshold": args.neutral_threshold,
                    "class_weights": class_weights.tolist(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("[CNN] early stopping at epoch=%s", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_pred, val_prob = tensor_predict(model, val_loader, device)
    confusion = confusion_matrix(bundle.y_val, val_pred, labels=[0, 1, 2])
    class_acc = per_class_accuracy(confusion)

    prediction_frame = pd.DataFrame(
        {
            "timestamp": bundle.val_timestamps,
            "true_label": [CLASS_NAMES[int(value)] for value in bundle.y_val],
            "predicted_label": [CLASS_NAMES[int(value)] for value in val_pred],
            "up_prob": val_prob[:, 0],
            "neutral_prob": val_prob[:, 1],
            "down_prob": val_prob[:, 2],
        }
    )

    prediction_frame.to_csv(model_dir / "val_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    save_json(model_dir / "training_history.json", history)
    pd.DataFrame(confusion, index=[CLASS_NAMES[i] for i in range(3)], columns=[CLASS_NAMES[i] for i in range(3)]).to_csv(
        model_dir / "confusion_matrix.csv"
    )
    plot_history(
        history,
        ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_balanced_accuracy"],
        model_dir / "training_curves.png",
        "CNN Training Curves",
    )
    plot_confusion(confusion, model_dir / "confusion_matrix.png")

    metrics = {
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_val_loss": best_val_loss,
        "val_accuracy": float(accuracy_score(bundle.y_val, val_pred)),
        "val_balanced_accuracy": float(balanced_accuracy_score(bundle.y_val, val_pred)),
        "train_windows": int(len(bundle.X_train)),
        "val_windows": int(len(bundle.X_val)),
        "neutral_threshold": args.neutral_threshold,
        "class_distribution_train": class_distribution(bundle.y_train),
        "class_distribution_val": class_distribution(bundle.y_val),
        "class_weight_balanced": {CLASS_NAMES[idx]: float(class_weights[idx]) for idx in range(3)},
        "per_class_accuracy": class_acc,
    }
    LOGGER.info("[CNN] best_epoch=%s metrics=%s", best_epoch, metrics)
    save_json(model_dir / "metrics.json", metrics)
    save_json(
        model_dir / "config.json",
        {
            "feature_columns": feature_columns,
            "window_size": args.cnn_window,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "neutral_threshold": args.neutral_threshold,
        },
    )
    save_json(model_dir / "per_class_accuracy.json", class_acc)
    return ClassificationArtifacts(metrics=metrics, history=history, prediction_frame=prediction_frame, confusion=confusion)


def run_for_symbol(symbol: str, args: argparse.Namespace, root_run_dir: Path) -> dict[str, Any]:
    symbol_dir = root_run_dir / sanitize_symbol(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    frame, source_name = load_symbol_frame(symbol, args)
    LOGGER.info("Loaded %s rows for %s from %s", len(frame), symbol, source_name)

    split_idx = int(len(frame) * args.train_split)
    if split_idx <= max(args.reg_window, args.cnn_window):
        raise ValueError("Training split leaves too little history for the requested windows.")
    if len(frame) - split_idx <= 20:
        raise ValueError("Validation split is too small after chronological split.")

    feature_table = build_feature_table(frame)
    regression_columns = select_regression_columns(feature_table)
    cnn_columns = select_cnn_columns(feature_table)
    if len(regression_columns) < 5:
        raise ValueError("Regression feature set is too small after filtering.")
    if len(cnn_columns) < 5:
        raise ValueError("CNN feature set is too small after filtering.")

    reg_feature_frame = feature_table[regression_columns].copy()
    cnn_feature_frame = feature_table[cnn_columns].copy()

    reg_scaled, reg_scaler, reg_medians = impute_and_scale_features(reg_feature_frame, split_idx)
    cnn_scaled, _, _ = impute_and_scale_features(cnn_feature_frame, split_idx)

    arima_order = tuple(int(part.strip()) for part in args.arima_order.split(","))
    arima_result, arima_baseline = compute_arima_predictions(frame["log_close"], split_idx, arima_order)
    residual_target = frame["log_close"].values - arima_baseline

    reg_bundle = build_regression_sequences(
        scaled_features=reg_scaled,
        residual_target=residual_target,
        timestamps=frame["timestamp"],
        close_values=frame["close"].values.astype(float),
        arima_baseline=arima_baseline,
        split_idx=split_idx,
        window_size=args.reg_window,
    )
    cnn_bundle = build_cnn_sequences(
        feature_values=cnn_scaled,
        close_values=frame["close"].values.astype(float),
        timestamps=frame["timestamp"],
        split_idx=split_idx,
        window_size=args.cnn_window,
        neutral_threshold=args.neutral_threshold,
    )

    LOGGER.info("Regression features=%s", regression_columns)
    LOGGER.info("CNN features=%s", cnn_columns)
    LOGGER.info(
        "Prepared sequences for %s: reg_train=%s reg_val=%s cnn_train=%s cnn_val=%s",
        symbol,
        len(reg_bundle.X_train),
        len(reg_bundle.X_val),
        len(cnn_bundle.X_train),
        len(cnn_bundle.X_val),
    )
    LOGGER.info("CNN class distribution overall=%s", class_distribution(np.concatenate([cnn_bundle.y_train, cnn_bundle.y_val])))

    regression_dir = symbol_dir / "arima_lstm"
    classification_dir = symbol_dir / "cnn_pattern"
    regression_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    regression_artifacts = train_regression_model(
        bundle=reg_bundle,
        baseline_predictions=arima_baseline,
        close_values=frame["close"].values.astype(float),
        log_close_values=frame["log_close"].values.astype(float),
        feature_columns=regression_columns,
        timestamps=frame["timestamp"],
        split_idx=split_idx,
        args=args,
        model_dir=regression_dir,
        imputation_medians=reg_medians,
        feature_scaler=reg_scaler,
        arima_result=arima_result,
    )
    classification_artifacts = train_cnn_model(
        bundle=cnn_bundle,
        feature_columns=cnn_columns,
        args=args,
        model_dir=classification_dir,
    )

    summary = {
        "symbol": symbol,
        "source": source_name,
        "rows": int(len(frame)),
        "train_rows": int(split_idx),
        "val_rows": int(len(frame) - split_idx),
        "regression_metrics": regression_artifacts.metrics,
        "classification_metrics": classification_artifacts.metrics,
    }
    save_json(symbol_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root_run_dir = Path(args.output_root) / f"regime_aware_{args.interval}_{timestamp}"
    configure_logging(root_run_dir)

    LOGGER.info("Starting regime-aware training run at %s", root_run_dir)
    LOGGER.info("Args: %s", vars(args))

    all_summaries: list[dict[str, Any]] = []
    failed_symbols: list[dict[str, str]] = []
    skipped_symbols: list[dict[str, str]] = []

    from src.db.queries import get_market_data_quality

    for symbol in args.symbols:
        LOGGER.info("=== Training symbol=%s ===", symbol)
        quality = get_market_data_quality(symbol, args.interval, dataset_type="historical")
        if quality is not None and not quality.get("train_ready"):
            LOGGER.warning(
                "Skipping %s due to historical quality gate: status=%s details=%s",
                symbol,
                quality.get("status"),
                quality.get("details_json"),
            )
            skipped_symbols.append(
                {
                    "symbol": symbol,
                    "reason": "historical_quality_gate",
                    "details": str(quality.get("details_json")),
                }
            )
            continue
        try:
            summary = run_for_symbol(symbol, args, root_run_dir)
            all_summaries.append(summary)
            LOGGER.info("Completed symbol=%s summary=%s", symbol, summary)
        except Exception as exc:
            LOGGER.exception("Training failed for %s: %s", symbol, exc)
            failed_symbols.append({"symbol": symbol, "error": str(exc)})

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root_run_dir),
        "symbols_requested": args.symbols,
        "symbols_succeeded": [summary["symbol"] for summary in all_summaries],
        "symbols_skipped": skipped_symbols,
        "symbols_failed": failed_symbols,
        "config": vars(args),
        "summaries": all_summaries,
    }
    save_json(root_run_dir / "run_manifest.json", manifest)

    if failed_symbols:
        raise SystemExit(1)

    LOGGER.info("All training jobs completed successfully.")


if __name__ == "__main__":
    main()
