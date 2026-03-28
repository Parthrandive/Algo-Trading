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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

try:
    from pmdarima import auto_arima as pmd_auto_arima
except Exception:  # pragma: no cover - optional runtime dependency
    pmd_auto_arima = None

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
from src.agents.technical.label_utils import (
    choose_neutral_threshold as choose_neutral_threshold_unified,
    build_labels as build_labels_unified,
    atr_effective_threshold,
    recall_balance as compute_recall_balance,
    directional_coverage as compute_directional_coverage,
    class_balance_report,
)
from src.agents.technical.validation_metrics import (
    post_cost_sharpe,
    expected_calibration_error,
)
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


REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}
LOGGER = logging.getLogger("train_regime_aware")
CLASS_NAMES = {0: "up", 1: "neutral", 2: "down"}
FIXED_FOCAL_GAMMA = 3.0
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
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_timestamps: list[str]
    val_timestamps: list[str]
    test_timestamps: list[str]


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
        default=None,
        help="One or more equity symbols to train, e.g. INFY.NS RELIANCE.NS TCS.NS",
    )
    parser.add_argument("--data-path", default=None, help="Optional CSV/Parquet file containing the training dataset.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="Optional override for DATABASE_URL.")
    parser.add_argument("--gold-dir", default="data/gold", help="Gold parquet fallback directory.")
    parser.add_argument("--interval", default="1d", help="Candle interval to use when falling back to OHLCV DB loader.")
    parser.add_argument("--limit", type=int, default=4000, help="Maximum rows to load per symbol.")
    parser.add_argument("--output-root", default="/tmp/training_runs", help="Directory for run artifacts (defaults to /tmp to avoid disk bloat).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu.")

    parser.add_argument("--epochs", type=int, default=150, help="Maximum epochs. Recommended 100-200.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=7, help="ReduceLROnPlateau patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization via AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate between 0.2 and 0.3.")

    parser.add_argument("--reg-window", type=int, default=20, help="Sequence length for ARIMA-LSTM residual model.")
    parser.add_argument("--cnn-window", type=int, default=30, help="Sequence length for CNN classifier.")
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=None,
        help="Optional fixed neutral threshold override. Default derives threshold from each symbol's training returns.",
    )
    parser.add_argument("--threshold-target-min", type=float, default=0.20, help="Target minimum neutral class ratio.")
    parser.add_argument("--threshold-target-max", type=float, default=0.25, help="Target maximum neutral class ratio.")
    parser.add_argument("--threshold-target-neutral", type=float, default=0.22, help="Fallback neutral ratio target.")
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Optional fixed focal gamma override. Default derives gamma from each symbol's class imbalance.",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha coefficient for focal loss.")
    parser.add_argument("--cv-max-splits", type=int, default=5, help="Maximum walk-forward folds for per-symbol CV.")
    parser.add_argument("--cv-min-train", type=int, default=200, help="Minimum train samples per CV split.")
    parser.add_argument("--cv-min-val", type=int, default=50, help="Minimum validation samples per CV split.")
    parser.add_argument(
        "--disable-cnn-cv",
        action="store_true",
        help="Disable CNN walk-forward CV (enabled by default).",
    )
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order as p,d,q.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--min-rows", type=int, default=300, help="Minimum usable rows after cleaning.")
    parser.add_argument(
        "--fx-context-symbol",
        default=FOREX_SYMBOLS[0],
        help="Forex context symbol merged as external feature for equity targets.",
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
    return parser.parse_args()


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("=", "_").replace(".", "_")


def quality_gate_split_counts(frame: pd.DataFrame) -> SplitCounts:
    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return SplitCounts(
        train_rows=train_end,
        val_rows=val_end - train_end,
        test_rows=n_rows - val_end,
    )


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


def check_nulls(df: pd.DataFrame, symbol: str, stage: str) -> None:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    LOGGER.info("[%s] %s — %s rows, nulls: %s", symbol, stage, len(df), nulls.to_dict())


def _load_raw_bars(symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    raw_loader = DataLoader(args.database_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db"))
    return raw_loader.load_historical_bars(
        symbol=symbol,
        limit=args.limit,
        use_nse_fallback=False,
        interval=args.interval,
        include_macro=True,
    )


def _load_fx_context_frame(args: argparse.Namespace) -> pd.DataFrame:
    fx_symbol = str(args.fx_context_symbol).strip()
    if not fx_symbol:
        return pd.DataFrame(columns=["timestamp", "usdinr_close"])

    frame = pd.DataFrame()
    try:
        gold_loader = RegimeDataLoader(database_url=args.database_url, gold_dir=args.gold_dir)
        frame = gold_loader.load_features(symbol=fx_symbol, limit=args.limit)
    except Exception as exc:
        LOGGER.warning("Failed loading FX context from gold for %s: %s", fx_symbol, exc)

    if frame.empty or "timestamp" not in frame.columns or "close" not in frame.columns:
        try:
            frame = _load_raw_bars(fx_symbol, args)
        except Exception as exc:
            LOGGER.warning("Failed loading FX context from OHLCV for %s: %s", fx_symbol, exc)
            return pd.DataFrame(columns=["timestamp", "usdinr_close"])

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "usdinr_close"])

    return frame[["timestamp", "close"]].rename(columns={"close": "usdinr_close"}).reset_index(drop=True)


def load_symbol_frame(
    symbol: str,
    args: argparse.Namespace,
    fx_context_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
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
        gold_frame = gold_loader.load_features(symbol=symbol, limit=args.limit)
        raw_frame = pd.DataFrame()
        try:
            raw_frame = _load_raw_bars(symbol, args)
        except Exception as exc:
            LOGGER.warning("Failed loading raw OHLCV bars for %s: %s", symbol, exc)

        frame = gold_frame
        source_name = "gold_features"
        if frame.empty or not REQUIRED_OHLCV.issubset(frame.columns):
            frame = raw_frame
            source_name = "ohlcv_bars"
        elif symbol.endswith(".NS") and not raw_frame.empty:
            if len(raw_frame) >= max(400, int(len(frame) * 1.10)):
                frame = raw_frame
                source_name = "ohlcv_bars"

    if frame.empty:
        raise ValueError(f"No data found for {symbol}.")
    raw_rows_loaded = int(len(frame))

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
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    check_nulls(frame, symbol, "after load")
    all_null_columns = [column for column in frame.columns if frame[column].isna().all()]
    if all_null_columns:
        frame = frame.drop(columns=all_null_columns)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame.ffill()
    essential_columns = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    if essential_columns:
        frame = frame.dropna(subset=essential_columns)
    frame = frame.reset_index()
    check_nulls(frame, symbol, "after ffill_keep_sparse")
    rows_after_ffill = int(len(frame))

    rolling_window = max(3, min(20, max(len(frame) // 5, 3)))
    frame["rolling_mean"] = frame["close"].rolling(rolling_window, min_periods=1).mean()

    if symbol.endswith(".NS"):
        fx_frame = fx_context_frame if fx_context_frame is not None else _load_fx_context_frame(args)
        if not fx_frame.empty:
            frame = pd.merge_asof(
                frame.sort_values("timestamp"),
                fx_frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            frame["usdinr_close"] = pd.to_numeric(frame["usdinr_close"], errors="coerce").ffill()

    if len(frame) < args.min_rows:
        raise ValueError(f"Only {len(frame)} rows available for {symbol}; need at least {args.min_rows}.")

    is_forex_target = is_forex(symbol)
    frame = engineer_features(frame, is_forex=is_forex_target)
    frame["symbol"] = symbol
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["daily_return"] = frame["close"].pct_change()
    frame["log_close"] = np.log(frame["close"].clip(lower=1e-8))
    check_nulls(frame, symbol, "after features")

    post_feature_all_null = [column for column in frame.columns if frame[column].isna().all()]
    if post_feature_all_null:
        frame = frame.drop(columns=post_feature_all_null)
    frame = frame.sort_values("timestamp").ffill()
    post_feature_required = [column for column in ["open", "high", "low", "close", "volume", "log_return"] if column in frame.columns]
    if post_feature_required:
        frame = frame.dropna(subset=post_feature_required)
    frame = frame.reset_index(drop=True)
    check_nulls(frame, symbol, "after feature ffill_keep_sparse")
    diagnostics = {
        "raw_rows_loaded": raw_rows_loaded,
        "rows_after_ffill_dropna": rows_after_ffill,
        "rows_after_feature_engineering": int(len(frame)),
    }
    return frame, source_name, diagnostics


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


def impute_and_scale_features(feature_frame: pd.DataFrame, train_end: int) -> tuple[np.ndarray, StandardScaler, dict[str, float]]:
    train_features = feature_frame.iloc[:train_end].copy()
    medians = train_features.median(numeric_only=True)
    medians = medians.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    filled = feature_frame.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(filled.iloc[:train_end].values)
    scaled = scaler.transform(filled.values).astype(np.float32)
    return scaled, scaler, {column: float(value) for column, value in medians.items()}


def compute_arima_predictions(
    symbol: str,
    log_return: pd.Series,
    train_end: int,
    arima_order: tuple[int, int, int],
) -> tuple[Any, np.ndarray, tuple[int, int, int]]:
    train_series = log_return.iloc[:train_end].astype(float).values
    if len(train_series) < 30:
        raise ValueError("Need at least 30 training points to fit ARIMA.")

    baseline = np.full(len(log_return), np.nan, dtype=np.float64)

    if pmd_auto_arima is not None:
        try:
            auto_model = pmd_auto_arima(
                train_series,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
            )
            selected_order = tuple(int(part) for part in auto_model.order)
            LOGGER.info("[%s] Best ARIMA order: %s", symbol, selected_order)
            train_pred = np.asarray(auto_model.predict_in_sample(), dtype=np.float64).reshape(-1)
            baseline[: len(train_pred)] = train_pred

            rolling_model = auto_model
            for idx in range(train_end, len(log_return)):
                forecast = np.asarray(rolling_model.predict(n_periods=1), dtype=np.float64).reshape(-1)
                baseline[idx] = float(forecast[0])
                rolling_model.update(float(log_return.iloc[idx]))
            return auto_model, baseline, selected_order
        except Exception as exc:
            LOGGER.warning("[%s] auto_arima failed; falling back to statsmodels ARIMA(%s): %s", symbol, arima_order, exc)
    else:
        LOGGER.info("[%s] pmdarima not available; using statsmodels ARIMA(%s).", symbol, arima_order)

    model = sm.tsa.ARIMA(
        train_series,
        order=arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit()
    LOGGER.info("[%s] Using fallback statsmodels order=%s", symbol, arima_order)

    train_pred = np.asarray(result.predict(start=0, end=train_end - 1), dtype=np.float64).reshape(-1)
    baseline[:train_end] = train_pred

    rolling_result = result
    for idx in range(train_end, len(log_return)):
        forecast = np.asarray(rolling_result.forecast(steps=1), dtype=np.float64).reshape(-1)
        baseline[idx] = float(forecast[0])
        next_value = float(log_return.iloc[idx])
        try:
            rolling_result = rolling_result.append([next_value], refit=False)
        except Exception:
            history = log_return.iloc[: idx + 1].astype(float).values
            rolling_result = sm.tsa.ARIMA(
                history,
                order=arima_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()

    return result, baseline, arima_order


def build_regression_sequences(
    scaled_features: np.ndarray,
    residual_target: np.ndarray,
    timestamps: pd.Series,
    close_values: np.ndarray,
    arima_baseline: np.ndarray,
    train_end: int,
    val_end: int,
    window_size: int,
) -> SequenceDatasetBundle:
    train_X: list[np.ndarray] = []
    train_y: list[float] = []
    val_X: list[np.ndarray] = []
    val_y: list[float] = []
    test_X: list[np.ndarray] = []
    test_y: list[float] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    train_timestamps: list[str] = []
    val_timestamps: list[str] = []
    test_timestamps: list[str] = []

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
        if target_idx < train_end:
            train_X.append(window)
            train_y.append(float(residual_target[target_idx]))
            train_indices.append(target_idx)
            train_timestamps.append(timestamp)
        elif target_idx < val_end:
            val_X.append(window)
            val_y.append(float(residual_target[target_idx]))
            val_indices.append(target_idx)
            val_timestamps.append(timestamp)
        else:
            test_X.append(window)
            test_y.append(float(residual_target[target_idx]))
            test_indices.append(target_idx)
            test_timestamps.append(timestamp)

    if not train_X or not val_X or not test_X:
        raise ValueError("Unable to build non-empty train/val/test regression sequence sets.")

    return SequenceDatasetBundle(
        X_train=np.asarray(train_X, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.float32),
        X_val=np.asarray(val_X, dtype=np.float32),
        y_val=np.asarray(val_y, dtype=np.float32),
        X_test=np.asarray(test_X, dtype=np.float32),
        y_test=np.asarray(test_y, dtype=np.float32),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        test_timestamps=test_timestamps,
    )


def build_cnn_sequences(
    feature_values: np.ndarray,
    close_values: np.ndarray,
    timestamps: pd.Series,
    train_end: int,
    val_end: int,
    window_size: int,
    args: argparse.Namespace,
    neutral_threshold: float | None = None,
    next_log_returns: np.ndarray | None = None,
    label_threshold: float | None = None,
) -> SequenceDatasetBundle:
    train_X: list[np.ndarray] = []
    train_y: list[int] = []
    val_X: list[np.ndarray] = []
    val_y: list[int] = []
    test_X: list[np.ndarray] = []
    test_y: list[int] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    train_timestamps: list[str] = []
    val_timestamps: list[str] = []
    test_timestamps: list[str] = []

    for target_idx in range(window_size, len(feature_values)):
        window = feature_values[target_idx - window_size : target_idx].copy()
        if not np.isfinite(window).all():
            continue
        window_mean = window.mean(axis=0, keepdims=True)
        window_std = window.std(axis=0, keepdims=True)
        window_std[window_std < 1e-6] = 1.0
        window = (window - window_mean) / window_std

        if next_log_returns is not None:
            if target_idx >= len(next_log_returns):
                continue
            next_return = float(next_log_returns[target_idx])
            if not np.isfinite(next_return):
                continue
            threshold = float(label_threshold if label_threshold is not None else (neutral_threshold or 0.0))
            label = int(apply_labels(np.asarray([next_return], dtype=np.float64), threshold, args)[0])
        else:
            reference_close = float(close_values[target_idx - 1])
            target_close = float(close_values[target_idx])
            if not np.isfinite(reference_close) or not np.isfinite(target_close) or reference_close == 0.0:
                continue
            local_threshold = float(neutral_threshold if neutral_threshold is not None else 0.0)
            next_return = (target_close / reference_close) - 1.0
            if next_return > local_threshold:
                label = 0
            elif next_return < -local_threshold:
                label = 2
            else:
                label = 1

        timestamp = timestamps.iloc[target_idx].isoformat()
        window = np.expand_dims(window.astype(np.float32), axis=0)
        if target_idx < train_end:
            train_X.append(window)
            train_y.append(label)
            train_indices.append(target_idx)
            train_timestamps.append(timestamp)
        elif target_idx < val_end:
            val_X.append(window)
            val_y.append(label)
            val_indices.append(target_idx)
            val_timestamps.append(timestamp)
        else:
            test_X.append(window)
            test_y.append(label)
            test_indices.append(target_idx)
            test_timestamps.append(timestamp)

    if not train_X or not val_X or not test_X:
        raise ValueError("Unable to build non-empty train/val/test CNN sequence sets.")

    return SequenceDatasetBundle(
        X_train=np.asarray(train_X, dtype=np.float32),
        y_train=np.asarray(train_y, dtype=np.int64),
        X_val=np.asarray(val_X, dtype=np.float32),
        y_val=np.asarray(val_y, dtype=np.int64),
        X_test=np.asarray(test_X, dtype=np.float32),
        y_test=np.asarray(test_y, dtype=np.int64),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        test_timestamps=test_timestamps,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> TorchDataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32 if y.dtype.kind == "f" else torch.long),
    )
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def compute_trading_metrics(predicted_return: np.ndarray, actual_return: np.ndarray) -> dict[str, float]:
    signal = np.sign(predicted_return)
    strategy_return = signal * actual_return
    trades_mask = signal != 0
    trades = strategy_return[trades_mask]

    total_trades = int(trades.size)
    win_rate = float(np.mean(trades > 0)) if total_trades else 0.0
    positive_sum = float(trades[trades > 0].sum()) if total_trades else 0.0
    negative_sum = float(np.abs(trades[trades < 0].sum())) if total_trades else 0.0
    if negative_sum == 0.0:
        profit_factor = float("inf") if positive_sum > 0 else 0.0
    else:
        profit_factor = positive_sum / negative_sum

    avg_trade_return = float(trades.mean()) if total_trades else 0.0
    std = float(strategy_return.std(ddof=0))
    annualization = math.sqrt(252.0 * 6.5)
    sharpe = float(strategy_return.mean() / std * annualization) if std > 1e-12 else 0.0

    equity_curve = np.cumprod(1.0 + strategy_return)
    if equity_curve.size:
        running_peak = np.maximum.accumulate(equity_curve)
        drawdown = 1.0 - (equity_curve / np.clip(running_peak, 1e-12, None))
        max_drawdown = float(np.max(drawdown))
    else:
        max_drawdown = 0.0

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "average_trade_return": avg_trade_return,
        "total_trades": total_trades,
    }


def class_distribution_percentages(y: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {CLASS_NAMES[idx]: 0.0 for idx in sorted(CLASS_NAMES)}
    total = float(len(y))
    return {CLASS_NAMES[idx]: float((y == idx).sum() / total) for idx in sorted(CLASS_NAMES)}


def apply_labels(log_returns: np.ndarray, threshold: float, args: argparse.Namespace) -> np.ndarray:
    labels, _ = build_labels_unified(
        log_returns, 
        threshold=threshold, 
        use_binary=False, 
        mode="fixed"
    )
    return labels


def find_threshold(frame: pd.DataFrame, symbol: str, args: argparse.Namespace) -> tuple[float, float]:
    train_log_returns = frame["log_return"].values
    values = np.asarray(train_log_returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"[{symbol}] no finite training log returns available for threshold tuning.")

    if getattr(args, "label_mode", "fixed") == "atr":
        effective_threshold = atr_effective_threshold(
            high=frame["high"].values,
            low=frame["low"].values,
            close=frame["close"].values,
            k=getattr(args, "atr_k", 0.5),
            atr_period=int(getattr(args, "atr_period", 14)),
        )
        # Compute neutral ratio for logging
        y_check, _ = build_labels_unified(values, threshold=effective_threshold, mode="fixed")
        neutral_pct = float((y_check == 1).mean())
        LOGGER.info("[%s] ATR-scaled threshold selected: %.6f (neutral=%.1f%%)", symbol, effective_threshold, neutral_pct * 100.0)
        return effective_threshold, neutral_pct

    if args.neutral_threshold is not None:
        override = float(args.neutral_threshold)
        labels_fixed, _ = build_labels_unified(values, override, mode="fixed")
        neutral_pct = float((labels_fixed == 1).mean())
        LOGGER.info("[%s] Using fixed threshold override %.6f -> neutral=%.1f%%", symbol, override, neutral_pct * 100.0)
        return override, neutral_pct

    # Use unified threshold selector for distribution-based fixed thresholds
    return choose_neutral_threshold_unified(
        train_forward_returns=values,
        requested_threshold=0.005,
        min_neutral_ratio=float(args.threshold_target_min),
        max_neutral_ratio=float(args.threshold_target_max),
        target_neutral_ratio=float(args.threshold_target_neutral),
    )


def summarize_class_counts(y_train: np.ndarray, symbol: str) -> tuple[np.ndarray, float]:
    counts = np.bincount(y_train.astype(np.int64), minlength=3).astype(np.int64)
    dominant = float(counts.max()) if counts.size else 1.0
    positive_counts = counts[counts > 0]
    minority = float(positive_counts.min()) if positive_counts.size else 1.0
    imbalance = dominant / (minority + 1e-6)

    LOGGER.info(
        "[%s] Class counts=%s | imbalance=%.2f | focal_gamma=%.2f",
        symbol,
        counts.tolist(),
        imbalance,
        FIXED_FOCAL_GAMMA,
    )
    return counts, float(imbalance)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * ce_loss).mean()


def get_n_splits(n_samples: int, symbol: str, args: argparse.Namespace) -> int:
    max_candidate = max(2, int(args.cv_max_splits))
    candidates = list(range(max_candidate, 1, -1))
    for n_splits in candidates:
        fold_size = n_samples // (n_splits + 1)
        if fold_size >= args.cv_min_val and (n_samples - fold_size) >= args.cv_min_train:
            LOGGER.info("[%s] Using %s-fold CV (n_samples=%s)", symbol, n_splits, n_samples)
            return n_splits
    LOGGER.warning("[%s] too few samples for multi-fold CV (%s). Falling back to single split.", symbol, n_samples)
    return 1


def build_cnn_full_sequences(
    feature_values: np.ndarray,
    next_log_returns: np.ndarray,
    window_size: int,
    label_threshold: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    for target_idx in range(window_size, len(feature_values)):
        window = feature_values[target_idx - window_size : target_idx].copy()
        if not np.isfinite(window).all():
            continue
        if target_idx >= len(next_log_returns):
            continue
        next_return = float(next_log_returns[target_idx])
        if not np.isfinite(next_return):
            continue

        window_mean = window.mean(axis=0, keepdims=True)
        window_std = window.std(axis=0, keepdims=True)
        window_std[window_std < 1e-6] = 1.0
        window = (window - window_mean) / window_std

        label = int(apply_labels(np.asarray([next_return], dtype=np.float64), label_threshold, args)[0])
        all_X.append(np.expand_dims(window.astype(np.float32), axis=0))
        all_y.append(label)

    if not all_X:
        raise ValueError("Unable to build non-empty full CNN sequence set.")
    return np.asarray(all_X, dtype=np.float32), np.asarray(all_y, dtype=np.int64)


def run_cnn_walk_forward_cv(
    symbol: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, int]:
    n_splits = get_n_splits(len(X_full), symbol, args)
    device = torch.device(args.device)
    fold_scores: list[float] = []
    fold_total = 0
    if n_splits == 1:
        split_point = max(int(args.cv_min_train), len(X_full) - int(args.cv_min_val))
        if split_point >= len(X_full):
            raise ValueError(f"[{symbol}] unable to build single CV split for n_samples={len(X_full)}.")
        split_iter: Iterable[tuple[np.ndarray, np.ndarray]] = [
            (
                np.arange(0, split_point, dtype=np.int64),
                np.arange(split_point, len(X_full), dtype=np.int64),
            )
        ]
    else:
        split_iter = TimeSeriesSplit(n_splits=n_splits).split(X_full)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        X_tr_raw = X_full[train_idx]
        X_v_raw = X_full[val_idx]
        y_tr = y_full[train_idx]
        y_v = y_full[val_idx]

        if len(y_tr) < args.cv_min_train or len(y_v) < args.cv_min_val:
            continue
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_v)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_shape = X_tr_raw.shape
        X_v_shape = X_v_raw.shape
        X_tr = scaler.fit_transform(X_tr_raw.reshape(len(X_tr_raw), -1)).reshape(X_tr_shape).astype(np.float32)
        X_v = scaler.transform(X_v_raw.reshape(len(X_v_raw), -1)).reshape(X_v_shape).astype(np.float32)

        criterion = FocalLoss(gamma=FIXED_FOCAL_GAMMA, alpha=args.focal_alpha).to(device)

        train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
        val_loader = make_loader(X_v, y_v, args.batch_size)

        model = DirectionCNNClassifier(
            time_steps=X_tr.shape[-2],
            num_features=X_tr.shape[-1],
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
        )

        best_val_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for _epoch in range(1, args.epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    batch_size = X_batch.size(0)
                    val_loss_sum += float(loss.item()) * batch_size
                    val_count += batch_size

            val_loss = val_loss_sum / max(val_count, 1)
            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model.to(device)

        preds, _ = tensor_predict(model, val_loader, device)
        bal_acc = float(balanced_accuracy_score(y_v, preds))
        fold_total += 1
        fold_scores.append(bal_acc)
        LOGGER.info("[%s] Fold %s Balanced Accuracy: %.4f", symbol, fold_idx, bal_acc)

    if not fold_scores:
        raise ValueError(f"[{symbol}] walk-forward CV produced no valid folds.")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    LOGGER.info("[%s] CV Mean: %.4f ± %.4f", symbol, cv_mean, cv_std)
    if cv_std > 0.05:
        LOGGER.warning("[%s] high CV variance (%.4f) — consider stronger regularization.", symbol, cv_std)
    if cv_mean < 0.38:
        LOGGER.warning("[%s] low CV mean (%.4f) — check threshold and focal settings.", symbol, cv_mean)
    return cv_mean, cv_std, fold_total


def train_regression_model(
    bundle: SequenceDatasetBundle,
    baseline_predictions: np.ndarray,
    log_return_values: np.ndarray,
    feature_columns: list[str],
    timestamps: pd.Series,
    train_end: int,
    val_end: int,
    args: argparse.Namespace,
    model_dir: Path,
    imputation_medians: dict[str, float],
    feature_scaler: StandardScaler,
    arima_result: Any,
    selected_arima_order: tuple[int, int, int],
) -> RegressionArtifacts:
    device = torch.device(args.device)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(bundle.y_train.reshape(-1, 1)).astype(np.float32).ravel()
    y_val_scaled = target_scaler.transform(bundle.y_val.reshape(-1, 1)).astype(np.float32).ravel()
    y_test_scaled = target_scaler.transform(bundle.y_test.reshape(-1, 1)).astype(np.float32).ravel()

    train_loader = make_loader(bundle.X_train, y_train_scaled, args.batch_size)
    val_loader = make_loader(bundle.X_val, y_val_scaled, args.batch_size)
    test_loader = make_loader(bundle.X_test, y_test_scaled, args.batch_size)

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
    test_pred_scaled, _ = tensor_predict(model, test_loader, device)
    val_pred_residual = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
    test_pred_residual = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
    actual_residual = np.concatenate([bundle.y_val, bundle.y_test])

    val_baseline = baseline_predictions[bundle.val_indices]
    test_baseline = baseline_predictions[bundle.test_indices]
    predicted_val_return = val_baseline + val_pred_residual
    predicted_test_return = test_baseline + test_pred_residual
    actual_val_return = log_return_values[bundle.val_indices]
    actual_test_return = log_return_values[bundle.test_indices]

    rmse_val = float(np.sqrt(np.mean((predicted_val_return - actual_val_return) ** 2)))
    mae_val = float(np.mean(np.abs(predicted_val_return - actual_val_return)))
    rmse_test = float(np.sqrt(np.mean((predicted_test_return - actual_test_return) ** 2)))
    mae_test = float(np.mean(np.abs(predicted_test_return - actual_test_return)))
    predicted_direction = np.sign(predicted_val_return)
    actual_direction = np.sign(actual_val_return)
    directional_accuracy = float(np.mean(predicted_direction == actual_direction))
    test_trading_metrics = compute_trading_metrics(predicted_test_return, actual_test_return)

    prediction_frame = pd.DataFrame(
        {
            "timestamp": [*bundle.val_timestamps, *bundle.test_timestamps],
            "dataset_split": ["val"] * len(bundle.val_timestamps) + ["test"] * len(bundle.test_timestamps),
            "actual_return": np.concatenate([actual_val_return, actual_test_return]),
            "predicted_return": np.concatenate([predicted_val_return, predicted_test_return]),
            "arima_only_return": np.concatenate([val_baseline, test_baseline]),
            "actual_residual": actual_residual,
            "predicted_residual": np.concatenate([val_pred_residual, test_pred_residual]),
            "absolute_error": np.concatenate(
                [
                    np.abs(predicted_val_return - actual_val_return),
                    np.abs(predicted_test_return - actual_test_return),
                ]
            ),
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
        "val_rmse_returns": rmse_val,
        "val_mae_returns": mae_val,
        "test_rmse_returns": rmse_test,
        "test_mae_returns": mae_test,
        "val_directional_accuracy": directional_accuracy,
        "arima_order": [int(part) for part in selected_arima_order],
        "train_sequences": int(len(bundle.X_train)),
        "val_sequences": int(len(bundle.X_val)),
        "test_sequences": int(len(bundle.X_test)),
        "train_split_index": int(train_end),
        "val_split_index": int(val_end),
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "early_stopped": bool(best_epoch < args.epochs),
        "test_sharpe": test_trading_metrics["sharpe"],
        "test_max_drawdown": test_trading_metrics["max_drawdown"],
        "test_win_rate": test_trading_metrics["win_rate"],
        "test_profit_factor": test_trading_metrics["profit_factor"],
        "test_average_trade_return": test_trading_metrics["average_trade_return"],
        "test_total_trades": test_trading_metrics["total_trades"],
    }
    LOGGER.info("[ARIMA-LSTM] best_epoch=%s metrics=%s", best_epoch, metrics)
    save_json(model_dir / "metrics.json", metrics)
    save_json(
        model_dir / "config.json",
        {
            "feature_columns": feature_columns,
            "window_size": args.reg_window,
            "arima_order": [int(part) for part in selected_arima_order],
            "hidden_size": args.lstm_hidden_size,
            "lstm_layers": args.lstm_layers,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        },
    )
    return RegressionArtifacts(metrics=metrics, history=history, prediction_frame=prediction_frame)


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
    symbol: str,
    bundle: SequenceDatasetBundle,
    feature_columns: list[str],
    focal_gamma: float,
    label_threshold: float,
    args: argparse.Namespace,
    model_dir: Path,
) -> ClassificationArtifacts:
    device = torch.device(args.device)
    counts_train, imbalance_ratio = summarize_class_counts(bundle.y_train, symbol)
    LOGGER.info("[CNN] class distribution train=%s", class_distribution(bundle.y_train))
    LOGGER.info("[CNN] class distribution val=%s", class_distribution(bundle.y_val))

    train_loader = make_loader(bundle.X_train, bundle.y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(bundle.X_val, bundle.y_val, args.batch_size)
    test_loader = make_loader(bundle.X_test, bundle.y_test, args.batch_size)

    model = DirectionCNNClassifier(
        time_steps=bundle.X_train.shape[-2],
        num_features=bundle.X_train.shape[-1],
        dropout=args.dropout,
    ).to(device)
    criterion = FocalLoss(
        gamma=focal_gamma,
        alpha=args.focal_alpha,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_val_bal_acc = -math.inf
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

        improved_bal_acc = val_bal_acc > best_val_bal_acc + 1e-6
        tie_bal_acc_better_loss = abs(val_bal_acc - best_val_bal_acc) <= 1e-6 and val_loss < best_val_loss - 1e-6
        if improved_bal_acc or tie_bal_acc_better_loss:
            best_val_loss = val_loss
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": feature_columns,
                    "window_size": args.cnn_window,
                    "dropout": args.dropout,
                    "neutral_threshold": label_threshold,
                    "focal_gamma": focal_gamma,
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
    test_pred, test_prob = tensor_predict(model, test_loader, device)
    confusion = confusion_matrix(bundle.y_val, val_pred, labels=[0, 1, 2])
    test_confusion = confusion_matrix(bundle.y_test, test_pred, labels=[0, 1, 2])
    class_acc = per_class_accuracy(confusion)
    test_class_acc = per_class_accuracy(test_confusion)
    val_pred_distribution = np.bincount(val_pred.astype(int), minlength=3).tolist()
    val_actual_distribution = np.bincount(bundle.y_val.astype(int), minlength=3).tolist()
    LOGGER.info("[%s] Predicted distribution: %s", model_dir.parent.name, val_pred_distribution)
    LOGGER.info("[%s] Actual distribution:    %s", model_dir.parent.name, val_actual_distribution)
    LOGGER.info(
        "[%s] Val Accuracy: %.4f | Val Balanced Accuracy: %.4f",
        model_dir.parent.name,
        accuracy_score(bundle.y_val, val_pred),
        balanced_accuracy_score(bundle.y_val, val_pred),
    )

    prediction_frame = pd.DataFrame(
        {
            "timestamp": [*bundle.val_timestamps, *bundle.test_timestamps],
            "dataset_split": ["val"] * len(bundle.val_timestamps) + ["test"] * len(bundle.test_timestamps),
            "true_label": [CLASS_NAMES[int(value)] for value in np.concatenate([bundle.y_val, bundle.y_test])],
            "predicted_label": [CLASS_NAMES[int(value)] for value in np.concatenate([val_pred, test_pred])],
            "up_prob": np.concatenate([val_prob[:, 0], test_prob[:, 0]]),
            "neutral_prob": np.concatenate([val_prob[:, 1], test_prob[:, 1]]),
            "down_prob": np.concatenate([val_prob[:, 2], test_prob[:, 2]]),
        }
    )

    prediction_frame.to_csv(model_dir / "val_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    save_json(model_dir / "training_history.json", history)
    pd.DataFrame(confusion, index=[CLASS_NAMES[i] for i in range(3)], columns=[CLASS_NAMES[i] for i in range(3)]).to_csv(
        model_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(
        test_confusion,
        index=[CLASS_NAMES[i] for i in range(3)],
        columns=[CLASS_NAMES[i] for i in range(3)],
    ).to_csv(model_dir / "test_confusion_matrix.csv")
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
        "best_val_balanced_accuracy": best_val_bal_acc,
        "val_accuracy": float(accuracy_score(bundle.y_val, val_pred)),
        "val_balanced_accuracy": float(balanced_accuracy_score(bundle.y_val, val_pred)),
        "test_accuracy": float(accuracy_score(bundle.y_test, test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(bundle.y_test, test_pred)),
        "recall_balance": compute_recall_balance(bundle.y_test, test_pred),
        "directional_coverage": compute_directional_coverage(test_pred),
        "train_windows": int(len(bundle.X_train)),
        "val_windows": int(len(bundle.X_val)),
        "test_windows": int(len(bundle.X_test)),
        "neutral_threshold": label_threshold,
        "focal_gamma": focal_gamma,
        "class_distribution_train": class_distribution(bundle.y_train),
        "class_distribution_val": class_distribution(bundle.y_val),
        "class_distribution_test": class_distribution(bundle.y_test),
        "class_distribution_train_pct": class_distribution_percentages(bundle.y_train),
        "class_distribution_val_pct": class_distribution_percentages(bundle.y_val),
        "class_distribution_test_pct": class_distribution_percentages(bundle.y_test),
        "predicted_distribution_val": {CLASS_NAMES[idx]: int(val_pred_distribution[idx]) for idx in range(3)},
        "actual_distribution_val": {CLASS_NAMES[idx]: int(val_actual_distribution[idx]) for idx in range(3)},
        "per_class_accuracy": class_acc,
        "per_class_accuracy_test": test_class_acc,
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "early_stopped": bool(best_epoch < args.epochs),
        "class_counts_train": counts_train.tolist(),
        "imbalance_ratio_train": imbalance_ratio,
    }
    if metrics["val_balanced_accuracy"] <= 0.34:
        LOGGER.warning(
            "[CNN] val_balanced_accuracy=%.4f is near random baseline; check class collapse and feature quality.",
            metrics["val_balanced_accuracy"],
        )
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
            "neutral_threshold": label_threshold,
            "focal_gamma": focal_gamma,
        },
    )
    save_json(model_dir / "per_class_accuracy.json", class_acc)
    return ClassificationArtifacts(metrics=metrics, history=history, prediction_frame=prediction_frame, confusion=confusion)


def add_volatility_regime_features(frame: pd.DataFrame, train_end: int) -> tuple[pd.DataFrame, int]:
    frame = frame.copy()
    window = max(3, min(20, max(train_end // 5, 3)))

    rolling_vol = frame["close"].rolling(window).std()
    vol_regime = pd.Series(1.0, index=frame.index, dtype=float)
    valid_vol = rolling_vol.dropna()
    if valid_vol.nunique() >= 3:
        try:
            buckets = pd.qcut(valid_vol, 3, labels=[0, 1, 2], duplicates="drop").astype(float)
            vol_regime.loc[buckets.index] = buckets.values
        except ValueError:
            LOGGER.warning("Failed to compute qcut volatility regime; falling back to neutral regime.")
    frame["vol_regime"] = vol_regime.ffill().fillna(1.0)

    ret_mean = frame["log_return"].rolling(window).mean()
    ret_std = frame["log_return"].rolling(window).std().replace(0.0, np.nan)
    frame["return_zscore"] = ((frame["log_return"] - ret_mean) / ret_std).replace([np.inf, -np.inf], np.nan)
    frame["return_zscore"] = frame["return_zscore"].ffill().fillna(0.0)

    volume_mean = frame["volume"].rolling(window).mean()
    volume_std = frame["volume"].rolling(window).std().replace(0.0, np.nan)
    frame["volume_zscore"] = ((frame["volume"] - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan)
    frame["volume_zscore"] = frame["volume_zscore"].ffill().fillna(0.0)

    frame["rolling_mean"] = frame["close"].rolling(window, min_periods=1).mean()
    return frame, window


def run_for_symbol(
    symbol: str,
    args: argparse.Namespace,
    root_run_dir: Path,
    fx_context_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    symbol_dir = root_run_dir / sanitize_symbol(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    frame, source_name, diagnostics = load_symbol_frame(symbol, args, fx_context_frame=fx_context_frame)
    LOGGER.info("Loaded %s rows for %s from %s", len(frame), symbol, source_name)
    LOGGER.info("[%s] Raw rows loaded: %s", symbol, diagnostics["raw_rows_loaded"])
    LOGGER.info("[%s] Rows after ffill + dropna: %s", symbol, diagnostics["rows_after_ffill_dropna"])

    assert len(frame) > 400, (
        f"{symbol} only has {len(frame)} rows after preprocessing. "
        f"Check ffill and feature engineering nulls."
    )

    pre_feature_train_end = int(len(frame) * 0.70)
    frame, regime_window = add_volatility_regime_features(frame, pre_feature_train_end)
    check_nulls(frame, symbol, f"after regime features (window={regime_window})")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill()
    required_after_regime = [column for column in ["open", "high", "low", "close", "volume", "log_return"] if column in frame.columns]
    if required_after_regime:
        frame = frame.dropna(subset=required_after_regime)
    frame = frame.reset_index(drop=True)

    assert len(frame) > 400, (
        f"{symbol} only has {len(frame)} rows after regime feature cleanup. "
        f"Check ffill and feature engineering nulls."
    )

    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    if train_end <= max(args.reg_window, args.cnn_window):
        raise ValueError("Training split leaves too little history for the requested windows.")
    if val_end <= train_end:
        raise ValueError("Validation split is invalid; val_end must be greater than train_end.")
    if n_rows - val_end <= max(20, args.cnn_window):
        raise ValueError("Test split is too small after chronological split.")

    feature_table = build_feature_table(frame)
    regression_columns = select_regression_columns(feature_table)
    cnn_columns = select_cnn_columns(feature_table)
    if len(regression_columns) < 5:
        raise ValueError("Regression feature set is too small after filtering.")
    if len(cnn_columns) < 5:
        raise ValueError("CNN feature set is too small after filtering.")

    reg_feature_frame = feature_table[regression_columns].copy()
    cnn_feature_frame = feature_table[cnn_columns].copy()

    reg_scaled, reg_scaler, reg_medians = impute_and_scale_features(reg_feature_frame, train_end)
    cnn_scaled, _, _ = impute_and_scale_features(cnn_feature_frame, train_end)

    train_log_returns = frame["log_return"].iloc[:train_end].dropna().values
    dynamic_threshold, neutral_pct = find_threshold(frame.iloc[:train_end], symbol, args)
    next_log_returns = frame["log_return"].shift(-1).astype(float).values

    split_label_inputs = {
        "train": next_log_returns[:train_end],
        "val": next_log_returns[train_end:val_end],
        "test": next_log_returns[val_end:],
    }
    split_label_stats: dict[str, dict[str, float]] = {}
    for split_name, split_returns in split_label_inputs.items():
        finite_returns = split_returns[np.isfinite(split_returns)]
        if finite_returns.size == 0:
            continue
        split_labels = apply_labels(finite_returns, dynamic_threshold, args)
        up_pct = float((split_labels == 0).mean())
        neutral_pct_split = float((split_labels == 1).mean())
        down_pct = float((split_labels == 2).mean())
        split_label_stats[split_name] = {
            "up": up_pct,
            "neutral": neutral_pct_split,
            "down": down_pct,
        }
        LOGGER.info(
            "[%s] %s labels: up=%.1f%% neutral=%.1f%% down=%.1f%%",
            symbol,
            split_name,
            up_pct * 100.0,
            neutral_pct_split * 100.0,
            down_pct * 100.0,
        )
        if neutral_pct_split > 0.40:
            LOGGER.warning("[%s] neutral class still high in %s (%.1f%%)", symbol, split_name, neutral_pct_split * 100.0)
        if up_pct < 0.25 or down_pct < 0.25:
            LOGGER.warning("[%s] up/down class is small in %s (up=%.1f%% down=%.1f%%)", symbol, split_name, up_pct * 100.0, down_pct * 100.0)

    arima_order = tuple(int(part.strip()) for part in args.arima_order.split(","))
    arima_result, arima_baseline, selected_arima_order = compute_arima_predictions(
        symbol=symbol,
        log_return=frame["log_return"],
        train_end=train_end,
        arima_order=arima_order,
    )
    residual_target = frame["log_return"].values.astype(float) - arima_baseline

    reg_bundle = build_regression_sequences(
        scaled_features=reg_scaled,
        residual_target=residual_target,
        timestamps=frame["timestamp"],
        close_values=frame["close"].values.astype(float),
        arima_baseline=arima_baseline,
        train_end=train_end,
        val_end=val_end,
        window_size=args.reg_window,
    )
    cnn_bundle = build_cnn_sequences(
        feature_values=cnn_scaled,
        close_values=frame["close"].values.astype(float),
        timestamps=frame["timestamp"],
        train_end=train_end,
        val_end=val_end,
        window_size=args.cnn_window,
        args=args,
        neutral_threshold=None,
        next_log_returns=next_log_returns,
        label_threshold=dynamic_threshold,
    )

    class_counts, imbalance_ratio = summarize_class_counts(cnn_bundle.y_train, symbol)
    focal_gamma = FIXED_FOCAL_GAMMA

    if args.disable_cnn_cv:
        cv_mean = float("nan")
        cv_std = float("nan")
        cv_folds = 0
    else:
        X_full_cnn, y_full_cnn = build_cnn_full_sequences(
            feature_values=cnn_scaled,
            next_log_returns=next_log_returns,
            window_size=args.cnn_window,
            label_threshold=dynamic_threshold,
            args=args,
        )
        cv_mean, cv_std, cv_folds = run_cnn_walk_forward_cv(
            symbol=symbol,
            X_full=X_full_cnn,
            y_full=y_full_cnn,
            args=args,
        )

    LOGGER.info("Regression features=%s", regression_columns)
    LOGGER.info("CNN features=%s", cnn_columns)
    LOGGER.info(
        "[%s] Train/Val/Test rows: %s / %s / %s",
        symbol,
        train_end,
        val_end - train_end,
        len(frame) - val_end,
    )
    LOGGER.info(
        "Prepared sequences for %s: reg_train=%s reg_val=%s reg_test=%s cnn_train=%s cnn_val=%s cnn_test=%s",
        symbol,
        len(reg_bundle.X_train),
        len(reg_bundle.X_val),
        len(reg_bundle.X_test),
        len(cnn_bundle.X_train),
        len(cnn_bundle.X_val),
        len(cnn_bundle.X_test),
    )
    LOGGER.info(
        "[%s] CNN class %% train=%s val=%s test=%s",
        symbol,
        class_distribution_percentages(cnn_bundle.y_train),
        class_distribution_percentages(cnn_bundle.y_val),
        class_distribution_percentages(cnn_bundle.y_test),
    )
    LOGGER.info(
        "CNN class distribution overall=%s",
        class_distribution(np.concatenate([cnn_bundle.y_train, cnn_bundle.y_val, cnn_bundle.y_test])),
    )

    regression_dir = symbol_dir / "arima_lstm"
    classification_dir = symbol_dir / "cnn_pattern"
    regression_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    regression_artifacts = train_regression_model(
        bundle=reg_bundle,
        baseline_predictions=arima_baseline,
        log_return_values=frame["log_return"].values.astype(float),
        feature_columns=regression_columns,
        timestamps=frame["timestamp"],
        train_end=train_end,
        val_end=val_end,
        args=args,
        model_dir=regression_dir,
        imputation_medians=reg_medians,
        feature_scaler=reg_scaler,
        arima_result=arima_result,
        selected_arima_order=selected_arima_order,
    )
    classification_artifacts = train_cnn_model(
        symbol=symbol,
        bundle=cnn_bundle,
        feature_columns=cnn_columns,
        focal_gamma=focal_gamma,
        label_threshold=dynamic_threshold,
        args=args,
        model_dir=classification_dir,
    )

    LOGGER.info(
        "[%s] ARIMA-LSTM RMSE: %.4f | MAE: %.4f | order=%s",
        symbol,
        regression_artifacts.metrics["val_rmse_returns"],
        regression_artifacts.metrics["val_mae_returns"],
        regression_artifacts.metrics["arima_order"],
    )
    LOGGER.info(
        "[%s] CNN Val Accuracy: %.4f | Val Balanced Accuracy: %.4f",
        symbol,
        classification_artifacts.metrics["val_accuracy"],
        classification_artifacts.metrics["val_balanced_accuracy"],
    )
    LOGGER.info(
        "[%s] Epochs trained: reg=%s cnn=%s | reg train/val loss=%.6f/%.6f | cnn train/val loss=%.6f/%.6f",
        symbol,
        regression_artifacts.metrics["epochs_ran"],
        classification_artifacts.metrics["epochs_ran"],
        regression_artifacts.metrics["final_train_loss"] or 0.0,
        regression_artifacts.metrics["final_val_loss"] or 0.0,
        classification_artifacts.metrics["final_train_loss"] or 0.0,
        classification_artifacts.metrics["final_val_loss"] or 0.0,
    )
    LOGGER.info(
        "[%s] Test trading metrics: sharpe=%.4f max_drawdown=%.4f win_rate=%.4f",
        symbol,
        regression_artifacts.metrics["test_sharpe"],
        regression_artifacts.metrics["test_max_drawdown"],
        regression_artifacts.metrics["test_win_rate"],
    )

    test_balanced_accuracy = float(classification_artifacts.metrics["test_balanced_accuracy"])
    symbol_config = {
        "threshold": float(dynamic_threshold),
        "neutral_pct": float(neutral_pct),
        "gamma": float(focal_gamma),
        "class_counts": class_counts.tolist(),
        "imbalance_ratio": float(imbalance_ratio),
        "label_split_distribution": split_label_stats,
        "cv_folds": int(cv_folds),
    }
    symbol_results = {
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "test_bal_acc": test_balanced_accuracy,
    }

    summary = {
        "symbol": symbol,
        "source": source_name,
        "rows": int(len(frame)),
        "raw_rows_loaded": diagnostics["raw_rows_loaded"],
        "rows_after_ffill_dropna": diagnostics["rows_after_ffill_dropna"],
        "rows_after_feature_engineering": diagnostics["rows_after_feature_engineering"],
        "train_rows": int(train_end),
        "val_rows": int(val_end - train_end),
        "test_rows": int(len(frame) - val_end),
        "class_distribution_train_pct": class_distribution_percentages(cnn_bundle.y_train),
        "class_distribution_val_pct": class_distribution_percentages(cnn_bundle.y_val),
        "class_distribution_test_pct": class_distribution_percentages(cnn_bundle.y_test),
        "symbol_config": symbol_config,
        "symbol_results": symbol_results,
        "model_paths": {
            "arima_lstm": str(regression_dir / "best_lstm_checkpoint.pt"),
            "cnn_pattern": str(classification_dir / "best_cnn_checkpoint.pt"),
        },
        "scaler_paths": {
            "feature_scaler": str(regression_dir / "feature_scaler.pkl"),
            "target_scaler": str(regression_dir / "target_scaler.pkl"),
        },
        "regression_metrics": regression_artifacts.metrics,
        "classification_metrics": classification_artifacts.metrics,
    }
    save_json(symbol_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.symbols = dedupe_symbols(args.symbols or [])
    set_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root_run_dir = Path(args.output_root) / f"regime_aware_{args.interval}_{timestamp}"
    configure_logging(root_run_dir)

    LOGGER.info("Starting regime-aware training run at %s", root_run_dir)
    LOGGER.info("Args: %s", vars(args))
    fx_context_frame = _load_fx_context_frame(args)
    if fx_context_frame.empty:
        raise ValueError(f"FX context symbol {args.fx_context_symbol} must be available before training.")
    LOGGER.info(
        "Loaded FX context symbol %s with %s rows before training.",
        args.fx_context_symbol,
        len(fx_context_frame),
    )

    all_summaries: list[dict[str, Any]] = []
    failed_symbols: list[dict[str, str]] = []
    skipped_symbols: list[dict[str, str]] = []
    SYMBOL_CONFIGS: dict[str, dict[str, Any]] = {}
    SYMBOL_MODELS: dict[str, dict[str, str]] = {}
    SYMBOL_SCALERS: dict[str, dict[str, str]] = {}
    SYMBOL_RESULTS: dict[str, dict[str, float]] = {}

    def validate_symbol(symbol: str):
        try:
            frame, _, diagnostics = load_symbol_frame(symbol, args, fx_context_frame=fx_context_frame)
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        outcome = validate_equity_symbol(
            symbol=symbol,
            frame=frame,
            interval=args.interval,
            split_counts=quality_gate_split_counts(frame),
        )
        outcome.diagnostics.update(diagnostics)
        return outcome

    discovery = discover_training_symbols(
        interval=args.interval,
        requested_symbols=args.symbols or None,
        database_url=args.database_url,
        validator=validate_symbol,
        print_fn=lambda message: LOGGER.info(message),
    )
    skipped_symbols.extend(
        [
            {"symbol": symbol, "reason": discovery.skipped_reasons[symbol], "details": str(discovery.diagnostics.get(symbol, {}))}
            for symbol in discovery.skipped_symbols
        ]
    )
    training_symbols = list(discovery.active_symbols)
    if not training_symbols:
        raise ValueError("No active equity symbols passed the training quality gate.")
    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
        LOGGER.info("\n%s", "=" * 60)
        LOGGER.info("Processing: %s", symbol)
        LOGGER.info("%s", "=" * 60)
        try:
            summary = run_for_symbol(symbol, args, root_run_dir, fx_context_frame=fx_context_frame)
            all_summaries.append(summary)
            SYMBOL_CONFIGS[symbol] = summary.get("symbol_config", {})
            SYMBOL_MODELS[symbol] = summary.get("model_paths", {})
            SYMBOL_SCALERS[symbol] = summary.get("scaler_paths", {})
            SYMBOL_RESULTS[symbol] = summary.get("symbol_results", {})
            LOGGER.info("Completed symbol=%s summary=%s", symbol, summary)
        except Exception as exc:
            LOGGER.exception("Training failed for %s: %s", symbol, exc)
            failed_symbols.append({"symbol": symbol, "error": str(exc)})

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(root_run_dir),
        "symbols_requested": training_symbols,
        "symbols_succeeded": [summary["symbol"] for summary in all_summaries],
        "symbols_skipped": skipped_symbols,
        "symbols_failed": failed_symbols,
        "symbol_configs": SYMBOL_CONFIGS,
        "symbol_models": SYMBOL_MODELS,
        "symbol_scalers": SYMBOL_SCALERS,
        "symbol_results": SYMBOL_RESULTS,
        "config": vars(args),
        "summaries": all_summaries,
    }
    save_json(root_run_dir / "run_manifest.json", manifest)

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("CROSS-SYMBOL SUMMARY")
    LOGGER.info("%s", "=" * 80)
    LOGGER.info(
        "%-15s %7s %6s %9s %8s %7s %13s %8s",
        "Symbol",
        "Thresh",
        "Gamma",
        "Neutral%",
        "CV Mean",
        "CV Std",
        "Test Bal Acc",
        "Status",
    )
    LOGGER.info("%s", "-" * 80)
    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
        cfg = SYMBOL_CONFIGS.get(symbol)
        res = SYMBOL_RESULTS.get(symbol)
        if not cfg or not res:
            LOGGER.info("%-15s %7s %6s %9s %8s %7s %13s %8s", symbol, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "WARN")
            continue
        cv_mean = float(res.get("cv_mean", float("nan")))
        cv_std = float(res.get("cv_std", float("nan")))
        test_bal = float(res.get("test_bal_acc", float("nan")))
        status = "OK" if np.isfinite(cv_mean) and np.isfinite(cv_std) and (cv_mean >= 0.38 and cv_std <= 0.05) else "WARN"
        LOGGER.info(
            "%-15s %7.3f %6.2f %8.1f%% %8.4f %7.4f %13.4f %8s",
            symbol,
            float(cfg.get("threshold", float("nan"))),
            float(cfg.get("gamma", float("nan"))),
            float(cfg.get("neutral_pct", 0.0)) * 100.0,
            cv_mean,
            cv_std,
            test_bal,
            status,
        )
    LOGGER.info(FX_RESULTS_NOTE)

    if failed_symbols:
        raise SystemExit(1)

    LOGGER.info("All training jobs completed successfully.")


if __name__ == "__main__":
    main()
