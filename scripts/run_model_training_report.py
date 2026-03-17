from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sqlalchemy import text

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at runtime
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts import train_regime_aware as trainer
from src.agents.regime.data_loader import RegimeDataLoader
from src.agents.technical.backtest import TechnicalBacktester, WalkForwardConfig
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from src.db.connection import get_engine


LOGGER = logging.getLogger("run_model_training_report")
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["up", "neutral", "down"]
DIRECTION_LABELS = [1, 0, -1]
DIRECTION_NAMES = ["up", "neutral", "down"]


@dataclass
class SymbolDataset:
    symbol: str
    source_name: str
    raw_frame: pd.DataFrame
    clean_frame: pd.DataFrame
    engineered_frame: pd.DataFrame
    feature_table: pd.DataFrame
    regression_columns: list[str]
    cnn_columns: list[str]
    split_idx: int
    feature_coverage_pct: dict[str, float]
    rows_dropped_initial_clean: int
    rows_dropped_feature_engineering: int
    rows_with_missing_features_pre_impute: int
    cnn_total_windows: int
    cnn_train_windows: int
    cnn_val_windows: int
    macro_quality_report: dict[str, dict]
    macro_excluded_features: list[str]
    training_window: tuple[str, str] | None
    validation_window: tuple[str, str] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset readiness, run training, evaluate, and write a model training report."
    )
    parser.add_argument("--symbols", type=str, help="Optional comma-separated symbol filter.")
    parser.add_argument("--interval", default="1h", help="Target interval.")
    parser.add_argument("--limit", type=int, default=4000, help="Maximum rows to load per symbol.")
    parser.add_argument("--min-rows", type=int, default=300, help="Minimum rows required for training.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split used by the trainer.")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=10, help="Scheduler patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--reg-window", type=int, default=20, help="ARIMA-LSTM window size.")
    parser.add_argument("--cnn-window", type=int, default=30, help="CNN window size.")
    parser.add_argument("--neutral-threshold", type=float, default=0.0045, help="Neutral threshold for classifier labels.")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order as p,d,q.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--output-root",
        default="data/reports/training_runs",
        help="Root directory for training artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory where the markdown report and charts will be written.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Optional DATABASE_URL override.",
    )
    parser.add_argument("--gold-dir", default="data/gold", help="Gold feature fallback directory.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _report_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def discover_quality_rows(args: argparse.Namespace) -> pd.DataFrame:
    engine = get_engine(args.database_url)
    query = """
        SELECT *
        FROM market_data_quality
        WHERE dataset_type = 'historical'
          AND interval = :interval
          AND exchange = 'NSE'
          AND asset_type = 'equity'
        ORDER BY symbol ASC
    """
    df = pd.read_sql(text(query), engine, params={"interval": args.interval})
    if df.empty:
        raise ValueError(f"No NSE equity historical quality rows found for interval={args.interval}.")

    for column in ("first_timestamp", "last_timestamp", "updated_at"):
        if column in df.columns and not df.empty:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    if args.symbols:
        requested = {value.strip().upper() for value in args.symbols.split(",") if value.strip()}
        df = df[df["symbol"].str.upper().isin(requested)].copy()
    if df.empty:
        raise ValueError("No symbols matched the requested filter after applying quality metadata.")
    return df.reset_index(drop=True)


def build_training_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=None,
        database_url=args.database_url,
        gold_dir=args.gold_dir,
        interval=args.interval,
        limit=args.limit,
        output_root=args.output_root,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        reg_window=args.reg_window,
        cnn_window=args.cnn_window,
        neutral_threshold=args.neutral_threshold,
        arima_order=args.arima_order,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        train_split=args.train_split,
        min_rows=args.min_rows,
    )


def load_source_frame(symbol: str, train_args: argparse.Namespace) -> tuple[pd.DataFrame, str, dict[str, dict], list[str]]:
    gold_loader = RegimeDataLoader(database_url=train_args.database_url, gold_dir=train_args.gold_dir)
    frame = gold_loader.load_features(symbol=symbol, limit=train_args.limit)
    source_name = "gold_features"
    macro_quality_report: dict[str, dict] = {}
    macro_excluded_features: list[str] = []

    if frame.empty or not trainer.REQUIRED_OHLCV.issubset(frame.columns) or len(frame) < train_args.min_rows:
        raw_loader = DataLoader(
            train_args.database_url or os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
        )
        frame = raw_loader.load_historical_bars(
            symbol=symbol,
            limit=train_args.limit,
            use_nse_fallback=False,
            interval=train_args.interval,
            include_macro=True,
        )
        source_name = "ohlcv_bars"
        macro_quality_report = getattr(raw_loader, "last_macro_quality_report", {}) or {}
        macro_excluded_features = getattr(raw_loader, "last_macro_excluded_features", []) or []

    if frame.empty:
        raise ValueError(f"No rows available for {symbol} from gold_features or ohlcv_bars.")
    return frame.copy(), source_name, macro_quality_report, macro_excluded_features


def prepare_symbol_dataset(symbol: str, train_args: argparse.Namespace) -> SymbolDataset:
    raw_frame, source_name, macro_quality_report, macro_excluded_features = load_source_frame(symbol, train_args)
    raw_rows = len(raw_frame)

    frame = raw_frame.copy()
    if "timestamp" not in frame.columns:
        raise ValueError(f"Dataset for {symbol} must include a timestamp column.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

    missing_ohlcv = trainer.REQUIRED_OHLCV - set(frame.columns)
    if missing_ohlcv:
        raise ValueError(f"Dataset for {symbol} is missing required columns: {sorted(missing_ohlcv)}")

    for column in trainer.REQUIRED_OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    clean_rows = len(frame)
    rows_dropped_initial_clean = max(0, raw_rows - clean_rows)

    if clean_rows < train_args.min_rows:
        raise ValueError(f"Only {clean_rows} clean rows available for {symbol}; need at least {train_args.min_rows}.")

    is_forex = symbol.endswith("=X")
    engineered = engineer_features(frame, is_forex=is_forex).reset_index(drop=True)
    engineered["symbol"] = symbol
    engineered["daily_return"] = engineered["close"].pct_change()
    engineered["log_close"] = np.log(engineered["close"].clip(lower=1e-8))

    feature_table = trainer.build_feature_table(engineered)
    regression_columns = trainer.select_regression_columns(feature_table)
    cnn_columns = trainer.select_cnn_columns(feature_table)
    if len(regression_columns) < 5:
        raise ValueError(f"Regression feature set is too small for {symbol}.")
    if len(cnn_columns) < 5:
        raise ValueError(f"CNN feature set is too small for {symbol}.")

    union_columns = sorted(dict.fromkeys([*regression_columns, *cnn_columns]))
    coverage_frame = feature_table[union_columns].copy()
    feature_coverage_pct = {
        column: round(float(coverage_frame[column].notna().mean() * 100.0), 4)
        for column in union_columns
    }
    rows_with_missing_features_pre_impute = int(coverage_frame.isna().any(axis=1).sum())
    rows_dropped_feature_engineering = max(0, clean_rows - len(engineered))

    split_idx = int(len(engineered) * train_args.train_split)
    if split_idx <= max(train_args.reg_window, train_args.cnn_window):
        raise ValueError(f"Training split for {symbol} leaves too little history for requested windows.")
    if len(engineered) - split_idx <= 20:
        raise ValueError(f"Validation split for {symbol} is too small.")

    cnn_total_windows = max(0, len(engineered) - train_args.cnn_window)
    cnn_train_windows = max(0, split_idx - train_args.cnn_window)
    cnn_val_windows = max(0, len(engineered) - max(split_idx, train_args.cnn_window))

    training_window = None
    validation_window = None
    if split_idx > 0:
        training_window = (
            engineered["timestamp"].iloc[0].isoformat(),
            engineered["timestamp"].iloc[split_idx - 1].isoformat(),
        )
    if split_idx < len(engineered):
        validation_window = (
            engineered["timestamp"].iloc[split_idx].isoformat(),
            engineered["timestamp"].iloc[-1].isoformat(),
        )

    return SymbolDataset(
        symbol=symbol,
        source_name=source_name,
        raw_frame=raw_frame,
        clean_frame=frame,
        engineered_frame=engineered,
        feature_table=feature_table,
        regression_columns=regression_columns,
        cnn_columns=cnn_columns,
        split_idx=split_idx,
        feature_coverage_pct=feature_coverage_pct,
        rows_dropped_initial_clean=rows_dropped_initial_clean,
        rows_dropped_feature_engineering=rows_dropped_feature_engineering,
        rows_with_missing_features_pre_impute=rows_with_missing_features_pre_impute,
        cnn_total_windows=cnn_total_windows,
        cnn_train_windows=cnn_train_windows,
        cnn_val_windows=cnn_val_windows,
        macro_quality_report=macro_quality_report,
        macro_excluded_features=macro_excluded_features,
        training_window=training_window,
        validation_window=validation_window,
    )


def aggregate_feature_coverage(datasets: list[SymbolDataset]) -> dict[str, float]:
    numerator: dict[str, int] = {}
    denominator: dict[str, int] = {}
    for dataset in datasets:
        for column, coverage_pct in dataset.feature_coverage_pct.items():
            total_rows = len(dataset.feature_table)
            non_null_rows = int(round((coverage_pct / 100.0) * total_rows))
            numerator[column] = numerator.get(column, 0) + non_null_rows
            denominator[column] = denominator.get(column, 0) + total_rows
    return {
        column: round((100.0 * numerator[column] / denominator[column]), 4)
        for column in sorted(denominator)
        if denominator[column] > 0
    }


def aggregate_readiness(datasets: list[SymbolDataset], quality_df: pd.DataFrame) -> dict[str, Any]:
    rows_per_symbol = []
    total_rows = 0
    total_feature_rows = 0
    total_rows_dropped_initial = 0
    total_rows_missing_features = 0
    total_cnn_windows = 0
    total_cnn_train_windows = 0
    total_cnn_val_windows = 0
    macro_excluded_union: set[str] = set()

    for dataset in datasets:
        total_rows += len(dataset.clean_frame)
        total_feature_rows += len(dataset.feature_table)
        total_rows_dropped_initial += dataset.rows_dropped_initial_clean
        total_rows_missing_features += dataset.rows_with_missing_features_pre_impute
        total_cnn_windows += dataset.cnn_total_windows
        total_cnn_train_windows += dataset.cnn_train_windows
        total_cnn_val_windows += dataset.cnn_val_windows
        macro_excluded_union.update(dataset.macro_excluded_features)
        rows_per_symbol.append(
            {
                "symbol": dataset.symbol,
                "source": dataset.source_name,
                "raw_rows": int(len(dataset.raw_frame)),
                "clean_rows": int(len(dataset.clean_frame)),
                "feature_rows": int(len(dataset.feature_table)),
                "rows_dropped_initial_clean": int(dataset.rows_dropped_initial_clean),
                "rows_dropped_feature_engineering": int(dataset.rows_dropped_feature_engineering),
                "rows_with_missing_features_pre_impute": int(dataset.rows_with_missing_features_pre_impute),
                "cnn_total_windows": int(dataset.cnn_total_windows),
                "cnn_train_windows": int(dataset.cnn_train_windows),
                "cnn_val_windows": int(dataset.cnn_val_windows),
                "training_window": dataset.training_window,
                "validation_window": dataset.validation_window,
            }
        )

    excluded = quality_df[quality_df["train_ready"] != True].copy()
    feature_coverage = aggregate_feature_coverage(datasets)

    train_window_start = min((dataset.training_window[0] for dataset in datasets if dataset.training_window), default=None)
    train_window_end = max((dataset.training_window[1] for dataset in datasets if dataset.training_window), default=None)
    validation_window_start = min((dataset.validation_window[0] for dataset in datasets if dataset.validation_window), default=None)
    validation_window_end = max((dataset.validation_window[1] for dataset in datasets if dataset.validation_window), default=None)

    return {
        "total_symbols_available": int(len(quality_df)),
        "symbols_train_ready": int((quality_df["train_ready"] == True).sum()),
        "symbols_excluded_by_quality": excluded["symbol"].astype(str).tolist(),
        "rows_per_symbol": rows_per_symbol,
        "total_rows": int(total_rows),
        "total_feature_rows": int(total_feature_rows),
        "total_rows_dropped_initial_clean": int(total_rows_dropped_initial),
        "total_rows_with_missing_features_pre_impute": int(total_rows_missing_features),
        "total_cnn_windows": int(total_cnn_windows),
        "total_cnn_train_windows": int(total_cnn_train_windows),
        "total_cnn_val_windows": int(total_cnn_val_windows),
        "feature_coverage_pct": feature_coverage,
        "macro_features_excluded_by_loader": sorted(macro_excluded_union),
        "final_training_window": [train_window_start, train_window_end],
        "validation_window": [validation_window_start, validation_window_end],
    }


def run_training(
    symbols: list[str],
    datasets: dict[str, SymbolDataset],
    train_args: argparse.Namespace,
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    trainer.set_seed(train_args.seed)
    trainer.configure_logging(run_root)
    LOGGER.info("Training run root: %s", run_root)

    for symbol in symbols:
        LOGGER.info("Training symbol=%s", symbol)
        try:
            summary = run_for_dataset(datasets[symbol], train_args, run_root)
            summaries.append(summary)
        except Exception as exc:
            LOGGER.exception("Training failed for %s: %s", symbol, exc)
            failures.append({"symbol": symbol, "error": str(exc)})
    return summaries, failures


def run_for_dataset(
    dataset: SymbolDataset,
    train_args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    symbol = dataset.symbol
    symbol_dir = run_root / trainer.sanitize_symbol(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    frame = dataset.engineered_frame.copy()
    split_idx = dataset.split_idx

    reg_feature_frame = dataset.feature_table[dataset.regression_columns].copy()
    cnn_feature_frame = dataset.feature_table[dataset.cnn_columns].copy()

    reg_scaled, reg_scaler, reg_medians = trainer.impute_and_scale_features(reg_feature_frame, split_idx)
    cnn_scaled, _, _ = trainer.impute_and_scale_features(cnn_feature_frame, split_idx)

    arima_order = tuple(int(part.strip()) for part in train_args.arima_order.split(","))
    arima_result, arima_baseline = trainer.compute_arima_predictions(frame["log_close"], split_idx, arima_order)
    residual_target = frame["log_close"].values - arima_baseline

    reg_bundle = trainer.build_regression_sequences(
        scaled_features=reg_scaled,
        residual_target=residual_target,
        timestamps=frame["timestamp"],
        close_values=frame["close"].values.astype(float),
        arima_baseline=arima_baseline,
        split_idx=split_idx,
        window_size=train_args.reg_window,
    )
    cnn_bundle = trainer.build_cnn_sequences(
        feature_values=cnn_scaled,
        close_values=frame["close"].values.astype(float),
        timestamps=frame["timestamp"],
        split_idx=split_idx,
        window_size=train_args.cnn_window,
        neutral_threshold=train_args.neutral_threshold,
    )

    LOGGER.info("Loaded %s rows for %s from %s", len(frame), symbol, dataset.source_name)
    LOGGER.info("Regression features=%s", dataset.regression_columns)
    LOGGER.info("CNN features=%s", dataset.cnn_columns)
    LOGGER.info(
        "Prepared sequences for %s: reg_train=%s reg_val=%s cnn_train=%s cnn_val=%s",
        symbol,
        len(reg_bundle.X_train),
        len(reg_bundle.X_val),
        len(cnn_bundle.X_train),
        len(cnn_bundle.X_val),
    )
    LOGGER.info(
        "CNN class distribution overall=%s",
        trainer.class_distribution(np.concatenate([cnn_bundle.y_train, cnn_bundle.y_val])),
    )

    regression_dir = symbol_dir / "arima_lstm"
    classification_dir = symbol_dir / "cnn_pattern"
    regression_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    regression_artifacts = trainer.train_regression_model(
        bundle=reg_bundle,
        baseline_predictions=arima_baseline,
        close_values=frame["close"].values.astype(float),
        log_close_values=frame["log_close"].values.astype(float),
        feature_columns=dataset.regression_columns,
        timestamps=frame["timestamp"],
        split_idx=split_idx,
        args=train_args,
        model_dir=regression_dir,
        imputation_medians=reg_medians,
        feature_scaler=reg_scaler,
        arima_result=arima_result,
    )
    classification_artifacts = trainer.train_cnn_model(
        bundle=cnn_bundle,
        feature_columns=dataset.cnn_columns,
        args=train_args,
        model_dir=classification_dir,
    )

    summary = {
        "symbol": symbol,
        "source": dataset.source_name,
        "rows": int(len(frame)),
        "train_rows": int(split_idx),
        "val_rows": int(len(frame) - split_idx),
        "regression_metrics": regression_artifacts.metrics,
        "classification_metrics": classification_artifacts.metrics,
    }
    trainer.save_json(symbol_dir / "summary.json", summary)
    return summary


def _safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_validation_metrics(
    run_root: Path,
    symbols: list[str],
) -> dict[str, Any]:
    all_true: list[str] = []
    all_pred: list[str] = []
    train_accuracy_weighted_sum = 0.0
    train_weight_total = 0
    val_accuracy_weighted_sum = 0.0
    val_weight_total = 0
    overfit_gaps: list[dict[str, float]] = []

    for symbol in symbols:
        symbol_dir = run_root / trainer.sanitize_symbol(symbol) / "cnn_pattern"
        predictions_path = symbol_dir / "val_predictions.csv"
        history_path = symbol_dir / "training_history.json"
        metrics_path = symbol_dir / "metrics.json"
        if not predictions_path.exists() or not history_path.exists() or not metrics_path.exists():
            continue

        predictions = pd.read_csv(predictions_path)
        metrics = _safe_read_json(metrics_path)
        history = _safe_read_json(history_path)

        all_true.extend(predictions["true_label"].astype(str).tolist())
        all_pred.extend(predictions["predicted_label"].astype(str).tolist())

        best_epoch = int(metrics["best_epoch"])
        history_by_epoch = {int(row["epoch"]): row for row in history}
        best_row = history_by_epoch.get(best_epoch, history[-1])
        train_accuracy = float(best_row.get("train_accuracy", 0.0))
        val_accuracy = float(best_row.get("val_accuracy", metrics.get("val_accuracy", 0.0)))
        val_windows = int(metrics.get("val_windows", len(predictions)))
        train_windows = int(metrics.get("train_windows", 0))
        train_accuracy_weighted_sum += train_accuracy * train_windows
        train_weight_total += train_windows
        val_accuracy_weighted_sum += val_accuracy * val_windows
        val_weight_total += val_windows
        overfit_gaps.append({"symbol": symbol, "gap": train_accuracy - val_accuracy})

    if not all_true:
        raise ValueError("Validation predictions were not generated for any symbol.")

    accuracy = float(accuracy_score(all_true, all_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        labels=CLASS_NAMES,
        average="weighted",
        zero_division=0,
    )
    confusion = confusion_matrix(all_true, all_pred, labels=CLASS_NAMES)

    return {
        "training_accuracy": round(train_accuracy_weighted_sum / max(train_weight_total, 1), 6),
        "validation_accuracy": round(val_accuracy_weighted_sum / max(val_weight_total, 1), 6),
        "validation_accuracy_recomputed": round(accuracy, 6),
        "precision_weighted": round(float(precision), 6),
        "recall_weighted": round(float(recall), 6),
        "f1_weighted": round(float(f1), 6),
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": CLASS_NAMES,
        "prediction_distribution": {
            label: int(sum(pred == label for pred in all_pred))
            for label in CLASS_NAMES
        },
        "class_balance": {
            label: int(sum(true == label for true in all_true))
            for label in CLASS_NAMES
        },
        "overfitting_gap": round(
            (train_accuracy_weighted_sum / max(train_weight_total, 1))
            - (val_accuracy_weighted_sum / max(val_weight_total, 1)),
            6,
        ),
        "per_symbol_overfitting_gap": overfit_gaps,
    }


def build_backtest_config(frame: pd.DataFrame) -> WalkForwardConfig:
    if len(frame) < 4000:
        start_date = frame["timestamp"].min().strftime("%Y-%m-%d")
        return WalkForwardConfig(
            train_days=15,
            test_days=3,
            step_days=3,
            start_date=start_date,
        )
    return WalkForwardConfig(train_months=6, test_months=1, step_months=1, start_date="2019-01-01")


def direction_series_to_names(values: pd.Series) -> list[str]:
    mapping = {1: "up", 0: "neutral", -1: "down"}
    return [mapping.get(int(value), "neutral") for value in values.fillna(0).astype(int)]


def evaluate_test_predictions(
    datasets: dict[str, SymbolDataset],
    symbols: list[str],
) -> dict[str, Any]:
    all_predictions: list[pd.DataFrame] = []
    per_symbol: list[dict[str, Any]] = []

    for symbol in symbols:
        dataset = datasets[symbol]
        config = build_backtest_config(dataset.clean_frame)
        backtester = TechnicalBacktester(config=config)
        prepared = backtester._prepare_market_df(dataset.clean_frame.copy())
        splits = backtester.generate_walk_forward_splits(prepared)

        symbol_predictions: list[pd.DataFrame] = []
        for split in splits:
            split_predictions = backtester._run_cnn_split(split)
            if split_predictions.empty:
                continue
            split_predictions = split_predictions.copy()
            split_predictions["symbol"] = symbol
            split_predictions["split_id"] = split.split_id
            symbol_predictions.append(split_predictions)

        if not symbol_predictions:
            continue

        combined = pd.concat(symbol_predictions, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        all_predictions.append(combined)
        metrics = backtester._compute_metrics(combined)
        per_symbol.append(
            {
                "symbol": symbol,
                "test_window": [
                    combined["timestamp"].min().isoformat(),
                    combined["timestamp"].max().isoformat(),
                ],
                "num_predictions": int(len(combined)),
                "metrics": metrics,
            }
        )

    if not all_predictions:
        raise ValueError("Walk-forward test predictions were not generated for any symbol.")

    test_predictions = pd.concat(all_predictions, ignore_index=True)
    test_predictions = test_predictions.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    y_true = direction_series_to_names(test_predictions["actual_direction"])
    y_pred = direction_series_to_names(test_predictions["predicted_direction"])
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=DIRECTION_NAMES,
        average="weighted",
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=DIRECTION_NAMES)

    aggregate_backtester = TechnicalBacktester(config=build_backtest_config(next(iter(datasets.values())).clean_frame))
    metrics = aggregate_backtester._compute_metrics(test_predictions)

    trade_mask = test_predictions["predicted_direction"].astype(int) != 0
    trade_returns = test_predictions.loc[trade_mask, "strategy_return"].astype(float)
    total_trades = int(trade_mask.sum())
    average_trade_return = float(trade_returns.mean()) if not trade_returns.empty else 0.0

    equity_curve = (
        test_predictions.groupby("timestamp", as_index=False)["strategy_return"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    equity_curve["equity"] = (1.0 + equity_curve["strategy_return"].astype(float)).cumprod()

    return {
        "test_accuracy": round(accuracy, 6),
        "precision_weighted": round(float(precision), 6),
        "recall_weighted": round(float(recall), 6),
        "f1_weighted": round(float(f1), 6),
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": DIRECTION_NAMES,
        "prediction_distribution": {
            label: int(sum(pred == label for pred in y_pred))
            for label in DIRECTION_NAMES
        },
        "class_balance": {
            label: int(sum(true == label for true in y_true))
            for label in DIRECTION_NAMES
        },
        "trading_metrics": {
            "sharpe_ratio": round(float(metrics["sharpe_ratio"]), 6) if not math.isnan(metrics["sharpe_ratio"]) else None,
            "max_drawdown": round(float(metrics["max_drawdown"]), 6) if not math.isnan(metrics["max_drawdown"]) else None,
            "win_rate": round(float(metrics["win_rate"]), 6) if not math.isnan(metrics["win_rate"]) else None,
            "profit_factor": round(float(metrics["profit_factor"]), 6) if not math.isnan(metrics["profit_factor"]) else None,
            "average_trade_return": round(float(average_trade_return), 6),
            "total_trades_simulated": total_trades,
        },
        "test_window": [
            equity_curve["timestamp"].min().isoformat(),
            equity_curve["timestamp"].max().isoformat(),
        ],
        "per_symbol": per_symbol,
        "equity_curve": equity_curve,
        "predictions": test_predictions,
    }


def build_symbol_cnn_bundle(dataset: SymbolDataset, train_args: argparse.Namespace) -> tuple[trainer.SequenceDatasetBundle, list[str]]:
    cnn_feature_frame = dataset.feature_table[dataset.cnn_columns].copy()
    cnn_scaled, _, _ = trainer.impute_and_scale_features(cnn_feature_frame, dataset.split_idx)
    bundle = trainer.build_cnn_sequences(
        feature_values=cnn_scaled,
        close_values=dataset.engineered_frame["close"].values.astype(float),
        timestamps=dataset.engineered_frame["timestamp"],
        split_idx=dataset.split_idx,
        window_size=train_args.cnn_window,
        neutral_threshold=train_args.neutral_threshold,
    )
    return bundle, dataset.cnn_columns


def compute_permutation_importance(
    run_root: Path,
    datasets: dict[str, SymbolDataset],
    symbols: list[str],
    train_args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(train_args.seed)
    weighted_importance: dict[str, float] = {}
    weight_totals: dict[str, int] = {}
    device = torch.device(train_args.device)

    for symbol in symbols:
        bundle, feature_columns = build_symbol_cnn_bundle(datasets[symbol], train_args)
        checkpoint_path = run_root / trainer.sanitize_symbol(symbol) / "cnn_pattern" / "best_cnn_checkpoint.pt"
        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = trainer.DirectionCNNClassifier(
            time_steps=bundle.X_train.shape[-2],
            num_features=bundle.X_train.shape[-1],
            dropout=float(checkpoint.get("dropout", train_args.dropout)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        base_loader = trainer.make_loader(bundle.X_val, bundle.y_val, max(train_args.batch_size, 128))
        base_pred, _ = trainer.tensor_predict(model, base_loader, device)
        base_acc = float(accuracy_score(bundle.y_val, base_pred))
        val_weight = int(len(bundle.y_val))

        for idx, column in enumerate(feature_columns):
            permuted = bundle.X_val.copy()
            permuted[:, :, :, idx] = permuted[rng.permutation(len(permuted)), :, :, idx]
            permuted_loader = trainer.make_loader(permuted, bundle.y_val, max(train_args.batch_size, 128))
            perm_pred, _ = trainer.tensor_predict(model, permuted_loader, device)
            perm_acc = float(accuracy_score(bundle.y_val, perm_pred))
            drop = base_acc - perm_acc
            weighted_importance[column] = weighted_importance.get(column, 0.0) + drop * val_weight
            weight_totals[column] = weight_totals.get(column, 0) + val_weight

    ranking = [
        {
            "feature": column,
            "accuracy_drop": round(weighted_importance[column] / weight_totals[column], 6),
        }
        for column in weight_totals
        if weight_totals[column] > 0
    ]
    return sorted(ranking, key=lambda item: item["accuracy_drop"], reverse=True)


def find_previous_metrics(symbols: list[str], interval: str) -> dict[str, Any] | None:
    metrics_path = PROJECT_ROOT / "data" / "reports" / "training_runs" / "day7_three_symbol_metrics.json"
    if not metrics_path.exists():
        return None
    payload = _safe_read_json(metrics_path)
    matched = [row for row in payload if row.get("symbol") in symbols and row.get("interval") == interval]
    if not matched:
        return None
    avg_val_acc = float(np.mean([row["cnn_best_val_acc"] for row in matched]))
    return {
        "source": str(metrics_path),
        "matched_symbols": [row["symbol"] for row in matched],
        "average_validation_accuracy": avg_val_acc,
        "symbol_count": len(matched),
    }


def plot_bar_chart(items: list[tuple[str, float]], title: str, ylabel: str, path: Path, top_n: int | None = None) -> None:
    if plt is None or not items:
        return
    selected = items[:top_n] if top_n is not None else items
    labels = [label for label, _ in selected]
    values = [value for _, value in selected]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#1f77b4")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(confusion: list[list[int]], labels: list[str], path: Path, title: str) -> None:
    if plt is None:
        return
    matrix = np.asarray(confusion)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_equity_curve(equity_curve: pd.DataFrame, path: Path) -> None:
    if plt is None or equity_curve.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve["timestamp"], equity_curve["equity"], color="#2ca02c")
    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    readiness: dict[str, Any],
    training_config: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    asset_paths: dict[str, Path],
) -> None:
    top_features = feature_importance[:15]
    feature_rows = [[row["feature"], row["accuracy_drop"]] for row in top_features]
    readiness_rows = [
        [
            row["symbol"],
            row["source"],
            row["raw_rows"],
            row["clean_rows"],
            row["feature_rows"],
            row["rows_dropped_initial_clean"],
            row["rows_with_missing_features_pre_impute"],
            row["cnn_train_windows"],
            row["cnn_val_windows"],
        ]
        for row in readiness["rows_per_symbol"]
    ]
    coverage_rows = [[feature, pct] for feature, pct in sorted(readiness["feature_coverage_pct"].items())]

    lines = [
        f"# Model Training Report {_report_timestamp()}",
        "",
        "## Dataset Summary",
        "",
        f"- Total NSE equity symbols available: `{readiness['total_symbols_available']}`",
        f"- Train-ready symbols: `{readiness['symbols_train_ready']}`",
        f"- Symbols excluded by quality gates: `{', '.join(readiness['symbols_excluded_by_quality']) if readiness['symbols_excluded_by_quality'] else 'none'}`",
        f"- Total clean rows: `{readiness['total_rows']}`",
        f"- Total feature rows: `{readiness['total_feature_rows']}`",
        f"- Rows dropped during initial OHLC cleanup: `{readiness['total_rows_dropped_initial_clean']}`",
        f"- Rows with missing selected features before imputation: `{readiness['total_rows_with_missing_features_pre_impute']}`",
        f"- Final training window: `{readiness['final_training_window'][0]}` -> `{readiness['final_training_window'][1]}`",
        f"- Validation window: `{readiness['validation_window'][0]}` -> `{readiness['validation_window'][1]}`",
        f"- Test window: `{test_metrics['test_window'][0]}` -> `{test_metrics['test_window'][1]}`",
        "",
        format_table(
            [
                "Symbol",
                "Source",
                "Raw Rows",
                "Clean Rows",
                "Feature Rows",
                "Dropped Clean",
                "Missing Features Pre-Impute",
                "CNN Train Windows",
                "CNN Val Windows",
            ],
            readiness_rows,
        ),
        "",
        "## Feature Coverage",
        "",
        format_table(["Feature", "Coverage %"], coverage_rows),
        "",
        "## Training Configuration",
        "",
        "```json",
        json.dumps(training_config, indent=2),
        "```",
        "",
        "## Validation Metrics",
        "",
        f"- Training accuracy: `{validation_metrics['training_accuracy']:.4f}`",
        f"- Validation accuracy: `{validation_metrics['validation_accuracy']:.4f}`",
        f"- Precision (weighted): `{validation_metrics['precision_weighted']:.4f}`",
        f"- Recall (weighted): `{validation_metrics['recall_weighted']:.4f}`",
        f"- F1 (weighted): `{validation_metrics['f1_weighted']:.4f}`",
        f"- Overfitting gap (train - validation): `{validation_metrics['overfitting_gap']:.4f}`",
        "",
        format_table(
            ["", *validation_metrics["confusion_labels"]],
            [[label, *row] for label, row in zip(validation_metrics["confusion_labels"], validation_metrics["confusion_matrix"])],
        ),
        "",
        "## Test Metrics",
        "",
        f"- Test accuracy: `{test_metrics['test_accuracy']:.4f}`",
        f"- Precision (weighted): `{test_metrics['precision_weighted']:.4f}`",
        f"- Recall (weighted): `{test_metrics['recall_weighted']:.4f}`",
        f"- F1 (weighted): `{test_metrics['f1_weighted']:.4f}`",
        "",
        format_table(
            ["", *test_metrics["confusion_labels"]],
            [[label, *row] for label, row in zip(test_metrics["confusion_labels"], test_metrics["confusion_matrix"])],
        ),
        "",
        "## Trading Performance",
        "",
        f"- Sharpe ratio: `{test_metrics['trading_metrics']['sharpe_ratio']}`",
        f"- Max drawdown: `{test_metrics['trading_metrics']['max_drawdown']}`",
        f"- Win rate: `{test_metrics['trading_metrics']['win_rate']}`",
        f"- Profit factor: `{test_metrics['trading_metrics']['profit_factor']}`",
        f"- Average trade return: `{test_metrics['trading_metrics']['average_trade_return']}`",
        f"- Total trades simulated: `{test_metrics['trading_metrics']['total_trades_simulated']}`",
        "",
        "## Diagnostics",
        "",
        f"- Validation prediction distribution: `{validation_metrics['prediction_distribution']}`",
        f"- Validation class balance: `{validation_metrics['class_balance']}`",
        f"- Test prediction distribution: `{test_metrics['prediction_distribution']}`",
        f"- Test class balance: `{test_metrics['class_balance']}`",
        "",
        "## Feature Importance",
        "",
        format_table(["Feature", "Validation Accuracy Drop"], feature_rows),
        "",
    ]

    if comparison is None:
        lines.extend(
            [
                "## Previous Run Comparison",
                "",
                "- No direct prior run with overlapping symbol coverage was found in stored metrics.",
                "",
            ]
        )
    else:
        validation_delta = validation_metrics["validation_accuracy"] - comparison["average_validation_accuracy"]
        lines.extend(
            [
                "## Previous Run Comparison",
                "",
                f"- Comparison source: `{comparison['source']}`",
                f"- Matched symbols: `{', '.join(comparison['matched_symbols'])}`",
                f"- Previous average validation accuracy: `{comparison['average_validation_accuracy']:.4f}`",
                f"- Current validation accuracy: `{validation_metrics['validation_accuracy']:.4f}`",
                f"- Validation accuracy delta: `{validation_delta:.4f}`",
                f"- Symbol coverage change: `{readiness['symbols_train_ready'] - comparison['symbol_count']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Charts",
            "",
            f"![Feature Importance]({asset_paths['feature_importance_chart'].name})",
            "",
            f"![Confusion Matrix]({asset_paths['confusion_matrix_chart'].name})",
            "",
            f"![Equity Curve]({asset_paths['equity_curve_chart'].name})",
            "",
            f"![Prediction Distribution]({asset_paths['prediction_distribution_chart'].name})",
            "",
            "## Conclusions",
            "",
            f"- The dataset now supports `{readiness['symbols_train_ready']}` train-ready NSE equity symbols at `{training_config['interval']}`.",
            f"- Validation accuracy is `{validation_metrics['validation_accuracy']:.4f}` and out-of-sample test accuracy is `{test_metrics['test_accuracy']:.4f}`.",
            f"- Trading Sharpe is `{test_metrics['trading_metrics']['sharpe_ratio']}` with max drawdown `{test_metrics['trading_metrics']['max_drawdown']}`.",
            f"- The largest validation feature-importance drivers are `{', '.join(row['feature'] for row in top_features[:5])}`.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_logging()
    args = parse_args()
    train_args = build_training_args(args)

    quality_df = discover_quality_rows(args)
    train_ready_symbols = quality_df.loc[quality_df["train_ready"] == True, "symbol"].astype(str).tolist()
    if not train_ready_symbols:
        raise ValueError("No train-ready symbols found after applying quality gates.")

    datasets: dict[str, SymbolDataset] = {}
    preparation_failures: list[dict[str, str]] = []
    for symbol in train_ready_symbols:
        try:
            datasets[symbol] = prepare_symbol_dataset(symbol, train_args)
        except Exception as exc:
            LOGGER.exception("Dataset preparation failed for %s: %s", symbol, exc)
            preparation_failures.append({"symbol": symbol, "error": str(exc)})

    symbols = sorted(datasets)
    if not symbols:
        raise ValueError("No symbols remained after dataset preparation.")

    run_timestamp = _report_timestamp()
    run_root = PROJECT_ROOT / train_args.output_root / f"regime_aware_{args.interval}_{run_timestamp}"
    summaries, training_failures = run_training(symbols, datasets, train_args, run_root)
    if training_failures:
        failed_symbols = {item["symbol"] for item in training_failures}
        symbols = [symbol for symbol in symbols if symbol not in failed_symbols]
    if not symbols:
        raise ValueError("Training failed for every prepared symbol.")

    readiness = aggregate_readiness([datasets[symbol] for symbol in symbols], quality_df)
    validation_metrics = aggregate_validation_metrics(run_root, symbols)
    test_metrics = evaluate_test_predictions(datasets, symbols)
    feature_importance = compute_permutation_importance(run_root, datasets, symbols, train_args)
    comparison = find_previous_metrics(symbols, args.interval)

    report_dir = PROJECT_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"model_training_report_{run_timestamp}.md"
    detail_path = report_dir / f"model_training_report_{run_timestamp}.json"

    feature_importance_chart = report_dir / f"model_training_feature_importance_{run_timestamp}.png"
    confusion_chart = report_dir / f"model_training_confusion_matrix_{run_timestamp}.png"
    equity_curve_chart = report_dir / f"model_training_equity_curve_{run_timestamp}.png"
    prediction_distribution_chart = report_dir / f"model_training_prediction_distribution_{run_timestamp}.png"

    plot_bar_chart(
        [(item["feature"], item["accuracy_drop"]) for item in feature_importance],
        "Permutation Feature Importance (Validation Accuracy Drop)",
        "Accuracy Drop",
        feature_importance_chart,
        top_n=15,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["confusion_labels"],
        confusion_chart,
        "Out-of-Sample Test Confusion Matrix",
    )
    plot_equity_curve(test_metrics["equity_curve"], equity_curve_chart)
    plot_bar_chart(
        list(test_metrics["prediction_distribution"].items()),
        "Out-of-Sample Prediction Distribution",
        "Count",
        prediction_distribution_chart,
    )

    training_config = {
        "symbols": symbols,
        "interval": args.interval,
        "limit": args.limit,
        "train_split": args.train_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "scheduler_patience": args.scheduler_patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "reg_window": args.reg_window,
        "cnn_window": args.cnn_window,
        "neutral_threshold": args.neutral_threshold,
        "arima_order": args.arima_order,
        "lstm_hidden_size": args.lstm_hidden_size,
        "lstm_layers": args.lstm_layers,
        "seed": args.seed,
        "device": args.device,
        "run_root": str(run_root),
    }

    asset_paths = {
        "feature_importance_chart": feature_importance_chart,
        "confusion_matrix_chart": confusion_chart,
        "equity_curve_chart": equity_curve_chart,
        "prediction_distribution_chart": prediction_distribution_chart,
    }
    write_markdown_report(
        path=report_path,
        readiness=readiness,
        training_config=training_config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importance=feature_importance,
        comparison=comparison,
        asset_paths=asset_paths,
    )

    detail_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness,
        "training_config": training_config,
        "training_summaries": summaries,
        "preparation_failures": preparation_failures,
        "training_failures": training_failures,
        "validation_metrics": validation_metrics,
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
            if key not in {"equity_curve", "predictions"}
        },
        "feature_importance": feature_importance,
        "comparison": comparison,
    }
    save_json(detail_path, detail_payload)

    print(f"Report written: {report_path}")
    print(f"Detail JSON written: {detail_path}")
    print(f"Training run root: {run_root}")
    print(
        "Summary: "
        f"symbols={len(symbols)}, "
        f"validation_accuracy={validation_metrics['validation_accuracy']:.4f}, "
        f"test_accuracy={test_metrics['test_accuracy']:.4f}, "
        f"sharpe={test_metrics['trading_metrics']['sharpe_ratio']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
