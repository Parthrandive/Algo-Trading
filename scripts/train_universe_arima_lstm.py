import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import apply_daily_trend_confirmation, engineer_features
from src.agents.technical.models.arima_lstm import ArimaLstmHybrid, LSTMResidualModel
from config.symbols import (
    EQUITY_SYMBOLS,
    FOREX_SYMBOLS,
    MIN_ROWS,
    dedupe_symbols,
    SplitCounts,
    SymbolValidationResult,
    discover_training_symbols
)
from config.symbols import is_forex, validate_equity_symbol
from src.agents.technical.label_utils import (
    build_labels as build_labels_unified,
    choose_neutral_threshold as choose_neutral_threshold_unified,
    atr_effective_threshold,
    recall_balance as compute_recall_balance,
    directional_coverage as compute_directional_coverage,
    class_balance_report,
)
from src.agents.technical.validation_metrics import (
    post_cost_sharpe as compute_post_cost_sharpe,
    expected_calibration_error,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2

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

def parse_requested_symbols(symbol: Optional[str], symbols: Optional[str]) -> list[str]:
    values: list[str] = []
    if symbol:
        values.append(symbol.strip())
    if symbols:
        values.extend([item.strip() for item in str(symbols).split(",") if item.strip()])
    return dedupe_symbols(values)


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
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


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    torch.save(payload, path)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None

RAW_MACRO_COLUMNS = {
    "CPI",
    "WPI",
    "IIP",
    "FII_FLOW",
    "DII_FLOW",
    "FX_RESERVES",
    "INDIA_US_10Y_SPREAD",
    "RBI_BULLETIN",
    "REPO_RATE",
    "US_10Y",
}


def set_seed(seed: int):
    """Set reproducibility seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_data(df: pd.DataFrame) -> None:
    """Validate data quality before training."""
    if len(df) < 40: # Lowered from 100 to support smaller datasets
        raise ValueError(f"Need at least 40 rows to train ARIMA-LSTM. Got {len(df)}.")

    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check NaNs
    for col in required_cols:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.05:
            raise ValueError(f"Column '{col}' has {nan_pct:.1%} NaNs (max 5% allowed).")

    # Check constant close
    if df['close'].nunique() == 1:
        logger.warning("Target column 'close' is constant. Model cannot learn.")


def custom_train_lstm(
    hybrid: ArimaLstmHybrid,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    patience: int,
    output_dir: str,
    symbol: str,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Custom training loop w/ early stopping, scheduler, and best checkpoint."""
    hybrid.lstm_model = LSTMResidualModel(
        input_size=len(hybrid.feature_columns),
        hidden_size=hybrid.lstm_hidden_size,
        num_layers=hybrid.lstm_layers
    ).to(hybrid.device)
    if init_state_dict:
        hybrid.lstm_model.load_state_dict(init_state_dict)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hybrid.lstm_model.parameters(), lr=hybrid.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights_path = checkpoint_path or os.path.join(output_dir, f"best_model_{sanitize_symbol(symbol)}.pt")
    best_train_loss = float('inf')
    
    logger.info("Training LSTM on residuals...")
    
    for epoch in range(epochs):
        # Train
        hybrid.lstm_model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = hybrid.lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= max(len(train_dataset), 1)

        # Validate
        hybrid.lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = hybrid.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= max(len(val_dataset), 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            epochs_no_improve = 0
            safe_torch_save(hybrid.lstm_model.state_dict(), Path(best_weights_path))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (val loss did not improve for {patience} epochs).")
                break
                
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

    # Load best weights
    if os.path.exists(best_weights_path):
        hybrid.lstm_model.load_state_dict(torch.load(best_weights_path, map_location=hybrid.device))
    hybrid.lstm_model.eval()
    hybrid.is_trained = True
    
    return {
        "train_loss": train_loss,
        "best_train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "epochs_run": epoch + 1,
    }


def build_no_leakage_residuals(
    train_close: pd.Series,
    val_close: pd.Series,
    test_close: pd.Series,
    arima_order: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    import statsmodels.api as sm

    arima_model = sm.tsa.ARIMA(
        train_close.values,
        order=arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    arima_results = arima_model.fit()

    train_preds = np.asarray(arima_results.predict(start=0, end=len(train_close) - 1, typ='levels'), dtype=float)
    train_residuals = train_close.values - train_preds

    def rolling_predict(close_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        nonlocal arima_results
        residuals: list[float] = []
        forecasts: list[float] = []
        history_values = train_close.values.copy()
        for actual in close_series.values:
            try:
                forecast = float(arima_results.forecast(steps=1)[0])
            except Exception:
                forecast = float(history_values[-1]) if len(history_values) else float(actual)

            forecasts.append(forecast)
            residuals.append(float(actual - forecast))

            try:
                arima_results = arima_results.append([actual], refit=False)
            except Exception:
                history_values = np.append(history_values, actual)
                arima_results = sm.tsa.ARIMA(
                    history_values,
                    order=arima_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            else:
                history_values = np.append(history_values, actual)
        return np.asarray(residuals, dtype=float), np.asarray(forecasts, dtype=float)

    val_residuals, val_preds = rolling_predict(val_close)
    test_residuals, test_preds = rolling_predict(test_close)
    return train_residuals, val_residuals, test_residuals, train_preds, val_preds, test_preds, arima_results


def evaluate_mse(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    if len(X) == 0:
        return float('nan')
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(X, y)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            total_loss += float(loss.item()) * batch_X.size(0)
            total_count += int(batch_X.size(0))
    return total_loss / max(total_count, 1)


def scores_to_soft_probs(scores: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Heuristic soft prob mapper aligned to the active symbol threshold."""
    probs = np.zeros((len(scores), 3), dtype=np.float64)
    band = max(float(threshold), 1e-6)
    for i, s in enumerate(scores):
        if s > band:
            probs[i] = [0.05, 0.10, 0.85]
        elif s < -band:
            probs[i] = [0.85, 0.10, 0.05]
        else:
            probs[i] = [0.10, 0.80, 0.10]
    return probs


def predict_lstm_scores(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0,))
    model.eval()
    dataset = TensorDataset(X)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch_X,) in loader:
            p = model(batch_X).squeeze(-1).cpu().numpy()
            preds.extend(p)
    return np.array(preds)


def returns_to_labels(returns: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.full(len(returns), CLASS_NEUTRAL, dtype=np.int64)
    labels[np.asarray(returns) < -float(threshold)] = CLASS_DOWN
    labels[np.asarray(returns) > float(threshold)] = CLASS_UP
    return labels


def label_distribution(labels: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(labels, dtype=np.int64)
    if len(arr) == 0:
        return {
            "total": 0,
            "counts": {"down": 0, "neutral": 0, "up": 0},
            "ratios": {"down": 0.0, "neutral": 0.0, "up": 0.0},
        }
    counts = np.bincount(arr, minlength=3)
    total = int(counts.sum())
    return {
        "total": total,
        "counts": {
            "down": int(counts[CLASS_DOWN]),
            "neutral": int(counts[CLASS_NEUTRAL]),
            "up": int(counts[CLASS_UP]),
        },
        "ratios": {
            "down": float(counts[CLASS_DOWN] / total),
            "neutral": float(counts[CLASS_NEUTRAL] / total),
            "up": float(counts[CLASS_UP] / total),
        },
    }


def directional_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "test_accuracy": 0.0,
            "directional_accuracy": 0.0,
            "recall_balance": 0.0,
            "directional_coverage": 0.0,
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
        "recall_balance": compute_recall_balance(y_true, y_pred),
        "directional_coverage": compute_directional_coverage(y_pred),
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


def quality_gate_split_counts(df: pd.DataFrame) -> SplitCounts:
    n_rows = len(df)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return SplitCounts(
        train_rows=train_end,
        val_rows=val_end - train_end,
        test_rows=n_rows - val_end,
    )


def extract_symbol_features(
    symbol: str,
    df: pd.DataFrame,
    hybrid: ArimaLstmHybrid,
    *,
    include_daily_features: bool,
) -> tuple:
    """Extracts features, splits chronologically, computes ARIMA residuals, entirely per-symbol."""
    validate_data(df)
    
    is_forex_target = is_forex(symbol)
    df_features = engineer_features(
        df,
        is_forex=is_forex_target,
        include_daily_features=include_daily_features,
    )
    df_features = df_features.sort_values('timestamp').reset_index(drop=True)

    n = len(df_features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_raw = df_features.iloc[:train_end].copy()
    val_raw = df_features.iloc[train_end:val_end].copy()
    test_raw = df_features.iloc[val_end:].copy()

    target_col = 'log_return'
    exclude_cols = {'symbol', 'timestamp', 'close', target_col}
    feature_columns = [
        c for c in train_raw.columns
        if c not in exclude_cols 
        and pd.api.types.is_numeric_dtype(train_raw[c])
        and not train_raw[c].isna().all()
        and c not in RAW_MACRO_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"{symbol}: no usable numeric feature columns after preprocessing.")

    # Drop highly sparse features first to avoid wiping out the full train split.
    nan_ratio = train_raw[feature_columns].isna().mean()
    sparse_cols = [c for c in feature_columns if float(nan_ratio[c]) > 0.40]
    if sparse_cols:
        logger.info(
            "[%s] Dropping %s sparse feature(s) (>40%% NaN in train split).",
            symbol,
            len(sparse_cols),
        )
    feature_columns = [c for c in feature_columns if c not in sparse_cols]
    if not feature_columns:
        raise ValueError(f"{symbol}: all features dropped as sparse after NaN filtering.")

    for part in (train_raw, val_raw, test_raw):
        part.loc[:, feature_columns] = part[feature_columns].replace([np.inf, -np.inf], np.nan)

    train_df = train_raw.copy()
    train_df[feature_columns] = train_df[feature_columns].fillna(train_df[feature_columns].median(numeric_only=True))
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    medians = train_df[feature_columns].median(numeric_only=True).fillna(0.0)
    train_df[feature_columns] = train_df[feature_columns].fillna(medians)
    
    val_df = val_raw.copy()
    val_df[feature_columns] = val_df[feature_columns].fillna(medians)
    val_df = val_df.dropna(subset=[target_col]).reset_index(drop=True)
    
    test_df = test_raw.copy()
    test_df[feature_columns] = test_df[feature_columns].fillna(medians)
    test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) < hybrid.window_size + 5:
        raise ValueError(
            f"Insufficient rows after feature cleanup. "
            f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, "
            f"Window={hybrid.window_size}"
        )

    # NO LEAKAGE ARIMA
    train_residuals, val_residuals, test_residuals, train_preds, val_preds, test_preds, arima_results = build_no_leakage_residuals(
        train_df['close'], val_df['close'], test_df['close'], hybrid.arima_order
    )
    
    train_df['residual'] = train_residuals
    val_df['residual'] = val_residuals
    test_df['residual'] = test_residuals
    
    train_df['arima_forecast'] = train_preds
    val_df['arima_forecast'] = val_preds
    test_df['arima_forecast'] = test_preds

    return train_df, val_df, test_df, feature_columns, medians, arima_results


def main():
    parser = argparse.ArgumentParser(description="Train GLOBAL Universe ARIMA-LSTM hybrid model.")
    parser.add_argument("--symbol", default=None, help="Ignored in Universe mode")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=25, help="Max epochs for per-symbol fine-tuning")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (larger for global)")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order (p,d,q)")
    parser.add_argument("--window-size", type=int, default=10, help="LSTM window size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--finetune-patience", type=int, default=8, help="Per-symbol fine-tuning early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/models/arima_lstm/", help="Output directory")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1h", help="Candle interval, e.g. 1d, 1h. Default: 1h")
    parser.add_argument(
        "--disable-daily-features",
        action="store_true",
        help="Disable daily timeframe feature fusion onto hourly rows.",
    )
    parser.add_argument(
        "--disable-daily-trend-filter",
        action="store_true",
        help="Disable UP->NEUTRAL gating when daily trend is not bullish.",
    )
    parser.add_argument(
        "--daily-filter-mode",
        choices=["hard", "soft"],
        default="hard",
        help="Daily trend confirmation mode. hard=force UP->NEUTRAL, soft=penalize UP confidence first.",
    )
    parser.add_argument(
        "--daily-up-penalty",
        type=float,
        default=0.50,
        help="Penalty factor (0-1) applied to UP probability in daily soft filter mode.",
    )
    parser.add_argument("--class-threshold", type=float, default=0.005, help="Return threshold used to derive up/down/neutral labels for evaluation.")
    parser.add_argument(
        "--class-threshold-min",
        type=float,
        default=0.001,
        help="Minimum adaptive class threshold floor (supports tighter directional bands than 0.005).",
    )
    parser.add_argument("--min-neutral-ratio", type=float, default=0.15, help="Minimum target neutral prediction ratio for per-symbol thresholding.")
    parser.add_argument("--max-neutral-ratio", type=float, default=0.20, help="Maximum target neutral prediction ratio for per-symbol thresholding.")
    parser.add_argument("--target-neutral-ratio", type=float, default=0.175, help="Target neutral prediction ratio for per-symbol thresholding.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional orchestration run identifier for artifact alignment.",
    )
    parser.add_argument(
        "--feature-schema-version",
        type=str,
        default="technical_features_v1",
        help="Feature schema version tag persisted in training artifacts.",
    )
    parser.add_argument(
        "--recalibrate-only",
        action="store_true",
        help="Skip training/fine-tuning and only regenerate per-symbol ARIMA-LSTM probabilities using stored model weights with adaptive thresholds.",
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
    include_daily_features = not bool(args.disable_daily_features)
    apply_daily_trend_filter = not bool(args.disable_daily_trend_filter)
    run_id = str(args.run_id).strip() if args.run_id else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_script = str(Path(__file__).name)
    feature_schema_version = str(args.feature_schema_version).strip()
    class_threshold_min = max(float(args.class_threshold_min), 1e-6)
    logger.info(
        "Daily feature fusion: %s | Daily trend UP filter: %s (mode=%s, up_penalty=%.2f) | run_id=%s",
        "enabled" if include_daily_features else "disabled",
        "enabled" if apply_daily_trend_filter else "disabled",
        args.daily_filter_mode,
        float(np.clip(args.daily_up_penalty, 0.0, 1.0)),
        run_id,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    arima_order = tuple(map(int, args.arima_order.split(',')))

    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    requested_symbols = parse_requested_symbols(args.symbol, args.symbols)

    def validate_symbol(symbol: str):
        try:
            frame = loader.load_historical_bars(
                symbol, limit=args.limit, use_nse_fallback=args.use_nse, min_fallback_rows=40, interval=args.interval,
            )
        except Exception as exc:
            return SymbolValidationResult(symbol=symbol, is_active=False, reason=f"load_failed: {exc}")
        return validate_equity_symbol(symbol=symbol, frame=frame, interval=args.interval, split_counts=quality_gate_split_counts(frame))

    discovery = discover_training_symbols(
        interval=args.interval,
        requested_symbols=requested_symbols or None,
        database_url=db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    training_symbols = list(discovery.active_symbols)
    if not training_symbols:
        logger.error("No active equity symbols passed the training quality gate.")
        sys.exit(1)

    logger.info("Initializing Global Hybrid parameters...")
    hybrid = ArimaLstmHybrid(arima_order=arima_order, learning_rate=args.lr, window_size=args.window_size)

    # --- Phase 1: Extract features per symbol ---
    symbol_data = {}
    all_train_dfs = []
    global_feature_columns = None
    
    for symbol in training_symbols:
        if is_forex(symbol):
            continue
        try:
            logger.info("Extracting features and ARIMA residuals for %s...", symbol)
            frame = loader.load_historical_bars(symbol, limit=args.limit, use_nse_fallback=args.use_nse, min_fallback_rows=40, interval=args.interval)
            train_df, val_df, test_df, f_cols, medians, arima_results = extract_symbol_features(
                symbol,
                frame,
                hybrid,
                include_daily_features=include_daily_features,
            )
            symbol_data[symbol] = {
                "train": train_df, 
                "val": val_df, 
                "test": test_df,
                "medians": medians,
                "arima_results": arima_results
            }
            all_train_dfs.append(train_df[f_cols].copy())
            if not global_feature_columns:
                global_feature_columns = f_cols
        except Exception as exc:
            logger.error("Skipping %s due to extraction error: %s", symbol, exc)
            
    if not symbol_data:
        logger.error("No valid symbol data extracted.")
        sys.exit(1)
        
    hybrid.feature_columns = global_feature_columns

    global_dir = output_root / "GLOBAL"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_scaler_path = global_dir / "feature_scaler.pkl"

    import pickle

    # --- Phase 2: Build / Load Global Scaler ---
    if args.recalibrate_only:
        if not global_scaler_path.exists():
            logger.error("Recalibrate-only mode requested but scaler is missing: %s", global_scaler_path)
            sys.exit(1)
        with open(global_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded existing GLOBAL scaler from %s (recalibration mode).", global_scaler_path)
    else:
        logger.info("Fitting GLOBAL scaler...")
        global_train_df = pd.concat(all_train_dfs, ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(global_train_df[global_feature_columns].values)

    # --- Phase 3: Build Windows ---
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for symbol, splits in symbol_data.items():
        for s_name in ["train", "val", "test"]:
            df = splits[s_name].copy()
            df = df.astype({col: np.float64 for col in global_feature_columns}, copy=False)
            df.loc[:, global_feature_columns] = scaler.transform(df[global_feature_columns].to_numpy(dtype=np.float64))
            X, y = hybrid._prepare_lstm_data(df, df['residual'])
            timestamps = df["timestamp"].iloc[hybrid.window_size:].values if len(df) > hybrid.window_size else np.empty(0)
            arima_forecasts = df["arima_forecast"].iloc[hybrid.window_size:].values if len(df) > hybrid.window_size else np.empty(0)
            actual_returns = df["log_return"].iloc[hybrid.window_size:].values if len(df) > hybrid.window_size else np.empty(0)
            if len(df) > hybrid.window_size and "daily_trend_bullish" in df.columns:
                daily_trend = df["daily_trend_bullish"].iloc[hybrid.window_size:].values
            else:
                daily_trend = np.full(len(timestamps), np.nan, dtype=np.float64)
            
            splits[f"X_{s_name}"] = X
            splits[f"y_{s_name}"] = y
            splits[f"timestamps_{s_name}"] = timestamps
            splits[f"arima_forecast_{s_name}"] = arima_forecasts
            splits[f"actual_return_{s_name}"] = actual_returns
            splits[f"daily_trend_{s_name}"] = daily_trend
            
            if s_name == "train":
                X_train_list.append(X)
                y_train_list.append(y)
            elif s_name == "val":
                X_val_list.append(X)
                y_val_list.append(y)
            else:
                X_test_list.append(X)
                y_test_list.append(y)

    X_train_global = torch.cat(X_train_list, dim=0).to(hybrid.device) if X_train_list else torch.empty((0,)).to(hybrid.device)
    y_train_global = torch.cat(y_train_list, dim=0).to(hybrid.device) if y_train_list else torch.empty((0,)).to(hybrid.device)
    X_val_global = torch.cat(X_val_list, dim=0).to(hybrid.device) if X_val_list else torch.empty((0,)).to(hybrid.device)
    y_val_global = torch.cat(y_val_list, dim=0).to(hybrid.device) if y_val_list else torch.empty((0,)).to(hybrid.device)

    logger.info("GLOBAL Train shape: %s | Val shape: %s", X_train_global.shape, X_val_global.shape)

    # --- Phase 4: Train (or Load) Global Model ---
    global_checkpoint_path = str(global_dir / "best_model_GLOBAL.pt")
    metrics: Dict[str, Any] = {}

    if args.recalibrate_only:
        existing_global_model = first_existing([global_dir / "best_model.pt", global_dir / "best_model_GLOBAL.pt"])
        if existing_global_model is None:
            logger.error("Recalibrate-only mode requested but GLOBAL model weights were not found in %s", global_dir)
            sys.exit(1)
        hybrid.lstm_model = LSTMResidualModel(
            input_size=len(hybrid.feature_columns),
            hidden_size=hybrid.lstm_hidden_size,
            num_layers=hybrid.lstm_layers,
        ).to(hybrid.device)
        hybrid.lstm_model.load_state_dict(torch.load(existing_global_model, map_location=hybrid.device))
        hybrid.lstm_model.eval()
        hybrid.is_trained = True
        logger.info("Loaded existing GLOBAL model weights from %s (recalibration mode).", existing_global_model)
    else:
        logger.info("========== Training GLOBAL Base Model ==========")
        metrics = custom_train_lstm(
            hybrid, X_train_global, y_train_global, X_val_global, y_val_global, 
            epochs=args.epochs, batch_size=args.batch_size, 
            patience=args.patience, output_dir=str(global_dir), symbol="GLOBAL",
            checkpoint_path=global_checkpoint_path,
        )
        logger.info("Total epochs trained (early stopping aware): %s", metrics['epochs_run'])

        safe_torch_save(hybrid.lstm_model.state_dict(), global_dir / "best_model.pt")
        with open(global_scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "GLOBAL",
            "symbol_canonical": "GLOBAL",
            "run_id": run_id,
            "interval": args.interval,
            "source_script": source_script,
            "feature_schema_version": feature_schema_version,
            "trained_on": list(symbol_data.keys()),
            "hyperparameters": {
                "arima_order": hybrid.arima_order,
                "window_size": hybrid.window_size,
                "learning_rate": hybrid.learning_rate,
                "epochs_run": metrics["epochs_run"],
                "batch_size": args.batch_size,
                "patience": args.patience,
                "seed": args.seed,
                "class_threshold": args.class_threshold,
                "class_threshold_min": class_threshold_min,
                "min_neutral_ratio": args.min_neutral_ratio,
                "max_neutral_ratio": args.max_neutral_ratio,
                "target_neutral_ratio": args.target_neutral_ratio,
                "include_daily_features": include_daily_features,
                "apply_daily_trend_filter": apply_daily_trend_filter,
                "daily_filter_mode": args.daily_filter_mode,
                "daily_up_penalty": float(np.clip(args.daily_up_penalty, 0.0, 1.0)),
                "run_id": run_id,
                "interval": args.interval,
                "feature_schema_version": feature_schema_version,
            },
            "metrics": {
                "final_train_loss": metrics["train_loss"],
                "best_train_loss": metrics["best_train_loss"],
                "best_val_loss": metrics["val_loss"]
            }
        }
        safe_write_json(global_dir / "training_meta.json", meta)
        safe_write_json(global_dir / "hyperparams.json", meta["hyperparameters"])
    global_state_dict = clone_state_dict(hybrid.lstm_model)

    # --- Phase 5: Evaluate Per Symbol ---
    logger.info("========== Evaluating Per Symbol ==========")
    for symbol, splits in symbol_data.items():
        sym_dir = output_root / sanitize_symbol(symbol)
        sym_dir.mkdir(parents=True, exist_ok=True)

        symbol_hybrid = ArimaLstmHybrid(arima_order=arima_order, learning_rate=args.lr, window_size=args.window_size)
        symbol_hybrid.feature_columns = list(global_feature_columns)
        symbol_hybrid.lstm_model = LSTMResidualModel(
            input_size=len(symbol_hybrid.feature_columns),
            hidden_size=symbol_hybrid.lstm_hidden_size,
            num_layers=symbol_hybrid.lstm_layers,
        ).to(symbol_hybrid.device)
        symbol_hybrid.lstm_model.load_state_dict(global_state_dict)
        symbol_hybrid.lstm_model.eval()
        symbol_hybrid.is_trained = True

        X_train_sym = splits.get("X_train")
        y_train_sym = splits.get("y_train")
        X_val_sym = splits.get("X_val")
        y_val_sym = splits.get("y_val")
        X_test_sym = splits.get("X_test")

        finetune_applied = False
        finetune_stats: Dict[str, float] = {}

        split_payloads = {}
        for s_name in ["train", "val", "test"]:
            X_data = splits.get(f"X_{s_name}")
            y_data = splits.get(f"y_{s_name}")
            timestamps = splits.get(f"timestamps_{s_name}")
            arima_forecasts = splits.get(f"arima_forecast_{s_name}")
            actual_returns = splits.get(f"actual_return_{s_name}")
            daily_trend = splits.get(f"daily_trend_{s_name}")
            
            if X_data is None or len(X_data) == 0:
                continue
                
            if s_name == "test":
                test_mse = evaluate_mse(
                    symbol_hybrid.lstm_model,
                    X_data.to(symbol_hybrid.device),
                    y_data.to(symbol_hybrid.device),
                    args.batch_size,
                )
                logger.info("[%s] Test MSE: %.6f", symbol, test_mse)

            lstm_res = predict_lstm_scores(symbol_hybrid.lstm_model, X_data.to(symbol_hybrid.device), batch_size=args.batch_size)
            final_scores = arima_forecasts + lstm_res
            split_payloads[s_name] = {
                "timestamps": timestamps,
                "actual_returns": actual_returns if isinstance(actual_returns, np.ndarray) else np.empty((0,), dtype=np.float64),
                "final_scores": final_scores,
                "daily_trend_bullish": daily_trend if isinstance(daily_trend, np.ndarray) else np.full(len(final_scores), np.nan),
            }

        train_scores = split_payloads.get("train", {}).get("final_scores", np.empty((0,), dtype=np.float64))
        train_actual_returns = split_payloads.get("train", {}).get("actual_returns", np.empty((0,), dtype=np.float64))
        
        # Adaptive Threshold Selection
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
            # Dummy ratio for logging
            y_check = returns_to_labels(train_actual_returns, symbol_threshold)
            train_neutral_ratio = float((y_check == CLASS_NEUTRAL).mean())
            logger.info("[%s] ATR-scaled threshold: %.6f (neutral share %.2f%%)", symbol, symbol_threshold, train_neutral_ratio*100)
        else:
            symbol_threshold, train_neutral_ratio = choose_neutral_threshold_unified(
                train_forward_returns=train_actual_returns,
                requested_threshold=float(args.class_threshold),
                min_neutral_ratio=float(args.min_neutral_ratio),
                max_neutral_ratio=float(args.max_neutral_ratio),
                target_neutral_ratio=float(args.target_neutral_ratio),
            )
            logger.info("[%s] Adaptive threshold: %.6f (neutral share %.2f%%)", symbol, symbol_threshold, train_neutral_ratio*100)

        split_prediction_distribution = {}
        class_metrics = {}
        
        for s_name in ["train", "val", "test"]:
            payload = split_payloads.get(s_name)
            if payload is None:
                continue
            final_scores = payload["final_scores"]
            timestamps = payload["timestamps"]
            actual_returns = payload["actual_returns"]
            daily_trend_bullish = payload["daily_trend_bullish"]

            probs = scores_to_soft_probs(final_scores, threshold=symbol_threshold)
            pred_labels = returns_to_labels(final_scores, threshold=symbol_threshold)
            daily_filter_mask = np.zeros(len(pred_labels), dtype=bool)
            if apply_daily_trend_filter:
                pred_labels, probs, daily_filter_mask = apply_daily_trend_confirmation(
                    pred_labels,
                    probs,
                    daily_trend_bullish=daily_trend_bullish,
                    mode=args.daily_filter_mode,
                    up_penalty=float(np.clip(args.daily_up_penalty, 0.0, 1.0)),
                )
            
            pred_dist = label_distribution(pred_labels)
            y_true_split = returns_to_labels(actual_returns, threshold=symbol_threshold) if len(actual_returns) > 0 else np.empty(0)
            split_cls_metrics = directional_classification_metrics(y_true=y_true_split, y_pred=pred_labels)
            
            pred_dist["directional_metrics"] = {
                "directional_accuracy": float(split_cls_metrics["directional_accuracy"]),
                "recall_balance": split_cls_metrics["recall_balance"],
                "directional_coverage": split_cls_metrics["directional_coverage"],
                "up_recall": float(split_cls_metrics["up_recall"]),
                "down_recall": float(split_cls_metrics["down_recall"]),
                "up_precision": float(split_cls_metrics["up_precision"]),
                "down_precision": float(split_cls_metrics["down_precision"]),
                "post_cost_sharpe": compute_post_cost_sharpe(y_true_split, pred_labels, actual_returns) if len(actual_returns) > 0 else 0.0,
            }
            split_prediction_distribution[s_name] = pred_dist
            if s_name == "test":
                class_metrics = split_cls_metrics
                
            prob_df = pd.DataFrame({
                "timestamp": timestamps,
                "split": s_name,
                "lstm_prob_down": probs[:, 0],
                "lstm_prob_neutral": probs[:, 1],
                "lstm_prob_up": probs[:, 2],
                "predicted_label": pred_labels,
                "lstm_final_score": final_scores,
                "actual_return": actual_returns if len(actual_returns) == len(pred_labels) else np.full(len(pred_labels), np.nan),
                "class_threshold": symbol_threshold,
            })
            sym_dir.mkdir(parents=True, exist_ok=True)
            prob_df.to_parquet(sym_dir / f"{s_name}_predictions.parquet", index=False)

        symbol_hyperparams = {
            "symbol": symbol,
            "symbol_canonical": symbol,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": source_script,
            "feature_schema_version": feature_schema_version,
            "trained_as_part_of_universe": True,
            "base_model_symbol": "GLOBAL",
            "arima_order": hybrid.arima_order,
            "window_size": hybrid.window_size,
            "learning_rate": hybrid.learning_rate,
            "feature_columns": list(global_feature_columns),
            "class_threshold": float(args.class_threshold),
            "class_threshold_min": class_threshold_min,
            "effective_class_threshold": symbol_threshold,
            "label_mode": getattr(args, "label_mode", "fixed"),
        }
        safe_write_json(sym_dir / "hyperparams.json", symbol_hyperparams)

        split_counts = {
            "train": int(len(X_train_sym)) if isinstance(X_train_sym, torch.Tensor) else 0,
            "val": int(len(X_val_sym)) if isinstance(X_val_sym, torch.Tensor) else 0,
            "test": int(len(X_test_sym)) if isinstance(X_test_sym, torch.Tensor) else 0,
        }
        sym_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "symbol_canonical": symbol,
            "run_id": run_id,
            "interval": args.interval,
            "source_script": source_script,
            "feature_schema_version": feature_schema_version,
            "trained_as_part_of_universe": True,
            "hyperparameters": symbol_hyperparams,
            "split_counts": split_counts,
            "metrics": {
                "test_mse": test_mse if 'test_mse' in locals() else 0.0,
                "prediction_distribution": split_prediction_distribution,
                "test_accuracy": class_metrics.get("test_accuracy", 0.0),
                "directional_accuracy": class_metrics.get("directional_accuracy", 0.0),
                "recall_balance": class_metrics.get("recall_balance", 0.0),
                "directional_coverage": class_metrics.get("directional_coverage", 0.0),
                "up_recall": class_metrics.get("up_recall", 0.0),
                "down_recall": class_metrics.get("down_recall", 0.0),
                "up_precision": class_metrics.get("up_precision", 0.0),
                "down_precision": class_metrics.get("down_precision", 0.0),
                "test_confusion_matrix": class_metrics.get("test_confusion_matrix", []),
            },
        }
        safe_write_json(sym_dir / "training_meta.json", sym_meta)

    logger.info("Universe ARIMA-LSTM training pipeline complete.")


if __name__ == "__main__":
    main()
