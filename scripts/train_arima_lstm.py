import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from src.agents.technical.models.arima_lstm import ArimaLstmHybrid, LSTMResidualModel
from config.symbols import SplitCounts, SymbolValidationResult, discover_training_symbols, is_forex, validate_equity_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
) -> dict:
    """Custom training loop w/ early stopping, scheduler, and best checkpoint."""
    hybrid.lstm_model = LSTMResidualModel(
        input_size=len(hybrid.feature_columns),
        hidden_size=hybrid.lstm_hidden_size,
        num_layers=hybrid.lstm_layers
    ).to(hybrid.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hybrid.lstm_model.parameters(), lr=hybrid.learning_rate)
    # Keras callback equivalent for val_loss monitoring:
    # EarlyStopping(patience=15, restore_best_weights=True)
    # ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
    # ModelCheckpoint(filepath='best_model_{symbol}.keras', save_best_only=True)
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
    best_weights_path = os.path.join("/tmp", f"best_model_{symbol}.keras")
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
        train_loss /= len(train_dataset)

        # Validate
        hybrid.lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = hybrid.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            epochs_no_improve = 0
            torch.save(hybrid.lstm_model.state_dict(), best_weights_path)
            logger.debug(f"Epoch {epoch+1:03d}: Val loss improved to {val_loss:.6f}, saving best weights.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def rolling_residuals(close_series: pd.Series) -> np.ndarray:
        nonlocal arima_results
        residuals: list[float] = []
        history_values = train_close.values.copy()
        for actual in close_series.values:
            try:
                forecast = float(arima_results.forecast(steps=1)[0])
            except Exception:
                forecast = float(history_values[-1]) if len(history_values) else float(actual)

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
        return np.asarray(residuals, dtype=float)

    val_residuals = rolling_residuals(val_close)
    test_residuals = rolling_residuals(test_close)
    return train_residuals, val_residuals, test_residuals


def evaluate_mse(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
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


def quality_gate_split_counts(df: pd.DataFrame) -> SplitCounts:
    n_rows = len(df)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return SplitCounts(
        train_rows=train_end,
        val_rows=val_end - train_end,
        test_rows=n_rows - val_end,
    )


def main():
    parser = argparse.ArgumentParser(description="Train standalone ARIMA-LSTM model.")
    parser.add_argument("--symbol", default=None, help="Optional single equity symbol to train on")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order (p,d,q)")
    parser.add_argument("--window-size", type=int, default=10, help="LSTM window size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/models/arima_lstm/", help="Output directory")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1d", help="Candle interval, e.g. 1d, 1h. Default: 1d")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    arima_order = tuple(map(int, args.arima_order.split(',')))

    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)

    def validate_symbol(symbol: str):
        try:
            frame = loader.load_historical_bars(
                symbol,
                limit=args.limit,
                use_nse_fallback=args.use_nse,
                min_fallback_rows=40,
                interval=args.interval,
            )
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
        requested_symbols=[args.symbol] if args.symbol else None,
        database_url=db_url,
        validator=validate_symbol,
        print_fn=lambda message: logger.info(message),
    )
    training_symbols = list(discovery.active_symbols)
    if not training_symbols:
        logger.error("No active equity symbols passed the training quality gate.")
        sys.exit(1)
    if args.symbol is None and len(training_symbols) > 1:
        logger.info("No --symbol provided. Using first active discovered symbol: %s", training_symbols[0])
    for symbol in training_symbols:
        assert not is_forex(symbol), (
            f"{symbol} is a forex symbol and must never "
            f"be trained as a prediction target. "
            f"It must be used as an external feature only. "
            f"Remove it from the training symbol list."
        )
    args.symbol = args.symbol or training_symbols[0]

    # 1. Fetch Data
    logger.info(f"Fetching data for {args.symbol}...")
    try:
        df = loader.load_historical_bars(
            args.symbol,
            limit=args.limit,
            use_nse_fallback=args.use_nse,
            min_fallback_rows=40,
            interval=args.interval,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    if getattr(loader, "last_macro_quality_report", None):
        macro_report_path = os.path.join(args.output_dir, "macro_feature_validation.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(macro_report_path, "w") as f:
            json.dump(loader.last_macro_quality_report, f, indent=2)
        excluded = getattr(loader, "last_macro_excluded_features", [])
        logger.info(
            "Macro quality report saved to %s. Excluded by coverage gate: %s",
            macro_report_path,
            excluded,
        )

    # 2. Validate Data
    logger.info(f"Loaded {len(df)} rows. Validating...")
    validate_data(df)

    # 3. Feature Engineering
    logger.info("Engineering features...")
    is_forex_target = is_forex(args.symbol)
    df_features = engineer_features(df, is_forex=is_forex_target)
    df_features = df_features.sort_values('timestamp').reset_index(drop=True)

    # Initialize Model
    hybrid = ArimaLstmHybrid(
        arima_order=arima_order,
        learning_rate=args.lr,
        window_size=args.window_size
    )

    # 4. Strict chronological split: 70% train, 15% val, 15% test
    n = len(df_features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_raw = df_features.iloc[:train_end].copy()
    val_raw = df_features.iloc[train_end:val_end].copy()
    test_raw = df_features.iloc[val_end:].copy()

    # Identify feature columns from the train split only
    target_col = 'close'
    exclude_cols = {'symbol', 'timestamp', target_col}
    hybrid.feature_columns = [
        c for c in train_raw.columns
        if c not in exclude_cols 
        and pd.api.types.is_numeric_dtype(train_raw[c])
        and not train_raw[c].isna().all()
        and c not in RAW_MACRO_COLUMNS
    ]

    train_df = train_raw.dropna(subset=hybrid.feature_columns + [target_col]).reset_index(drop=True)
    val_df = val_raw.dropna(subset=hybrid.feature_columns + [target_col]).reset_index(drop=True)
    test_df = test_raw.dropna(subset=hybrid.feature_columns + [target_col]).reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) < hybrid.window_size + 5:
        raise ValueError(
            f"Insufficient rows after feature cleanup. "
            f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, "
            f"window_size={hybrid.window_size}"
        )

    # Fit scaler on train only; transform val/test without refit.
    scaler = StandardScaler()
    train_df.loc[:, hybrid.feature_columns] = scaler.fit_transform(train_df[hybrid.feature_columns].values)
    val_df.loc[:, hybrid.feature_columns] = scaler.transform(val_df[hybrid.feature_columns].values)
    test_df.loc[:, hybrid.feature_columns] = scaler.transform(test_df[hybrid.feature_columns].values)

    logger.info(f"Fitting ARIMA{hybrid.arima_order} on train split only...")
    train_residuals, val_residuals, test_residuals = build_no_leakage_residuals(
        train_close=train_df[target_col],
        val_close=val_df[target_col],
        test_close=test_df[target_col],
        arima_order=hybrid.arima_order,
    )
    train_df['residual'] = train_residuals
    val_df['residual'] = val_residuals
    test_df['residual'] = test_residuals

    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train, y_train = hybrid._prepare_lstm_data(train_df, train_df['residual'])
    X_val, y_val = hybrid._prepare_lstm_data(val_df, val_df['residual'])
    X_test, y_test = hybrid._prepare_lstm_data(test_df, test_df['residual'])

    X_train, y_train = X_train.to(hybrid.device), y_train.to(hybrid.device)
    X_val, y_val = X_val.to(hybrid.device), y_val.to(hybrid.device)
    X_test, y_test = X_test.to(hybrid.device), y_test.to(hybrid.device)

    # 6. Train LSTM w/ Custom Loop
    metrics = custom_train_lstm(
        hybrid, X_train, y_train, X_val, y_val, 
        epochs=args.epochs, batch_size=args.batch_size, 
        patience=args.patience, output_dir=args.output_dir, symbol=args.symbol
    )
    logger.info(f"Total epochs trained (early stopping aware): {metrics['epochs_run']}")

    train_mse = evaluate_mse(hybrid.lstm_model, X_train, y_train, args.batch_size)
    val_mse = evaluate_mse(hybrid.lstm_model, X_val, y_val, args.batch_size)
    test_mse = evaluate_mse(hybrid.lstm_model, X_test, y_test, args.batch_size)
    logger.info(f"Final MSE - Train: {train_mse:.6f}, Val: {val_mse:.6f}, Test: {test_mse:.6f}")

    # 7. Save Models and Meta
    # Skip persisting heavy .pt weights locally (saved to /tmp/ during training for early stopping only)
    # hybrid.save(args.output_dir)  # disabled to avoid local disk bloat
    
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "input_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_count": len(hybrid.feature_columns),
        "hyperparameters": {
            "arima_order": hybrid.arima_order,
            "window_size": hybrid.window_size,
            "learning_rate": hybrid.learning_rate,
            "epochs_run": metrics["epochs_run"],
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed
        },
        "metrics": {
            "final_train_loss": metrics["train_loss"],
            "best_train_loss": metrics["best_train_loss"],
            "best_val_loss": metrics["val_loss"],
            "train_mse": train_mse,
            "val_mse": val_mse,
            "test_mse": test_mse,
        }
    }
    
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
