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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from src.agents.technical.data_loader import DataLoader
from src.agents.technical.features import engineer_features
from src.agents.technical.models.arima_lstm import ArimaLstmHybrid, LSTMResidualModel

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
    output_dir: str
) -> dict:
    """Custom training loop w/ early stopping, scheduler, and best checkpoint."""
    hybrid.lstm_model = LSTMResidualModel(
        input_size=len(hybrid.feature_columns),
        hidden_size=hybrid.lstm_hidden_size,
        num_layers=hybrid.lstm_layers
    ).to(hybrid.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(hybrid.lstm_model.parameters(), lr=hybrid.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights_path = os.path.join(output_dir, "lstm_weights.pt")
    
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
        hybrid.lstm_model.load_state_dict(torch.load(best_weights_path))
    hybrid.lstm_model.eval()
    hybrid.is_trained = True
    
    return {"train_loss": train_loss, "val_loss": best_val_loss, "epochs_run": epoch + 1}


def main():
    parser = argparse.ArgumentParser(description="Train standalone ARIMA-LSTM model.")
    parser.add_argument("--symbol", default="TATASTEEL.NS", help="Stock symbol to train on")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--arima-order", default="5,1,0", help="ARIMA order (p,d,q)")
    parser.add_argument("--window-size", type=int, default=10, help="LSTM window size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/models/arima_lstm/", help="Output directory")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1d", help="Candle interval, e.g. 1d, 1h. Default: 1d")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    arima_order = tuple(map(int, args.arima_order.split(',')))

    # 1. Fetch Data
    logger.info(f"Fetching data for {args.symbol}...")
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    try:
        df = loader.load_historical_bars(args.symbol, limit=args.limit, use_nse_fallback=args.use_nse, min_fallback_rows=40, interval=args.interval)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Validate Data
    logger.info(f"Loaded {len(df)} rows. Validating...")
    validate_data(df)

    # 3. Feature Engineering
    logger.info("Engineering features...")
    is_forex = args.symbol.endswith("=X")
    df_features = engineer_features(df, is_forex=is_forex)
    
    # Initialize Model
    hybrid = ArimaLstmHybrid(
        arima_order=arima_order,
        learning_rate=args.lr,
        window_size=args.window_size
    )
    
    # Identify feature columns (ignore fully NaN columns and non-numeric types)
    target_col = 'close'
    exclude_cols = {'symbol', 'timestamp', target_col}
    hybrid.feature_columns = [
        c for c in df_features.columns 
        if c not in exclude_cols 
        and pd.api.types.is_numeric_dtype(df_features[c])
        and not df_features[c].isna().all()
        and c not in RAW_MACRO_COLUMNS
    ]
    
    # Now drop rows that have NaNs *only* in our selected features (e.g. from rolling windows)
    df_features = df_features.dropna(subset=hybrid.feature_columns + [target_col]).reset_index(drop=True)
    logger.info(f"Data shape after feature engineering: {df_features.shape}")
    
    logger.info(f"Fitting ARIMA{hybrid.arima_order} on {target_col}...")
    import statsmodels.api as sm
    arima_model = sm.tsa.ARIMA(
        df_features[target_col].values, 
        order=hybrid.arima_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    hybrid.arima_results = arima_model.fit()
    
    arima_preds = hybrid.arima_results.predict(typ='levels')
    residuals = df_features[target_col].values - arima_preds
    df_features['residual'] = residuals

    # 5. Train/Val Split (80/20 chronological)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    val_df = df_features.iloc[split_idx:]
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}")

    X_train, y_train = hybrid._prepare_lstm_data(train_df, train_df['residual'])
    X_val, y_val = hybrid._prepare_lstm_data(val_df, val_df['residual'])
    
    X_train, y_train = X_train.to(hybrid.device), y_train.to(hybrid.device)
    X_val, y_val = X_val.to(hybrid.device), y_val.to(hybrid.device)

    # 6. Train LSTM w/ Custom Loop
    metrics = custom_train_lstm(
        hybrid, X_train, y_train, X_val, y_val, 
        epochs=args.epochs, batch_size=args.batch_size, 
        patience=args.patience, output_dir=args.output_dir
    )

    # 7. Save Models and Meta
    logger.info(f"Saving model artifacts to {args.output_dir}")
    hybrid.save(args.output_dir)
    
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "input_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
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
            "best_val_loss": metrics["val_loss"]
        }
    }
    
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
