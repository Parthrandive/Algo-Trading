import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agents.technical.data_loader import DataLoader
from src.agents.technical.models.cnn_pattern import CnnPatternClassifier, CNNPatternModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set reproducibility seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_data(df: pd.DataFrame) -> None:
    """Validate data quality before training."""
    if len(df) < 40: # Lowered from 60 to support smaller datasets
        raise ValueError(f"Need at least 40 rows to train CNN Pattern Classifier. Got {len(df)}.")

    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required_cols:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.05:
            raise ValueError(f"Column '{col}' has {nan_pct:.1%} NaNs (max 5% allowed).")

    if df['close'].nunique() == 1:
        logger.warning("Target column 'close' is constant. Model cannot learn.")


def prepare_data_with_normalization(
    classifier: CnnPatternClassifier, df: pd.DataFrame, normalize: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates sliding windows with optional per-window min-max scaling."""
    features = df[classifier.feature_columns].values
    close_prices = df['close'].values
    
    X, y = [], []
    for i in range(len(features) - classifier.window_size):
        window = features[i:(i + classifier.window_size)].copy()
        
        if normalize:
            # Per-window, per-feature min-max scaling
            window_min = window.min(axis=0)
            window_max = window.max(axis=0)
            range_diff = window_max - window_min
            # Prevent div by zero
            range_diff[range_diff == 0] = 1e-8
            window = (window - window_min) / range_diff
            
        X.append(window)
        
        # Next bar close
        current_close = close_prices[i + classifier.window_size - 1]
        next_close = close_prices[i + classifier.window_size]
        
        if current_close == 0:
            pct_change = 0
        else:
            pct_change = (next_close - current_close) / current_close
        
        if pct_change > classifier.neutral_threshold:
            label = 0  # up
        elif pct_change < -classifier.neutral_threshold:
            label = 2  # down
        else:
            label = 1  # neutral
            
        y.append(label)
        
    X_np = np.array(X)
    X_np = np.expand_dims(X_np, axis=1) # (batch, 1, window_size, attrs)
    
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)


def custom_train_cnn(
    classifier: CnnPatternClassifier,
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
    classifier.model = CNNPatternModel(
        time_steps=classifier.window_size, 
        features=len(classifier.feature_columns),
        num_classes=classifier.num_classes
    ).to(classifier.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.model.parameters(), lr=classifier.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_weights_path = os.path.join(output_dir, "cnn_weights.pt")
    
    logger.info("Training CNN...")
    
    for epoch in range(epochs):
        # Train
        classifier.model.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = classifier.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_loss /= len(train_dataset)
        train_acc = correct_train / total_train

        # Validate
        classifier.model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = classifier.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = correct_val / total_val

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(classifier.model.state_dict(), best_weights_path)
            logger.debug(f"Epoch {epoch+1:03d}: Val loss improved to {val_loss:.4f}, saving best weights.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (val loss did not improve for {patience} epochs).")
                break
                
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%"
            )

    # Load best weights
    if os.path.exists(best_weights_path):
        classifier.model.load_state_dict(torch.load(best_weights_path))
    classifier.model.eval()
    classifier.is_trained = True
    
    return {"train_loss": train_loss, "val_loss": best_val_loss, "val_acc": best_val_acc, "epochs_run": epoch + 1}


def main():
    parser = argparse.ArgumentParser(description="Train standalone CNN Pattern model.")
    parser.add_argument("--symbol", default="TATASTEEL.NS", help="Stock symbol to train on")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to fetch from DB")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--window-size", type=int, default=20, help="Time steps per window")
    parser.add_argument("--neutral-threshold", type=float, default=0.001, help="Threshold for neutral price change")
    parser.add_argument("--normalize", action="store_true", default=True, help="Use per-window normalization")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/models/cnn_pattern/", help="Output directory")
    parser.add_argument("--use-nse", action="store_true", help="Fetch data natively from NSE if DB is empty/unavailable")
    parser.add_argument("--interval", default="1d", help="Candle interval (e.g. 1d, 1h). Default: 1d")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Fetch Data
    logger.info(f"Fetching data for {args.symbol}...")
    db_url = os.getenv("DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5432/sentinel_db")
    loader = DataLoader(db_url)
    try:
        df = loader.load_historical_bars(args.symbol, limit=args.limit, use_nse_fallback=args.use_nse, min_fallback_rows=40, interval=args.interval)
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Only dropna on the actual open/high/low/close/volume columns to prevent dropping rows due to unrelated columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        df = df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Validate Data
    logger.info(f"Loaded {len(df)} continuous rows. Validating...")
    validate_data(df)

    # 3. Initialize Model & Prepare Tensors
    classifier = CnnPatternClassifier(
        window_size=args.window_size,
        learning_rate=args.lr,
        neutral_threshold=args.neutral_threshold
    )
    classifier.feature_columns = ['open', 'high', 'low', 'close', 'volume']
    
    logger.info(f"Preparing windows (normalize={args.normalize})...")
    X, y = prepare_data_with_normalization(classifier, df, args.normalize)
    X, y = X.to(classifier.device), y.to(classifier.device)
    
    # Analyze Class Distribution
    classes, counts = torch.unique(y, return_counts=True)
    class_dist = {int(c): int(counts[i]) for i, c in enumerate(classes)}
    logger.info(f"Class distribution overall: Up(0): {class_dist.get(0,0)}, Neutral(1): {class_dist.get(1,0)}, Down(2): {class_dist.get(2,0)}")

    # 4. Train/Val Split (80/20 chronological)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}")

    # 5. Train CNN w/ Custom Loop
    metrics = custom_train_cnn(
        classifier, X_train, y_train, X_val, y_val, 
        epochs=args.epochs, batch_size=args.batch_size, 
        patience=args.patience, output_dir=args.output_dir
    )

    # 6. Save Models and Meta
    logger.info(f"Saving model artifacts to {args.output_dir}")
    # Need to keep track of normalization choice in the saved state for predict
    classifier.neutral_threshold = args.neutral_threshold # Ensure it's passed through
    
    hyperparams = {
        "window_size": classifier.window_size,
        "learning_rate": classifier.learning_rate,
        "feature_columns": classifier.feature_columns,
        "neutral_threshold": classifier.neutral_threshold,
        "normalize": args.normalize
    }
    with open(os.path.join(args.output_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)
        
    if classifier.model:
        torch.save(classifier.model.state_dict(), os.path.join(args.output_dir, "cnn_weights.pt"))
    
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "input_rows": len(df),
        "total_windows": len(X),
        "train_windows": len(X_train),
        "val_windows": len(X_val),
        "class_distribution": class_dist,
        "hyperparameters": {
            "window_size": classifier.window_size,
            "learning_rate": classifier.learning_rate,
            "epochs_run": metrics["epochs_run"],
            "batch_size": args.batch_size,
            "patience": args.patience,
            "neutral_threshold": args.neutral_threshold,
            "normalize": args.normalize,
            "seed": args.seed
        },
        "metrics": {
            "final_train_loss": metrics["train_loss"],
            "best_val_loss": metrics["val_loss"],
            "best_val_acc": metrics["val_acc"]
        }
    }
    
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info("CNN Training complete.")


if __name__ == "__main__":
    main()
