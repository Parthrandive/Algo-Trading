import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class CNNPatternModel(nn.Module):
    def __init__(self, time_steps: int = 20, features: int = 5, num_classes: int = 3):
        super().__init__()
        # Input shape: (Batch, Channels=1, Height=time_steps, Width=features)
        
        # Conv2D -> BatchNorm -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # Pool across time, keep features roughly similar
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Calculate flattened size
        self.flatten_dim = 32 * (time_steps // 4) * features
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Note: We don't apply Softmax here because we will use CrossEntropyLoss which includes it.
        # But for inference (predict), we will apply softmax to get probabilities.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, time_steps, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CnnPatternClassifier:
    """
    2D CNN Pattern Model for price direction forecasting.
    
    1. Creates 2D sliding windows of (time_steps, features) from OHLCV data.
    2. Labels the next bar's close direction as up (0), neutral (1), down (2).
    3. Trains a 2D CNN on this objective.
    """
    def __init__(
        self,
        window_size: int = 20,
        learning_rate: float = 0.001,
        neutral_threshold: float = 0.001,
        confidence_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.neutral_threshold = neutral_threshold
        self.confidence_threshold = confidence_threshold
        
        # Standard features for OHLCV
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        self.num_classes = 3
        
        self.model = None
        self.is_trained = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_data(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates sliding windows for CNN training."""
        features = df[self.feature_columns].values
        close_prices = df[target_col].values
        
        X, y = [], []
        for i in range(len(features) - self.window_size):
            # The 2D image is shape (window_size, num_features)
            window = features[i:(i + self.window_size)]
            
            # Subtlety: we could normalize the window here, e.g. min-max scaling per window
            # to make it scale-invariant. For simplicity, we just pass raw values or globally scaled values.
            # In a real system, you'd want:
            # window_min = window.min(axis=0)
            # window_max = window.max(axis=0)
            # window = (window - window_min) / (window_max - window_min + 1e-8)
            X.append(window)
            
            # Next bar close
            current_close = close_prices[i + self.window_size - 1]
            next_close = close_prices[i + self.window_size]
            
            pct_change = (next_close - current_close) / current_close
            
            # Classes: 0: up, 1: neutral, 2: down
            if pct_change > self.neutral_threshold:
                label = 0
            elif pct_change < -self.neutral_threshold:
                label = 2
            else:
                label = 1
                
            y.append(label)
            
        # Shape needed for PyTorch Conv2d: (batch, channels, height, width)
        # channels=1, height=window_size, width=num_features
        X_np = np.array(X)
        X_np = np.expand_dims(X_np, axis=1) # Add channel dim
        
        return torch.tensor(X_np, dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

    def fit(self, df: pd.DataFrame, target_col: str = 'close', epochs: int = 20, batch_size: int = 32):
        """
        Trains the CNN Model. Expects df to be sorted by time.
        """
        if len(df) < self.window_size + 10:
            raise ValueError(f"Not enough data to train CNN (needs > {self.window_size + 10} rows).")
            
        logger.info(f"Preparing data for CNN (window_size={self.window_size})...")
        
        X, y = self._prepare_data(df, target_col=target_col)
        X, y = X.to(self.device), y.to(self.device)
        
        self.model = CNNPatternModel(
            time_steps=self.window_size, 
            features=len(self.feature_columns),
            num_classes=self.num_classes
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Training CNN...")
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataset):.4f}, Acc: {accuracy:.2f}%")
                
        self.is_trained = True
        logger.info("CNN Training complete.")

    def predict(self, df: pd.DataFrame, confidence_threshold: Optional[float] = None) -> Tuple[str, Dict[str, float]]:
        """
        Predict the class probabilities for the NEXT bar using the most recent `window_size` bars.
        Returns: (predicted_class_name, probabilities_dict)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
            
        if len(df) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} bars to predict, got {len(df)}.")
            
        # Get the last window
        window = df[self.feature_columns].values[-self.window_size:]
        X_test_np = np.expand_dims(np.expand_dims(window, axis=0), axis=1) # (1, 1, window_size, features)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        class_names = ["up", "neutral", "down"]
        predicted_idx = int(np.argmax(probs))
        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        if float(np.max(probs)) < float(threshold):
            predicted_idx = 1  # neutral on low confidence
        
        prob_dict = {
            class_names[0]: float(probs[0]),
            class_names[1]: float(probs[1]),
            class_names[2]: float(probs[2]),
        }
        
        return class_names[predicted_idx], prob_dict

    def save(self, path: str = "data/models/cnn_pattern/"):
        """Save weights and hyperparameters."""
        os.makedirs(path, exist_ok=True)
        
        # Save Hyperparams
        hyperparams = {
            "window_size": self.window_size,
            "learning_rate": self.learning_rate,
            "feature_columns": self.feature_columns,
            "neutral_threshold": self.neutral_threshold,
            "confidence_threshold": self.confidence_threshold,
        }
        with open(os.path.join(path, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=4)
            
        if self.model:
            torch.save(self.model.state_dict(), os.path.join(path, "cnn_weights.pt"))
            
    def load(self, path: str = "data/models/cnn_pattern/"):
        """Load weights and hyperparameters."""
        with open(os.path.join(path, "hyperparams.json"), "r") as f:
            hp = json.load(f)
            
        self.window_size = hp["window_size"]
        self.learning_rate = hp["learning_rate"]
        self.feature_columns = hp["feature_columns"]
        self.neutral_threshold = hp["neutral_threshold"]
        self.confidence_threshold = hp.get("confidence_threshold", 0.5)
        
        weights_path = os.path.join(path, "cnn_weights.pt")
        if os.path.exists(weights_path):
            self.model = CNNPatternModel(
                time_steps=self.window_size, 
                features=len(self.feature_columns),
                num_classes=self.num_classes
            ).to(self.device)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            self.is_trained = True
