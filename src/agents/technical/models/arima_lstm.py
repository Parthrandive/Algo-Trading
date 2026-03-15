import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime

import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class LSTMResidualModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backprop through time (if applicable)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

class ArimaLstmHybrid:
    """
    ARIMA-LSTM Hybrid Model for price forecasting.
    
    1. Fits an ARIMA model on the 'close' price to capture linear trends.
    2. Calculates residuals (actual close - ARIMA predicted close).
    3. Trains an LSTM on engineered features to predict the ARIMA residuals.
    4. Prediction = ARIMA forecast + LSTM residual forecast.
    """
    def __init__(self, arima_order: Tuple[int, int, int] = (5, 1, 0), 
                 lstm_hidden_size: int = 64, 
                 lstm_layers: int = 1,
                 learning_rate: float = 0.001,
                 window_size: int = 10):
        self.arima_order = arima_order
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        self.arima_results = None
        self.lstm_model = None
        self.feature_columns = []
        self.is_trained = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_lstm_data(self, df: pd.DataFrame, target_series: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates sliding windows for LSTM training."""
        features = df[self.feature_columns].values
        targets = target_series.values
        
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:(i + self.window_size)])
            y.append(targets[i + self.window_size])
            
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def fit(self, df: pd.DataFrame, target_col: str = 'close', epochs: int = 50, batch_size: int = 32):
        """
        Trains the Hybrid model. Expects df to be sorted by time and include 'close'.
        We train on the whole dataset provided (assume train/val split is handled externally).
        """
        if len(df) < self.window_size + 10:
            raise ValueError("Not enough data to train ARIMA-LSTM.")
        
        # Identify feature columns (everything except symbol, timestamp, target, and standard OHLC that aren't engineered)
        exclude_cols = {'symbol', 'timestamp', target_col}
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        
        # 1. Fit ARIMA on the target series
        logger.info(f"Fitting ARIMA{self.arima_order} on {target_col}...")
        arima_model = sm.tsa.ARIMA(df[target_col].values, order=self.arima_order)
        self.arima_results = arima_model.fit()
        
        # 2. Get ARIMA Residuals
        # predictions are 1-step ahead in-sample
        arima_preds = self.arima_results.predict(typ='levels')
        # Since differencing drops first elements, align properly.
        # Note: statsmodels predictions typically match the length, but first few might be 0/volatile
        residuals = df[target_col].values - arima_preds
        residuals_series = pd.Series(residuals, index=df.index)
        
        # 3. Train LSTM on Residuals
        logger.info(f"Preparing data for LSTM (window_size={self.window_size})...")
        # We need to drop Nans from feature engineering before making tensors
        df_clean = df.copy()
        df_clean['residual'] = residuals_series
        df_clean = df_clean.dropna()
        
        if len(df_clean) <= self.window_size:
            raise ValueError(f"Not enough data after dropping NaNs! (len={len(df_clean)})")
            
        X, y = self._prepare_lstm_data(df_clean, df_clean['residual'])
        X, y = X.to(self.device), y.to(self.device)
        
        self.lstm_model = LSTMResidualModel(
            input_size=len(self.feature_columns),
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Training LSTM...")
        self.lstm_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataset):.4f}")
                
        self.is_trained = True
        logger.info("ARIMA-LSTM Training complete.")

    def predict(self, df: pd.DataFrame, target_col: str = 'close') -> float:
        """
        Given the historical context (at least `window_size` + lags),
        predict 1 step ahead.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
            
        # 1. ARIMA forecast
        # We append the new data to the existing ARIMA model to forecast 1 step
        # statsmodels `apply` or just re-fitting quickly. For simplicity in this demo,
        # we predict using the last trained state if it's contiguous, else we'd use `apply`.
        arima_model_ext = sm.tsa.ARIMA(df[target_col].values, order=self.arima_order)
        arima_res_ext = arima_model_ext.fit() # Fit on available context to get the local state
        forecast_arima = arima_res_ext.forecast(steps=1)[0]
        
        # 2. LSTM residual prediction
        # Get the last window for LSTM
        last_window = dict()
        for col in self.feature_columns:
            # Handle possible NaNs in context
            last_window[col] = df[col].ffill().bfill().values[-self.window_size:]
            
        X_test_np = np.column_stack([last_window[c] for c in self.feature_columns])
        X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.lstm_model.eval()
        with torch.no_grad():
            forecast_residual = self.lstm_model(X_test).item()
            
        # 3. Combine
        final_forecast = forecast_arima + forecast_residual
        return final_forecast

    def save(self, path: str = "data/models/arima_lstm/"):
        """Save weights and hyperparameters."""
        os.makedirs(path, exist_ok=True)
        
        # Save Hyperparams
        hyperparams = {
            "arima_order": self.arima_order,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_layers": self.lstm_layers,
            "learning_rate": self.learning_rate,
            "window_size": self.window_size,
            "feature_columns": self.feature_columns
        }
        with open(os.path.join(path, "hyperparams.json"), "w") as f:
            json.dump(hyperparams, f, indent=4)
            
        # Save LSTM
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), os.path.join(path, "lstm_weights.pt"))
            
        # ARIMA save is complex in sm, skipping binary for this sprint, we re-fit context on predict

    def load(self, path: str = "data/models/arima_lstm/"):
        """Load weights and hyperparameters."""
        with open(os.path.join(path, "hyperparams.json"), "r") as f:
            hp = json.load(f)
            
        self.arima_order = tuple(hp["arima_order"])
        self.lstm_hidden_size = hp["lstm_hidden_size"]
        self.lstm_layers = hp["lstm_layers"]
        self.learning_rate = hp["learning_rate"]
        self.window_size = hp["window_size"]
        self.feature_columns = hp["feature_columns"]
        
        weights_path = os.path.join(path, "lstm_weights.pt")
        if os.path.exists(weights_path):
            self.lstm_model = LSTMResidualModel(
                input_size=len(self.feature_columns),
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_layers
            ).to(self.device)
            self.lstm_model.load_state_dict(torch.load(weights_path))
            self.lstm_model.eval()
            self.is_trained = True
