import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _probe_torch_import() -> bool:
    """
    Probe torch import in a child process first.
    Some environments hard-abort on native torch import; probing avoids
    crashing the main interpreter during module import.
    """
    if os.getenv("ARIMA_LSTM_DISABLE_TORCH") == "1":
        return False

    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return True

    stderr = (probe.stderr or "").strip().splitlines()
    detail = stderr[-1] if stderr else "torch import probe failed"
    logger.warning(
        "torch backend unavailable for ARIMA-LSTM; using numpy fallback (%s)",
        detail[:240],
    )
    return False


_TORCH_AVAILABLE = False
torch = None
nn = None
optim = None
DataLoader = None
TensorDataset = None

if _probe_torch_import():
    try:
        import torch  # type: ignore[assignment]
        import torch.nn as nn  # type: ignore[assignment]
        import torch.optim as optim  # type: ignore[assignment]
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore[assignment]

        _TORCH_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - guarded fallback path
        logger.warning(
            "torch import failed in-process for ARIMA-LSTM; using numpy fallback (%s)",
            str(exc)[:240],
        )
        _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class LSTMResidualModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0.detach(), c0.detach()))
            return self.fc(out[:, -1, :])
else:
    class LSTMResidualModel:
        """
        Lightweight fallback when torch backend cannot load.
        Uses a linear fit over the last timestep of each feature window.
        """

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.weights = np.zeros(input_size, dtype=float)
            self.bias = 0.0

        def fit(self, X: np.ndarray, y: np.ndarray):
            if X.ndim != 3:
                raise ValueError("Expected X with shape (n_samples, window_size, n_features).")
            features = X[:, -1, :]
            targets = y.reshape(-1).astype(float)
            design = np.column_stack((features, np.ones(len(features), dtype=float)))
            params, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
            self.weights = params[:-1].astype(float)
            self.bias = float(params[-1])

        def predict_residual(self, window: np.ndarray) -> float:
            if window.ndim == 2:
                x = window[-1, :]
            else:
                x = window.reshape(-1)
            return float(np.dot(x.astype(float), self.weights) + self.bias)

        def state_dict(self) -> Dict[str, Any]:
            return {"weights": self.weights.tolist(), "bias": self.bias}

        def load_state_dict(self, state: Dict[str, Any]):
            weights = np.array(state.get("weights", []), dtype=float)
            if len(weights) != self.input_size:
                resized = np.zeros(self.input_size, dtype=float)
                n = min(len(weights), self.input_size)
                if n > 0:
                    resized[:n] = weights[:n]
                self.weights = resized
            else:
                self.weights = weights
            self.bias = float(state.get("bias", 0.0))

        def eval(self):
            return self

        def train(self):
            return self

class ArimaLstmHybrid:
    """
    ARIMA-LSTM Hybrid Model for price forecasting.
    
    1. Fits an ARIMA model on the 'close' price to capture linear trends.
    2. Calculates residuals (actual close - ARIMA predicted close).
    3. Trains an LSTM on engineered features to predict the ARIMA residuals.
    4. Prediction = ARIMA forecast + LSTM residual forecast.
    """
    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (5, 1, 0),
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        learning_rate: float = 0.001,
        window_size: int = 10,
    ):
        self.arima_order = arima_order
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        self.arima_results = None
        self.lstm_model = None
        self.feature_columns = []
        self.is_trained = False

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if _TORCH_AVAILABLE
            else "cpu"
        )

    def _prepare_lstm_data(self, df: pd.DataFrame, target_series: pd.Series) -> Tuple[Any, Any]:
        """Creates sliding windows for LSTM training."""
        features = df[self.feature_columns].values
        targets = target_series.values

        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:(i + self.window_size)])
            y.append(targets[i + self.window_size])

        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32).reshape(-1, 1)

        if _TORCH_AVAILABLE:
            return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
        return X_np, y_np

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        epochs: int = 50,
        batch_size: int = 32,
    ):
        """
        Trains the Hybrid model. Expects df to be sorted by time and include 'close'.
        We train on the whole dataset provided (assume train/val split is handled externally).
        """
        if len(df) < self.window_size + 10:
            raise ValueError("Not enough data to train ARIMA-LSTM.")

        # Identify feature columns (everything except symbol, timestamp, target, and standard OHLC that aren't engineered)
        exclude_cols = {"symbol", "timestamp", target_col}
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]

        # 1. Fit ARIMA on the target series
        logger.info(f"Fitting ARIMA{self.arima_order} on {target_col}...")
        arima_model = sm.tsa.ARIMA(df[target_col].values, order=self.arima_order)
        self.arima_results = arima_model.fit()

        # 2. Get ARIMA Residuals
        # predictions are 1-step ahead in-sample
        arima_preds = self.arima_results.predict(typ="levels")
        # Since differencing drops first elements, align properly.
        # Note: statsmodels predictions typically match the length, but first few might be 0/volatile
        residuals = df[target_col].values - arima_preds
        residuals_series = pd.Series(residuals, index=df.index)

        # 3. Train LSTM on Residuals
        logger.info(f"Preparing data for LSTM (window_size={self.window_size})...")
        # We need to drop Nans from feature engineering before making tensors
        df_clean = df.copy()
        df_clean["residual"] = residuals_series
        df_clean = df_clean.dropna()

        if len(df_clean) <= self.window_size:
            raise ValueError(f"Not enough data after dropping NaNs! (len={len(df_clean)})")

        X, y = self._prepare_lstm_data(df_clean, df_clean["residual"])

        if _TORCH_AVAILABLE:
            X, y = X.to(self.device), y.to(self.device)
            self.lstm_model = LSTMResidualModel(
                input_size=len(self.feature_columns),
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_layers,
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            logger.info("Training LSTM (torch backend)...")
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
                    logger.debug(
                        f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataset):.4f}"
                    )
        else:
            logger.info("Training residual model (numpy fallback backend)...")
            self.lstm_model = LSTMResidualModel(
                input_size=len(self.feature_columns),
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_layers,
            )
            self.lstm_model.fit(X, y)

        self.is_trained = True
        logger.info("ARIMA-LSTM Training complete.")

    def predict(self, df: pd.DataFrame, target_col: str = "close") -> float:
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
        arima_res_ext = arima_model_ext.fit()  # Fit on available context to get the local state
        forecast_arima = arima_res_ext.forecast(steps=1)[0]

        # 2. LSTM residual prediction
        # Get the last window for LSTM
        last_window = dict()
        for col in self.feature_columns:
            # Handle missing/all-NaN columns robustly during inference.
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
            else:
                series = pd.Series(np.nan, index=df.index, dtype=float)

            series = series.ffill().bfill()
            if series.isna().all():
                values = np.zeros(self.window_size, dtype=float)
            else:
                values = series.fillna(0.0).values
                if len(values) >= self.window_size:
                    values = values[-self.window_size:]
                else:
                    pad = np.full(self.window_size - len(values), values[0], dtype=float)
                    values = np.concatenate([pad, values])
            last_window[col] = values

        X_test_np = np.column_stack([last_window[c] for c in self.feature_columns])

        if _TORCH_AVAILABLE:
            X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.lstm_model.eval()
            with torch.no_grad():
                forecast_residual = self.lstm_model(X_test).item()
        else:
            forecast_residual = self.lstm_model.predict_residual(X_test_np)

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
        weights_path = os.path.join(path, "lstm_weights.pt")
        if self.lstm_model:
            if _TORCH_AVAILABLE:
                torch.save(self.lstm_model.state_dict(), weights_path)
            else:
                with open(weights_path, "w") as f:
                    json.dump(self.lstm_model.state_dict(), f)

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
            )
            if _TORCH_AVAILABLE:
                self.lstm_model = self.lstm_model.to(self.device)
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                except Exception:
                    with open(weights_path, "r") as f:
                        state_dict = json.load(f)
                self.lstm_model.load_state_dict(state_dict)
                self.lstm_model.eval()
            else:
                with open(weights_path, "r") as f:
                    state = json.load(f)
                self.lstm_model.load_state_dict(state)
            self.is_trained = True
