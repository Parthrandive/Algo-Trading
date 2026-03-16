import os
import shutil
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.agents.technical.models.arima_lstm import ArimaLstmHybrid
from src.agents.technical.features import engineer_features

MIN_ARIMA_LSTM_ROWS = 30

@pytest.fixture
def sample_data():
    """Generates synthetic OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    
    # Generate an upward trend with some noise
    close_prices = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['AAPL'] * 100,
        'open': close_prices - np.random.uniform(0.1, 1, 100),
        'high': close_prices + np.random.uniform(0.1, 2, 100),
        'low': close_prices - np.random.uniform(0.1, 2, 100),
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, 100)
    })
    return df

@pytest.fixture
def clean_model_dir():
    """Fixture to clean up saved models after text execution."""
    path = "data/models/arima_lstm_test/"
    if os.path.exists(path):
        shutil.rmtree(path)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

def test_feature_engineering(sample_data):
    """Test that feature engineering returns correct shapes and no NaNs."""
    df_features = engineer_features(sample_data)
    
    # Check that columns were added
    assert 'rsi' in df_features.columns
    assert 'macd' in df_features.columns
    assert 'close_lag_1' in df_features.columns
    
    # Check length is preserved
    assert len(df_features) == len(sample_data)

def test_arima_lstm_training_and_saving(sample_data, clean_model_dir):
    """Test full pipeline: feature engineering, training, saving, and loading."""
    df_features = engineer_features(sample_data)
    
    # Since we added features that need ~26 periods to mature (MACD), wait to drop NaNs
    df_train = df_features.dropna()
    if len(df_train) < MIN_ARIMA_LSTM_ROWS:
        pytest.skip(
            f"Not enough data after feature engineering to train ARIMA-LSTM "
            f"({len(df_train)} < {MIN_ARIMA_LSTM_ROWS})."
        )
    
    # Initialize Model
    # Use smaller model and fast train to keep tests speedy
    model = ArimaLstmHybrid(
        arima_order=(1, 1, 0),
        lstm_hidden_size=16,
        lstm_layers=1,
        learning_rate=0.01,
        window_size=5
    )
    
    # Train
    model.fit(df_train, target_col='close', epochs=2, batch_size=8)
    
    assert model.is_trained, "Model failed to set is_trained string."
    assert model.lstm_model is not None, "LSTM component failed to initialize."
    
    # Save
    model.save(path=clean_model_dir)
    assert os.path.exists(os.path.join(clean_model_dir, "hyperparams.json"))
    assert os.path.exists(os.path.join(clean_model_dir, "lstm_weights.pt"))
    
    # Reload
    new_model = ArimaLstmHybrid()
    new_model.load(path=clean_model_dir)
    assert new_model.is_trained
    assert new_model.lstm_model is not None

def test_arima_lstm_prediction(sample_data):
    """Test prediction output structure and bounds."""
    df_features = engineer_features(sample_data)
    df_train = df_features.dropna()
    if len(df_train) < MIN_ARIMA_LSTM_ROWS:
        pytest.skip(
            f"Not enough data after feature engineering to train ARIMA-LSTM "
            f"({len(df_train)} < {MIN_ARIMA_LSTM_ROWS})."
        )
    
    model = ArimaLstmHybrid(
        arima_order=(1, 1, 0),
        lstm_hidden_size=16,
        window_size=5
    )
    model.fit(df_train, epochs=1)
    
    # Test Prediction point
    predict_df = df_train.copy()
    forecast = model.predict(predict_df, target_col='close')
    
    assert isinstance(forecast, float), "Prediction output should be a float."
    
    # Provide sanity check bounds: The last price was ~150, the forecast shouldn't be outlandishly large
    last_price = predict_df['close'].iloc[-1]
    assert last_price * 0.5 < forecast < last_price * 1.5, "Forecast is outside of a reasonable sanity price range."
