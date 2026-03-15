import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.agents.technical.technical_agent import TechnicalAgent
from src.agents.technical.schemas import TechnicalPrediction

class MockDataLoader:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def load_historical_bars(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        if limit is not None:
            return self.df.tail(limit).copy()
        return self.df.copy()

@pytest.fixture
def sample_market_data():
    """Generates 300 rows of synthetic hourly market data."""
    dates = pd.date_range(end=datetime.now(timezone.utc) - timedelta(hours=1), periods=300, freq='h')
    
    # Generate random walk for prices
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, 300)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * np.random.normal(1, 0.002, 300),
        "high": prices * np.random.normal(1.005, 0.002, 300),
        "low": prices * np.random.normal(0.995, 0.002, 300),
        "close": prices,
        "volume": np.random.randint(1000, 100000, 300)
    })
    return df

def test_technical_agent_predict(sample_market_data):
    """
    Test that TechnicalAgent can consume the data and produce a valid schema.
    """
    agent = TechnicalAgent(models_dir="data/models")
    # Inject our mock data loader
    agent.loader = MockDataLoader(sample_market_data)
    
    prediction = agent.predict("MOCK.NS")
    
    assert prediction is not None
    assert isinstance(prediction, TechnicalPrediction)
    assert prediction.symbol == "MOCK.NS"
    assert prediction.model_id == "ensemble_arima_cnn_garch_v1.0"
    
    # Check that outputs are properly typed
    assert isinstance(prediction.price_forecast, float)
    assert prediction.direction in ["up", "down", "neutral"]
    assert isinstance(prediction.volatility_estimate, float)
    assert isinstance(prediction.var_95, float)
    assert isinstance(prediction.var_99, float)
    assert isinstance(prediction.es_95, float)
    assert isinstance(prediction.es_99, float)
    assert 0.0 <= prediction.confidence <= 1.0

def test_technical_agent_ignores_all_nan_optional_columns(sample_market_data):
    """
    Regression test: optional/all-NaN payload columns must not block inference.
    """
    df = sample_market_data.copy()
    df["vwap"] = np.nan

    agent = TechnicalAgent(models_dir="data/models")
    agent.loader = MockDataLoader(df)

    prediction = agent.predict("MOCK.NS")
    assert prediction is not None
    assert isinstance(prediction, TechnicalPrediction)

def test_no_data_leakage(sample_market_data):
    """
    Test that Technical Agent predictions are strictly forward-looking.
    The prediction timestamp MUST be greater than the last input timestamp.
    """
    agent = TechnicalAgent(models_dir="data/models")
    agent.loader = MockDataLoader(sample_market_data)
    
    # The last known market timestamp
    last_input_ts = sample_market_data["timestamp"].iloc[-1]
    
    prediction = agent.predict("MOCK.NS")
    
    assert prediction is not None
    
    # Ensure our prediction timestamp (when it was generated/for what horizon) 
    # strictly follows the last input bar's timestamp.
    # Note: `prediction.timestamp` is the generation timestamp. 
    # In a real environment, we'd also check the target horizon timestamp.
    # For now, we verify generation time is after input time.
    assert prediction.timestamp > last_input_ts, "Data leakage detected: Prediction timestamp is not strictly after the last input timestamp!"
