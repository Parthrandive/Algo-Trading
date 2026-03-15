import pytest
from datetime import datetime, timezone
import pandas as pd
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from src.agents.technical.schemas import TechnicalPrediction
from src.agents.technical.data_loader import DataLoader

def test_technical_prediction_schema_valid():
    """Test valid instantiation of TechnicalPrediction schema"""
    prediction = TechnicalPrediction(
        symbol="TATASTEEL.NS",
        timestamp=datetime.now(timezone.utc),
        price_forecast=150.5,
        direction="up",
        volatility_estimate=0.015,
        var_95=-2.5,
        var_99=-3.5,
        es_95=-3.0,
        es_99=-4.0,
        confidence=0.85,
        model_id="arima_lstm_v1"
    )
    
    assert prediction.symbol == "TATASTEEL.NS"
    assert prediction.direction == "up"
    assert prediction.confidence == 0.85
    assert prediction.schema_version == "1.0"

def test_technical_prediction_schema_invalid_direction():
    """Test schema fails on invalid prediction direction"""
    with pytest.raises(ValidationError):
        TechnicalPrediction(
             symbol="TATASTEEL.NS",
             timestamp=datetime.now(timezone.utc),
             price_forecast=150.5,
             direction="sideways",  # Invalid! Not up/down/neutral
             volatility_estimate=0.015,
             var_95=-2.5,
             var_99=-3.5,
             es_95=-3.0,
             es_99=-4.0,
             confidence=0.85,
             model_id="arima_lstm_v1"
        )
        
def test_technical_prediction_schema_invalid_confidence():
    """Test schema fails on out-of-bounds confidence score"""
    with pytest.raises(ValidationError):
         TechnicalPrediction(
              symbol="TATASTEEL.NS",
              timestamp=datetime.now(timezone.utc),
              price_forecast=150.5,
              direction="up",
              volatility_estimate=0.015,
              var_95=-2.5,
              var_99=-3.5,
              es_95=-3.0,
              es_99=-4.0,
              confidence=1.5, # Out of 0-1 bounds!
              model_id="arima_lstm_v1"
         )

@patch('src.agents.technical.data_loader.create_engine')
@patch('src.agents.technical.data_loader.pd.read_sql')
def test_data_loader_smoke(mock_read_sql, mock_create_engine):
    """Smoke test for DataLoader history fetching mechanism"""
    
    # Mock return value
    mock_df = pd.DataFrame({
         'symbol': ['WIPRO.NS', 'WIPRO.NS'],
         'timestamp': ['2026-03-09T03:45:00Z', '2026-03-09T04:45:00Z'],
         'close': [194.69, 196.16],
         'volume': [2558832, 2728430]
    })
    mock_read_sql.return_value = mock_df
    
    loader = DataLoader("sqlite:///:memory:")
    df = loader.load_historical_bars("WIPRO.NS", limit=2)
    
    assert not df.empty
    assert len(df) == 2
    assert "close" in df.columns
    assert mock_read_sql.called

@patch('src.agents.technical.data_loader.create_engine')
@patch('src.agents.technical.data_loader.pd.read_sql')
def test_data_loader_uses_parameterized_query(mock_read_sql, mock_create_engine):
    """Ensure query uses bind parameters instead of direct string interpolation."""
    mock_read_sql.return_value = pd.DataFrame(
        {"symbol": ["X"], "timestamp": ["2026-03-09T03:45:00Z"], "close": [100.0]}
    )

    loader = DataLoader("sqlite:///:memory:")
    malicious_symbol = "ABC'; DROP TABLE sentinel_db.ohlcv_bars; --"
    loader.load_historical_bars(malicious_symbol, limit=1)

    _, kwargs = mock_read_sql.call_args
    query = mock_read_sql.call_args.args[0]
    query_text = str(query)

    assert ":symbol" in query_text
    assert ":limit" in query_text
    assert kwargs["params"]["symbol"] == malicious_symbol
    assert kwargs["params"]["limit"] == 1
