import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch

from src.agents.technical.models.cnn_pattern import CNNPatternModel, CnnPatternClassifier

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing CNN."""
    dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(100)]
    np.random.seed(42)
    data = {
        'symbol': ['RELIANCE.NS'] * 100,
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(150, 250, 100),
        'low': np.random.uniform(50, 150, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }
    # Create simple trends to ensure multiple classes are generated
    for i in range(50, 100):
        data['close'][i] = data['close'][i-1] * 1.01 # Consistent up trend for a bit
        
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def test_cnn_forward_shape():
    """Test that the CNN model produces the expected output shape."""
    time_steps = 20
    features = 5
    batch_size = 4
    num_classes = 3
    
    model = CNNPatternModel(time_steps=time_steps, features=features, num_classes=num_classes)
    
    # Input shape: (Batch, Channels=1, Height=time_steps, Width=features)
    dummy_input = torch.randn(batch_size, 1, time_steps, features)
    
    output = model(dummy_input)
    
    # Output should be (Batch, NumClasses)
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"

def test_cnn_classifier_fit_predict(sample_ohlcv_data):
    """Test the training loop and predict method of the CnnPatternClassifier."""
    classifier = CnnPatternClassifier(window_size=20, learning_rate=0.01)
    
    # 1. Test fit
    # Training for 2 epochs as per requirements
    classifier.fit(sample_ohlcv_data, epochs=2, batch_size=16)
    
    assert classifier.is_trained is True
    assert classifier.model is not None
    
    # 2. Test predict
    # Predict takes a dataframe and uses the last `window_size` rows to output probabilities
    predicted_class, probs = classifier.predict(sample_ohlcv_data)
    
    assert predicted_class in ["up", "neutral", "down"], f"Invalid prediction label: {predicted_class}"
    
    # Check that it returns probabilities for all 3 classes
    assert len(probs) == 3
    assert "up" in probs and "neutral" in probs and "down" in probs
    
    # Check softmax functionality -> probabilities should sum to ~1.0
    sum_probs = sum(probs.values())
    assert abs(sum_probs - 1.0) < 1e-5, f"Probabilities do not sum to 1.0: {sum_probs}"

def test_cnn_classifier_not_enough_data():
    """Test that the classifier raises ValueError if there's insufficient data."""
    classifier = CnnPatternClassifier(window_size=20)
    
    # Create tiny dataframe
    tiny_df = pd.DataFrame({
        'open': [100]*10, 'high': [105]*10, 'low': [95]*10, 'close': [102]*10, 'volume': [1000]*10
    })
    
    with pytest.raises(ValueError, match="Not enough data"):
        classifier.fit(tiny_df)
    
    with pytest.raises(RuntimeError, match="Model is not trained yet"):
        classifier.predict(tiny_df)
