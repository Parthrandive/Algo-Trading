from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

class TechnicalPrediction(BaseModel):
    """
    Output contract for the Technical Agent's predictions.
    """
    symbol: str = Field(..., description="The stock symbol (e.g., TATASTEEL.NS)")
    timestamp: datetime = Field(..., description="Prediction generation timestamp")
    price_forecast: float = Field(..., description="Predicted price magnitude for the next horizon")
    direction: Literal["up", "down", "neutral"] = Field(..., description="Predicted price direction")
    volatility_estimate: float = Field(..., description="GARCH-based volatility estimate")
    var_95: float = Field(..., description="Value-at-Risk at 95% confidence level")
    var_99: float = Field(..., description="Value-at-Risk at 99% confidence level")
    es_95: float = Field(..., description="Expected Shortfall at 95% confidence level")
    es_99: float = Field(..., description="Expected Shortfall at 99% confidence level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0.0 to 1.0)")
    model_id: str = Field(..., description="Identifier for the model that generated this prediction")
    schema_version: str = Field(default="1.0", description="Schema version")
    
    model_config = {
        "protected_namespaces": ()
    }
