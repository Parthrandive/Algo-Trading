from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RegimeState(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    SIDEWAYS = "Sideways"
    CRISIS = "Crisis"
    RBI_BAND_TRANSITION = "RBI-Band transition"
    ALIEN = "Alien"


class RiskLevel(str, Enum):
    FULL_RISK = "full_risk"
    REDUCED_RISK = "reduced_risk"
    NEUTRAL_CASH = "neutral_cash"


class RegimePrediction(BaseModel):
    symbol: str = Field(..., description="Instrument symbol (for example, RELIANCE.NS).")
    timestamp: datetime = Field(..., description="Prediction generation timestamp (UTC).")
    regime_state: RegimeState = Field(..., description="Current detected regime.")
    transition_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Likelihood of transitioning away from the current regime."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    risk_level: RiskLevel = Field(..., description="Staged risk action for downstream agents.")
    model_id: str = Field(..., description="Identifier of the model/logic used.")
    schema_version: str = Field(default="1.0", description="Schema version.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional diagnostics to help with debugging or validation.",
    )

