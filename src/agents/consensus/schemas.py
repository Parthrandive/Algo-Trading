from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConsensusTransitionModel(str, Enum):
    LSTAR = "lstar"
    ESTAR = "estar"


class ConsensusRiskMode(str, Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    PROTECTIVE = "protective"


class ConsensusRegimeRiskLevel(str, Enum):
    FULL_RISK = "full_risk"
    REDUCED_RISK = "reduced_risk"
    NEUTRAL_CASH = "neutral_cash"


class AgentSignal(BaseModel):
    name: str
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_protective: bool = False

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConsensusInput(BaseModel):
    symbol: str = "UNKNOWN"
    technical: AgentSignal
    regime: AgentSignal
    sentiment: AgentSignal
    volatility: float = Field(ge=0.0)
    macro_differential: float = 0.0
    rbi_signal: float = 0.0
    sentiment_quantile: float = Field(ge=0.0, le=1.0)
    crisis_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment_is_stale: bool = False
    sentiment_is_missing: bool = False
    regime_ood_warning: bool = False
    regime_ood_alien: bool = False
    regime_risk_level: ConsensusRegimeRiskLevel = ConsensusRegimeRiskLevel.FULL_RISK
    daily_trend_bullish: bool | None = None
    atr_rank_20d: float = Field(default=0.0, ge=0.0, le=1.0)
    arima_dir_acc: float = 1.0
    cnn_dir_acc: float = 1.0
    generated_at_utc: datetime

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(UTC)


class ConsensusOutput(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    transition_score: float = Field(ge=0.0, le=1.0)
    transition_model: ConsensusTransitionModel
    risk_mode: ConsensusRiskMode
    divergence_score: float = Field(ge=0.0, le=1.0)
    crisis_weight: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float]
    generated_at_utc: datetime
    schema_version: str = "1.0"

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(UTC)
