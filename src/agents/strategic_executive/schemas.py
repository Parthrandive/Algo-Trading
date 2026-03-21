from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PHASE3_OBSERVATION_SCHEMA_VERSION = "1.0"


class QualityStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class SourceType(str, Enum):
    INTERNAL_PIPELINE = "internal_pipeline"
    OFFICIAL_API = "official_api"
    DERIVED = "derived"


class RegimeProbabilities(BaseModel):
    bull: float = Field(ge=0.0, le=1.0)
    bear: float = Field(ge=0.0, le=1.0)
    sideways: float = Field(ge=0.0, le=1.0)
    crisis: float = Field(ge=0.0, le=1.0)
    rbi_band_transition: float = Field(ge=0.0, le=1.0)
    alien: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "RegimeProbabilities":
        total = self.bull + self.bear + self.sideways + self.crisis + self.rbi_band_transition + self.alien
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"regime probabilities must sum to ~1.0 (got {total:.6f})")
        return self


class TechnicalRiskSnapshot(BaseModel):
    volatility_estimate: float = Field(ge=0.0)
    var_95: float
    var_99: float
    es_95: float
    es_99: float

    model_config = ConfigDict(extra="forbid", frozen=True)


class MacroSnapshot(BaseModel):
    macro_differential: float = 0.0
    rbi_signal: float = 0.0

    model_config = ConfigDict(extra="forbid", frozen=True)


class SentimentSnapshot(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    freshness_state: str = "unknown"
    quality_status: QualityStatus = QualityStatus.PASS

    model_config = ConfigDict(extra="forbid", frozen=True)


class ConsensusSnapshot(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_mode: str = "normal"
    crisis_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class MicrostructureSnapshot(BaseModel):
    feed_is_healthy: bool = False
    order_book_imbalance: float | None = Field(default=None, ge=-1.0, le=1.0)
    queue_pressure: float | None = Field(default=None, ge=-1.0, le=1.0)
    quality_flags: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_fast_loop_fields(self) -> "MicrostructureSnapshot":
        if self.feed_is_healthy:
            if self.order_book_imbalance is None or self.queue_pressure is None:
                raise ValueError(
                    "healthy microstructure feed must include order_book_imbalance and queue_pressure"
                )
        return self


class Phase3Observation(BaseModel):
    snapshot_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    generated_at: datetime
    expires_at: datetime
    schema_version: str = Field(default=PHASE3_OBSERVATION_SCHEMA_VERSION)
    quality_status: QualityStatus = QualityStatus.PASS
    source_type: SourceType = SourceType.INTERNAL_PIPELINE
    technical: TechnicalRiskSnapshot
    regime_probabilities: RegimeProbabilities
    macro: MacroSnapshot
    sentiment: SentimentSnapshot
    consensus: ConsensusSnapshot
    microstructure: MicrostructureSnapshot

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at", "expires_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != PHASE3_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={value!r}; expected {PHASE3_OBSERVATION_SCHEMA_VERSION!r}"
            )
        return value

    @model_validator(mode="after")
    def validate_expiry(self) -> "Phase3Observation":
        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be later than generated_at")
        return self
