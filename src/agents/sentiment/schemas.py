

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class SentimentLane(str, Enum):
    FAST = "fast"
    SLOW = "slow"
    DAILY_AGG = "daily_agg"


class CacheFreshnessState(str, Enum):
    MISS = "miss"
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    ERROR = "error"


class SentimentQualityStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class SentimentPrediction(BaseModel):
    source_id: str
    text_hash: str
    lane: SentimentLane
    label: SentimentLabel
    symbol: str | None = None
    source_type: str | None = None
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at_utc: datetime
    headline_timestamp_utc: datetime | None = None
    score_timestamp_utc: datetime | None = None
    ttl_seconds: int = Field(ge=1)
    model_name: str
    latency_ms: float | None = Field(default=None, ge=0.0)
    cache_hit: bool = False
    freshness_state: CacheFreshnessState = CacheFreshnessState.MISS
    downweighted: bool = False
    quality_status: SentimentQualityStatus = SentimentQualityStatus.PASS
    manipulation_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    manipulation_flags: tuple[str, ...] = ()
    reduced_risk_mode: bool = False
    fallback_mode: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("headline_timestamp_utc", "score_timestamp_utc")
    @classmethod
    def normalize_optional_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("optional timestamps must be timezone-aware when provided")
        return value.astimezone(UTC)


class SentimentCacheEntry(BaseModel):
    cache_key: str
    source_id: str
    text_hash: str
    lane: SentimentLane
    label: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at_utc: datetime
    ttl_seconds: int = Field(ge=1)
    model_name: str
    schema_version: str = "1.0"

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def expires_at_utc(self) -> datetime:
        return self.generated_at_utc + timedelta(seconds=self.ttl_seconds)


class DailySentimentAggregate(BaseModel):
    generated_at_utc: datetime
    symbol: str | None = None
    lane: SentimentLane = SentimentLane.DAILY_AGG
    sample_size: int = Field(ge=0)
    weighted_sentiment_score: float = Field(ge=-1.0, le=1.0)
    macro_adjustment: float = Field(ge=-1.0, le=1.0)
    z_t: float = Field(ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    quality_status: SentimentQualityStatus = SentimentQualityStatus.PASS
    provenance: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("generated_at_utc")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(UTC)


class NightlySentimentBatchResult(BaseModel):
    started_at_utc: datetime
    completed_at_utc: datetime
    lookback_hours: int = Field(ge=1)
    document_predictions: tuple[SentimentPrediction, ...] = ()
    symbol_aggregates: tuple[DailySentimentAggregate, ...] = ()
    market_aggregate: DailySentimentAggregate | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("started_at_utc", "completed_at_utc")
    @classmethod
    def normalize_batch_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("batch timestamps must be timezone-aware")
        return value.astimezone(UTC)
