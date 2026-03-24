from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agents.strategic.config import OBSERVATION_MAPPING_VERSION, OBSERVATION_SCHEMA_VERSION

ActionSpace = Literal["continuous", "discrete"]

OBSERVATION_FEATURE_NAMES = (
    "price_forecast",
    "direction_score",
    "var_95",
    "es_95",
    "regime_bull",
    "regime_bear",
    "regime_sideways",
    "regime_crisis",
    "regime_rbi_band_transition",
    "regime_alien",
    "transition_probability",
    "sentiment_score",
    "z_t",
    "final_direction_score",
    "final_confidence",
    "crisis_mode_flag",
    "current_position",
    "unrealized_pnl",
)


class StrategicObservation(BaseModel):
    symbol: str
    timestamp: datetime
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    mapping_version: str = OBSERVATION_MAPPING_VERSION
    observation_vector: tuple[float, ...] = Field(
        min_length=len(OBSERVATION_FEATURE_NAMES),
        max_length=len(OBSERVATION_FEATURE_NAMES),
    )
    feature_names: tuple[str, ...] = OBSERVATION_FEATURE_NAMES
    technical_model_id: str | None = None
    regime_model_id: str | None = None
    sentiment_model_id: str | None = None
    consensus_model_id: str | None = None
    alignment_tolerance_seconds: float = Field(ge=0.0, default=300.0)
    source_timestamps: dict[str, datetime] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class RewardLog(BaseModel):
    symbol: str
    timestamp: datetime
    episode_id: str
    reward_name: str
    reward_value: float
    portfolio_value: float | None = None
    gross_return: float | None = None
    net_return: float | None = None
    transaction_cost: float | None = None
    slippage_cost: float | None = None
    action: float | None = None
    position_before: float | None = None
    position_after: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class RLPolicyRegistryEntry(BaseModel):
    policy_id: str
    algorithm: str
    version: str = "1.0"
    stage: str = "foundation"
    training_status: str = "not_started"
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_space: ActionSpace = "continuous"
    checkpoint_path: str | None = None
    checkpoint_status: str = "not_available"
    is_teacher_policy: bool = True
    offline_only: bool = True
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(extra="forbid", frozen=True)


class RLTrainingRunRecord(BaseModel):
    policy_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "planned"
    split_label: str = "walk_forward"
    train_start: datetime | None = None
    train_end: datetime | None = None
    validation_start: datetime | None = None
    validation_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    reward_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: str | None = None
    notes: str | None = None
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator(
        "started_at",
        "completed_at",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    )
    @classmethod
    def normalize_optional_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)


class StepResult(BaseModel):
    reward: float
    gross_return: float
    net_return: float
    transaction_cost: float
    slippage_cost: float
    position_before: float
    position_after: float
    portfolio_value: float
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnsembleEvaluationResult(BaseModel):
    policy_ids: tuple[str, ...]
    equal_weight_action: float
    action_dispersion: float = Field(ge=0.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)
