from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.agents.strategic.config import (
    OBSERVATION_SCHEMA_VERSION,
    STRATEGIC_EXEC_CONTRACT_VERSION,
    TEACHER_POLICY_TYPE,
    WEEK2_ACTION_EXPORT_VERSION,
)


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"


class LoopType(str, Enum):
    FAST = "fast"
    SLOW = "slow"
    OFFLINE = "offline"


class PolicyType(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class StrategicObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: datetime
    symbol: str
    snapshot_id: str
    technical_direction: str
    technical_confidence: float = Field(ge=0.0, le=1.0)
    price_forecast: float
    var_95: float
    es_95: float
    regime_state: str
    regime_transition_prob: float = Field(ge=0.0, le=1.0)
    sentiment_score: float | None = None
    sentiment_z_t: float | None = None
    consensus_direction: str
    consensus_confidence: float = Field(ge=0.0, le=1.0)
    crisis_mode: bool = False
    agent_divergence: bool = False
    orderbook_imbalance: float | None = None
    queue_pressure: float | None = None
    current_position: float = 0.0
    unrealized_pnl: float = 0.0
    notional_exposure: float = 0.0
    portfolio_features: dict[str, Any] | None = None
    quality_status: str = "pass"
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION

    @property
    def observation_vector(self) -> list[float]:
        # Mapping helpers
        tech_map = {"up": 1.0, "down": -1.0, "neutral": 0.0}
        regime_map = {"bull": 1.0, "bear": -1.0, "sideways": 0.0, "unknown": 0.0}
        cons_map = {"up": 1.0, "down": -1.0, "neutral": 0.0, "hold": 0.0, "buy": 1.0, "sell": -1.0}

        return [
            float(self.technical_confidence),
            float(tech_map.get(self.technical_direction.lower(), 0.0)),
            float(self.price_forecast),
            float(self.var_95),
            float(regime_map.get(self.regime_state.lower(), 0.0)),
            float(self.regime_transition_prob),
            float(self.sentiment_score or 0.0),
            float(self.sentiment_z_t or 0.0),
            float(cons_map.get(self.consensus_direction.lower(), 0.0)),
            float(self.consensus_confidence),
            1.0 if self.crisis_mode else 0.0,
            1.0 if self.agent_divergence else 0.0,
            float(self.orderbook_imbalance or 0.0),
            float(self.queue_pressure or 0.0),
            float(self.current_position),
            float(self.unrealized_pnl),
            float(self.notional_exposure),
            float(len(self.portfolio_features or {})),
        ]

    @field_validator("technical_direction")
    @classmethod
    def _normalize_technical_direction(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"up", "down", "neutral"}:
            return "neutral"
        return normalized

    @field_validator("consensus_direction")
    @classmethod
    def _normalize_consensus_direction(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"BUY", "SELL", "NEUTRAL", "HOLD"}:
            return "HOLD"
        return normalized

    @field_validator("quality_status")
    @classmethod
    def _normalize_quality_status(cls, value: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized in {"pass", "warn", "fail"} else "warn"


class StrategicToExecutiveContract(BaseModel):
    """
    Finalized contract between strategic (Phase 3) and strategic_executive.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: str = STRATEGIC_EXEC_CONTRACT_VERSION
    timestamp: datetime
    symbol: str
    policy_id: str
    policy_type: PolicyType = PolicyType.TEACHER
    loop_type: LoopType = LoopType.SLOW
    action: ActionType
    action_size: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    observation_id: int | None = None
    snapshot_id: str
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    risk_override: str | None = None
    decision_reason: str = ""
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_teacher_loop_exclusion(self):
        if self.policy_type == PolicyType.TEACHER and self.loop_type == LoopType.FAST:
            raise ValueError("Teacher policies are blocked from Fast Loop execution path.")
        return self


class Week2ActionSpaceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export_schema_version: str = WEEK2_ACTION_EXPORT_VERSION
    contract_version: str = STRATEGIC_EXEC_CONTRACT_VERSION
    timestamp: datetime
    symbol: str
    policy_id: str
    policy_type: str = TEACHER_POLICY_TYPE
    loop_type: str = "slow"
    action: ActionType
    action_size: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    observation_id: int | None = None
    snapshot_id: str
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    quality_status: str = "pass"
    decision_reason: str = ""

    @model_validator(mode="after")
    def _validate_loop_type(self):
        if self.policy_type == TEACHER_POLICY_TYPE and self.loop_type == "fast":
            raise ValueError("Week 2 export cannot contain teacher actions for Fast Loop.")
        return self


class StepResult(BaseModel):
    reward: float
    gross_return: float
    net_return: float
    transaction_cost: float
    slippage_cost: float
    position_before: float
    position_after: float
    portfolio_value: float
    done: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


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
    schema_version: str = "1.0"

    model_config = ConfigDict(extra="allow", frozen=True)


class EnsembleEvaluationResult(BaseModel):
    policy_ids: tuple[str, ...]
    equal_weight_action: float
    action_dispersion: float
    mean_confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


class RLPolicyRegistryEntry(BaseModel):
    policy_id: str
    algorithm: str
    version: str = "1.0"
    training_status: str = "foundation"
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_space: str = "continuous"
    checkpoint_status: str = "not_available"
    checkpoint_path: str | None = None
    offline_only: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", frozen=True)


class RLTrainingRunRecord(BaseModel):
    policy_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "planned"
    reward_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: str | None = None
    notes: str | None = None
    schema_version: str = "1.0"

    model_config = ConfigDict(extra="allow", frozen=True)
