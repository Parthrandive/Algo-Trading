from __future__ import annotations

from datetime import datetime
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
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    quality_status: str = "pass"

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
