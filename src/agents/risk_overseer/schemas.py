from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agents.strategic.schemas import ActionType, RiskMode


class CrisisState(str, Enum):
    NORMAL = "normal"
    FULL_CRISIS = "full_crisis"
    AGENT_DIVERGENCE = "agent_divergence"
    SLOW_CRASH = "slow_crash"
    FEED_FREEZE = "feed_freeze"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    OOD_WARNING = "ood_warning"
    OOD_ALIEN = "ood_alien"


class RiskTriggerLayer(str, Enum):
    L1_MODEL = "l1_model"
    L2_PORTFOLIO = "l2_portfolio"
    L3_BROKER = "l3_broker"
    L4_MANUAL = "l4_manual"


class RiskTriggerCode(str, Enum):
    STUDENT_DRIFT = "student_drift"
    TEACHER_DIVERGENCE = "teacher_divergence"
    MAX_DRAWDOWN_BREACH = "max_drawdown_breach"
    DAILY_LOSS_BREACH = "daily_loss_breach"
    CONCENTRATION_BREACH = "concentration_breach"
    BROKER_API_ERROR = "broker_api_error"
    MARGIN_CALL = "margin_call"
    REJECTION_RATE_BREACH = "rejection_rate_breach"
    MANUAL_KILL_SWITCH = "manual_kill_switch"
    RECOVERY_STEP = "recovery_step"
    CRISIS_ENTRY = "crisis_entry"
    CRISIS_TIMEOUT_CLEAR = "crisis_timeout_clear"
    AGENT_DIVERGENCE = "agent_divergence"
    RE_RISK_STEP = "re_risk_step"
    SLOW_CRASH = "slow_crash"
    FEED_FREEZE = "feed_freeze"
    NEGATIVE_SENTIMENT_SPIKE = "negative_sentiment_spike"
    SENTIMENT_PRICE_MISMATCH = "sentiment_price_mismatch"
    OOD_WARNING = "ood_warning"
    OOD_ALIEN = "ood_alien"
    OOD_HARD_LIMIT_BREACH = "ood_hard_limit_breach"
    RISK_CAP_CHANGE = "risk_cap_change"


class ModelRiskSnapshot(BaseModel):
    student_drift: float = Field(default=0.0, ge=0.0)
    teacher_student_divergence: float = Field(default=0.0, ge=0.0)
    model_paused: bool = False
    active_policy_id: str | None = None
    fallback_policy_id: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioRiskSnapshot(BaseModel):
    max_drawdown: float = Field(default=0.0, ge=0.0)
    daily_loss: float = Field(default=0.0, ge=0.0)
    concentration_breach: bool = False
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    model_config = ConfigDict(extra="forbid", frozen=True)


class BrokerRiskSnapshot(BaseModel):
    api_healthy: bool = True
    margin_call_active: bool = False
    rejection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latest_error: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class CrisisRiskSnapshot(BaseModel):
    realized_vol: float = Field(default=0.0, ge=0.0)
    baseline_vol: float = Field(default=1.0, gt=0.0)
    liquidity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    agent_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    major_agent_disagreement_count: int = Field(default=0, ge=0)
    drawdown_velocity: float = Field(default=0.0, ge=0.0)
    feed_staleness_seconds: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class SentimentRiskSnapshot(BaseModel):
    z_t: float | None = None
    sentiment_score: float | None = None
    price_return: float = 0.0

    model_config = ConfigDict(extra="forbid", frozen=True)


class OODRiskSnapshot(BaseModel):
    warning: bool = False
    alien: bool = False
    provenance_reliability: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class RecoveryRequest(BaseModel):
    requested: bool = False
    all_conditions_resolved: bool = False
    operator_acknowledged: bool = False
    clear_manual_override: bool = False
    confirmed_alignment: bool = False
    requested_by: str | None = None
    note: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskEvaluationInput(BaseModel):
    timestamp: datetime
    symbol: str | None = None
    model: ModelRiskSnapshot = Field(default_factory=ModelRiskSnapshot)
    portfolio: PortfolioRiskSnapshot = Field(default_factory=PortfolioRiskSnapshot)
    broker: BrokerRiskSnapshot = Field(default_factory=BrokerRiskSnapshot)
    crisis: CrisisRiskSnapshot = Field(default_factory=CrisisRiskSnapshot)
    sentiment: SentimentRiskSnapshot = Field(default_factory=SentimentRiskSnapshot)
    ood: OODRiskSnapshot = Field(default_factory=OODRiskSnapshot)
    recovery: RecoveryRequest = Field(default_factory=RecoveryRequest)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("timestamp")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class RiskTriggerEvent(BaseModel):
    event_id: str
    timestamp: datetime
    layer: RiskTriggerLayer
    trigger_code: RiskTriggerCode
    mode: RiskMode
    reason: str
    value: float | None = None
    threshold: float | None = None
    operator_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("timestamp")
    @classmethod
    def _normalize_event_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class RiskAssessment(BaseModel):
    timestamp: datetime
    mode: RiskMode
    previous_mode: RiskMode
    approved: bool
    crisis_state: CrisisState = CrisisState.NORMAL
    veto_reason: str | None = None
    trigger_events: tuple[RiskTriggerEvent, ...] = ()
    permitted_actions: tuple[ActionType, ...] = ()
    exposure_cap: float = Field(default=1.0, ge=0.0, le=1.0)
    crisis_weight_cap: float = Field(default=0.70, ge=0.0, le=1.0)
    rerisk_budget_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    neutral_hold_active: bool = False
    hedge_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_active: bool = False
    recovery_reason: str | None = None
    source_service: str = "independent_risk_overseer"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("timestamp")
    @classmethod
    def _normalize_assessment_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    def can_submit_order(self, action: ActionType) -> bool:
        return action in self.permitted_actions and self.approved
