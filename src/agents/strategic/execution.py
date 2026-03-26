from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.agents.strategic.config import ExecutionConfig
from src.agents.strategic.schemas import (
    ComplianceCheckResult,
    ExecutionPlan,
    OrderInstruction,
    OrderType,
    RiskMode,
)


@dataclass(frozen=True)
class PreTradeLimits:
    max_notional: float = 100_000_000.0
    max_participation: float = 0.05
    min_confidence: float = 0.20


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: str
    target_quantity: int
    target_notional: float
    confidence: float
    order_type: OrderType = OrderType.LIMIT
    risk_mode: RiskMode = RiskMode.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionContext:
    timestamp: datetime
    symbol: str
    current_price: float
    orderbook_imbalance: float
    queue_pressure: float
    avg_volume_1h: float


class RoutingHealthMonitor:
    def __init__(self, failure_limit: int = 3):
        self.failure_limit = failure_limit
        self.failures = 0

    def mark_failure(self) -> None:
        self.failures += 1

    def is_healthy(self) -> bool:
        return self.failures < self.failure_limit


class SlippageModel:
    def estimate_bps(self, context: ExecutionContext, quantity: int) -> float:
        # Simple impact-based slippage model
        if context.avg_volume_1h <= 0:
            return 20.0
        pct_of_volume = (quantity / context.avg_volume_1h) * 100.0
        return 5.0 + 2.5 * pct_of_volume


class PreTradeComplianceChecker:
    """Mandatory pre-trade compliance gates."""

    def __init__(self, limits: PreTradeLimits | None = None):
        self.limits = limits or PreTradeLimits()

    def check(self, request: OrderRequest, context: ExecutionContext) -> ComplianceCheckResult:
        reasons: list[str] = []
        
        if request.target_notional > self.limits.max_notional:
            reasons.append(f"notional_limit_exceeded:{request.target_notional:,.0f}>{self.limits.max_notional:,.0f}")

        if request.confidence < self.limits.min_confidence:
            reasons.append(f"confidence_floor_breach:{request.confidence:.2f}<{self.limits.min_confidence:.2f}")

        participation = request.target_quantity / max(context.avg_volume_1h, 1.0)
        if participation > self.limits.max_participation:
            reasons.append(f"participation_cap_breach:{participation:.2%}>{self.limits.max_participation:.2%}")

        return ComplianceCheckResult(
            passed=len(reasons) == 0,
            reasons=tuple(reasons),
            risk_mode=request.risk_mode,
            metadata={
                "max_notional": self.limits.max_notional,
                "current_participation": participation,
            }
        )


class ExecutionEngine:
    """
    Phase 3 Strategic Execution Engine.
    Enforces compliance and records mandatory audit trails.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        compliance: PreTradeComplianceChecker | None = None,
    ):
        self.config = config or ExecutionConfig()
        self.compliance = compliance or PreTradeComplianceChecker()
        self.slippage = SlippageModel()
        self.routing = RoutingHealthMonitor(failure_limit=self.config.routing_health_failure_limit)

    def plan_execution(
        self,
        request: OrderRequest,
        context: ExecutionContext,
    ) -> ExecutionPlan:
        audit_events: list[dict[str, Any]] = []
        
        # 1. Compliance Check
        compliance_result = self.compliance.check(request, context)
        audit_events.append({
            "event": "compliance_check",
            "passed": compliance_result.passed,
            "reasons": compliance_result.reasons,
        })

        # 2. Slippage Estimation
        est_slippage = self.slippage.estimate_bps(context, request.target_quantity)
        
        # 3. Instruction Generation
        instruction = OrderInstruction(
            event_id=str(uuid4()),
            timestamp=context.timestamp,
            symbol=request.symbol,
            direction=request.direction,
            quantity=request.target_quantity,
            order_type=request.order_type,
            participation_limit=float(self.compliance.limits.max_participation),
            risk_mode=compliance_result.risk_mode,
            rationale=f"strategic_exec_v1; slippage={est_slippage:.1f}bps",
            metadata=request.metadata,
        )

        return ExecutionPlan(
            instruction=instruction,
            compliance=compliance_result,
            audit_events=tuple(audit_events),
            estimated_slippage_bps=est_slippage,
            routing_status="healthy" if self.routing.is_healthy() else "degraded",
        )


# Alias for main branch compatibility
ExecutionPlanner = ExecutionEngine


def audit_events_to_dict(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"audit_trail": list(events)}
