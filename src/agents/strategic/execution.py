from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Sequence
from uuid import uuid4

from src.agents.strategic.config import ExecutionConfig
from src.agents.strategic.impact_monitor import ImpactSlippageMonitor
from src.agents.strategic.risk_overseer import RiskOverseerDecision, RiskOverseerStateMachine, RiskSignalSnapshot
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

        direction = request.direction.strip().upper()
        if request.risk_mode == RiskMode.KILL_SWITCH:
            reasons.append("risk_mode_kill_switch_blocks_all_orders")
        elif request.risk_mode == RiskMode.CLOSE_ONLY and direction == "BUY":
            reasons.append("risk_mode_close_only_blocks_buy")
        elif request.risk_mode == RiskMode.REDUCE_ONLY and direction == "BUY":
            reasons.append("risk_mode_reduce_only_blocks_buy")

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
        impact_monitor: ImpactSlippageMonitor | None = None,
        risk_overseer: RiskOverseerStateMachine | None = None,
    ):
        self.config = config or ExecutionConfig()
        self.compliance = compliance or PreTradeComplianceChecker()
        self.slippage = SlippageModel()
        self.routing = RoutingHealthMonitor(failure_limit=self.config.routing_health_failure_limit)
        self.impact_monitor = impact_monitor
        self.risk_overseer = risk_overseer or RiskOverseerStateMachine()

    def plan_execution(
        self,
        request: OrderRequest,
        context: ExecutionContext,
        *,
        risk_snapshot: RiskSignalSnapshot | None = None,
        heartbeat_ok: bool = True,
        risk_authorizer: str = "execution_engine",
    ) -> ExecutionPlan:
        audit_events: list[dict[str, Any]] = []
        risk_budget_fraction = 1.0

        # 1. Risk Overseer evaluation (veto authority)
        risk_decision: RiskOverseerDecision | None = None
        effective_mode = request.risk_mode
        if self.risk_overseer is not None:
            snapshot = risk_snapshot or RiskSignalSnapshot(timestamp=context.timestamp)
            risk_decision = self.risk_overseer.evaluate(
                snapshot=snapshot,
                heartbeat_ok=heartbeat_ok,
                authorizer=risk_authorizer,
            )
            effective_mode = _more_restrictive(request.risk_mode, risk_decision.mode)
            risk_budget_fraction = _coerce_risk_budget_fraction(
                risk_decision.metadata.get("risk_budget_fraction"),
            )
            audit_events.append(
                {
                    "event": "risk_overseer_decision",
                    "mode": risk_decision.mode.value,
                    "trigger_layer": risk_decision.trigger_layer,
                    "trigger_reason": risk_decision.trigger_reason,
                    "block_new_orders": risk_decision.block_new_orders,
                    "should_cancel_orders": risk_decision.should_cancel_orders,
                    "fail_closed": risk_decision.fail_closed,
                    "event_id": risk_decision.event_id,
                    "risk_budget_fraction": risk_budget_fraction,
                    "metadata": risk_decision.metadata,
                }
            )

        request_for_check = replace(request, risk_mode=effective_mode)
        scaled_request = request_for_check
        if request.direction.strip().upper() == "BUY":
            scaled_quantity = int(math.floor(max(0, request.target_quantity) * risk_budget_fraction))
            scaled_request = replace(
                request_for_check,
                target_quantity=scaled_quantity,
                target_notional=float(request.target_notional) * risk_budget_fraction,
            )
            audit_events.append(
                {
                    "event": "risk_budget_scaling",
                    "symbol": request.symbol,
                    "direction": request.direction,
                    "requested_quantity": request.target_quantity,
                    "requested_notional": request.target_notional,
                    "risk_budget_fraction": risk_budget_fraction,
                    "scaled_quantity": scaled_quantity,
                    "scaled_notional": scaled_request.target_notional,
                }
            )

        # 2. Compliance Check
        compliance_result = self.compliance.check(scaled_request, context)
        audit_events.append({
            "event": "compliance_check",
            "passed": compliance_result.passed,
            "reasons": compliance_result.reasons,
            "risk_mode": compliance_result.risk_mode.value,
        })

        # 3. Slippage Estimation
        est_slippage = self.slippage.estimate_bps(context, scaled_request.target_quantity)
        effective_qty = int(scaled_request.target_quantity)
        if not compliance_result.passed:
            effective_qty = 0
        size_multiplier = 1.0
        if self.impact_monitor is not None:
            size_multiplier = self.impact_monitor.size_multiplier(request.symbol)
            effective_qty = int(math.floor(max(0, effective_qty) * size_multiplier))
        if not compliance_result.passed:
            effective_qty = 0

        # 4. Instruction Generation
        instruction = OrderInstruction(
            event_id=str(uuid4()),
            timestamp=context.timestamp,
            symbol=request.symbol,
            direction=request.direction,
            quantity=effective_qty,
            order_type=request.order_type,
            participation_limit=float(self.compliance.limits.max_participation),
            risk_mode=compliance_result.risk_mode,
            rationale=(
                f"strategic_exec_v1; slippage={est_slippage:.1f}bps; size_multiplier={size_multiplier:.2f}; "
                f"risk_mode={compliance_result.risk_mode.value}"
            ),
            metadata=request.metadata,
        )
        audit_events.append(
            {
                "event": "impact_size_multiplier",
                "symbol": request.symbol,
                "size_multiplier": size_multiplier,
                "effective_quantity": effective_qty,
            }
        )
        if risk_decision is not None and risk_decision.should_cancel_orders:
            audit_events.append(
                {
                    "event": "risk_cancel_orders",
                    "symbol": request.symbol,
                    "reason": risk_decision.trigger_reason,
                    "risk_mode": risk_decision.mode.value,
                }
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


def _more_restrictive(a: RiskMode, b: RiskMode) -> RiskMode:
    order = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return a if order[a] >= order[b] else b


def _coerce_risk_budget_fraction(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, fraction))
