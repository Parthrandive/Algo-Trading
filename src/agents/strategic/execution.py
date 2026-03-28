from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.agents.risk_overseer.schemas import RiskAssessment
from src.agents.strategic.config import ExecutionConfig
from src.agents.strategic.impact_monitor import ImpactSlippageMonitor
from src.agents.strategic.risk_overseer import RiskOverseerDecision, RiskOverseerStateMachine, RiskSignalSnapshot
from src.agents.strategic.portfolio import round_to_tradable_quantity
from src.agents.strategic.schemas import (
    ActionType,
    ComplianceCheckResult,
    ExecutionPlan,
    OrderInstruction,
    OrderType,
    PortfolioCheckResult,
    PortfolioIntent,
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
        if context.avg_volume_1h <= 0:
            return 20.0
        pct_of_volume = (quantity / context.avg_volume_1h) * 100.0
        return 5.0 + 2.5 * pct_of_volume


class PreTradeComplianceChecker:
    """Mandatory pre-trade compliance gates for the week-3+ execution path."""

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
            },
        )


class ExecutionEngine:
    """
    Phase 3 Strategic Execution Engine with backward-compatible planner surface.

    `plan_execution()` supports the newer Week 3+ path.
    `plan_order()` preserves the Week 1/2 planning API used by risk-overseer tests.
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

    def plan_order(
        self,
        *,
        intent: PortfolioIntent,
        portfolio_check: PortfolioCheckResult,
        market_price: float,
        available_margin: float,
        required_margin: float,
        lot_size: int = 1,
        instrument_state: Mapping[str, object] | None = None,
        routing_is_healthy: bool = True,
        model_version: str = "student_candidate",
        universe_version: str = "unknown",
        signal_source: str = "strategic_ensemble",
        risk_assessment: RiskAssessment | None = None,
    ) -> ExecutionPlan:
        instrument_state = dict(instrument_state or {})
        compliance = self.run_pre_trade_checks(
            intent=intent,
            portfolio_check=portfolio_check,
            available_margin=available_margin,
            required_margin=required_margin,
            lot_size=lot_size,
            instrument_state=instrument_state,
            routing_is_healthy=routing_is_healthy,
            risk_assessment=risk_assessment,
        )

        now = datetime.now(UTC)
        quantity = round_to_tradable_quantity(portfolio_check.adjusted_quantity, lot_size=lot_size)
        direction = _direction_from_decision(intent.decision.action)
        order_type = _choose_order_type(intent.decision.action, instrument_state, self.config.default_order_type)
        if not compliance.passed or quantity == 0:
            direction = "SELL" if intent.decision.action in {ActionType.CLOSE, ActionType.REDUCE, ActionType.SELL} else "BUY"
        limit_price = market_price if order_type == OrderType.LIMIT else None
        stop_price = market_price * 0.985 if order_type in {OrderType.SL, OrderType.SL_M} else None
        participation_limit = float(portfolio_check.metadata.get("participation_limit", 0.0))

        instruction = OrderInstruction(
            event_id=f"ord-{uuid4().hex[:12]}",
            timestamp=now,
            symbol=intent.symbol,
            direction=direction,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            participation_limit=participation_limit,
            risk_mode=compliance.risk_mode,
            rationale=intent.decision.rationale,
            metadata={
                "hedge_required": intent.hedge_required,
                "hedge_symbol": intent.hedge_symbol,
                "portfolio_rejections": portfolio_check.rejection_reasons,
                "risk_overlay_state": risk_assessment.crisis_state.value if risk_assessment else None,
            },
        )

        estimated_slippage_bps = self._estimate_slippage_bps(
            quantity=quantity,
            market_price=market_price,
            participation_limit=participation_limit,
            instrument_state=instrument_state,
        )
        partial_fill_plan = self._partial_fill_schedule(quantity)
        audit_events = tuple(
            self._build_audit_event(
                event_type=event_type,
                timestamp=now,
                instruction=instruction,
                compliance=compliance,
                model_version=model_version,
                signal_source=signal_source,
                universe_version=universe_version,
                data_source_tags=tuple(instrument_state.get("data_source_tags", ("primary_api",))),
            )
            for event_type in ("ORDER_INTENT", "ORDER_SUBMITTED" if compliance.passed else "REJECTION")
        )

        return ExecutionPlan(
            instruction=instruction,
            compliance=compliance,
            audit_events=audit_events,
            estimated_slippage_bps=estimated_slippage_bps,
            routing_status="healthy" if routing_is_healthy else "degraded",
            partial_fill_plan=partial_fill_plan,
        )

    def run_pre_trade_checks(
        self,
        *,
        intent: PortfolioIntent,
        portfolio_check: PortfolioCheckResult,
        available_margin: float,
        required_margin: float,
        lot_size: int,
        instrument_state: Mapping[str, object],
        routing_is_healthy: bool,
        risk_assessment: RiskAssessment | None,
    ) -> ComplianceCheckResult:
        reasons: list[str] = list(portfolio_check.rejection_reasons)
        warnings: list[str] = list(portfolio_check.warnings)
        risk_mode = portfolio_check.risk_mode

        if risk_assessment is None:
            reasons.append("risk_overseer_unreachable")
            risk_mode = RiskMode.KILL_SWITCH
        else:
            risk_mode = _max_risk_mode(risk_mode, risk_assessment.mode)
            if not risk_assessment.approved:
                reasons.append(risk_assessment.veto_reason or "risk_overseer_veto")
            elif not risk_assessment.can_submit_order(intent.decision.action):
                reasons.append(f"risk_mode_blocked:{risk_assessment.mode.value}")

        if not portfolio_check.approved:
            reasons.append("portfolio_risk_rejected")
        if available_margin < required_margin:
            reasons.append("insufficient_margin")
        if lot_size > 1 and round_to_tradable_quantity(portfolio_check.adjusted_quantity, lot_size=lot_size) == 0:
            reasons.append("lot_size_non_compliant")
        if bool(instrument_state.get("is_halted", False)):
            reasons.append("circuit_breaker_halt")
        if bool(instrument_state.get("hit_circuit", False)):
            warnings.append("instrument_hit_circuit_reduce_participation")
        if not bool(instrument_state.get("order_type_allowed", True)):
            reasons.append("order_type_not_allowed")
        if not routing_is_healthy:
            routing_risk_mode = (
                RiskMode.CLOSE_ONLY if len(reasons) >= self.config.routing_health_failure_limit else RiskMode.REDUCE_ONLY
            )
            risk_mode = _max_risk_mode(risk_mode, routing_risk_mode)
            reasons.append("routing_health_degraded")
        return ComplianceCheckResult(
            passed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(dict.fromkeys(warnings)),
            risk_mode=risk_mode,
            metadata={
                "available_margin": available_margin,
                "required_margin": required_margin,
                "risk_source": risk_assessment.source_service if risk_assessment else "risk_overseer_unreachable",
            },
        )

    def _estimate_slippage_bps(
        self,
        *,
        quantity: int,
        market_price: float,
        participation_limit: float,
        instrument_state: Mapping[str, object],
    ) -> float:
        adv = float(instrument_state.get("avg_daily_volume", 0.0) or 0.0)
        liquidity_score = float(instrument_state.get("liquidity_score", 1.0) or 1.0)
        if quantity <= 0 or market_price <= 0.0:
            return 0.0
        participation = (quantity / adv) if adv > 0 else max(participation_limit, 0.01)
        return max(1.0, 10_000.0 * participation / max(liquidity_score, 0.1))

    def _partial_fill_schedule(self, quantity: int) -> tuple[int, ...]:
        if quantity <= 0:
            return ()
        slices = min(self.config.max_partial_fill_count, max(1, quantity))
        base = quantity // slices
        remainder = quantity % slices
        plan = []
        for index in range(slices):
            plan.append(base + (1 if index < remainder else 0))
        return tuple(part for part in plan if part > 0)

    def _build_audit_event(
        self,
        *,
        event_type: str,
        timestamp: datetime,
        instruction: OrderInstruction,
        compliance: ComplianceCheckResult,
        model_version: str,
        signal_source: str,
        universe_version: str,
        data_source_tags: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "event_id": f"audit-{uuid4().hex[:12]}",
            "event_type": event_type,
            "timestamp_utc": timestamp.isoformat(),
            "instrument": instruction.symbol,
            "direction": instruction.direction,
            "quantity": instruction.quantity,
            "price": instruction.limit_price or instruction.stop_price or 0.0,
            "order_type": instruction.order_type.value,
            "model_version": model_version,
            "signal_source": signal_source,
            "universe_version": universe_version,
            "plan_version": "v1.3.7",
            "pre_trade_checks_passed": compliance.passed,
            "rejection_reason": ";".join(compliance.reasons) if compliance.reasons else None,
            "data_source_tags": list(data_source_tags),
            "risk_mode": compliance.risk_mode.value.upper(),
        }


ExecutionPlanner = ExecutionEngine


def audit_events_to_dict(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"audit_trail": list(events)}


def _direction_from_decision(action: ActionType) -> str:
    if action in {ActionType.SELL, ActionType.CLOSE, ActionType.REDUCE}:
        return "SELL"
    return "BUY"


def _choose_order_type(action: ActionType, instrument_state: Mapping[str, object], default_order_type: str) -> OrderType:
    if bool(instrument_state.get("requires_stop_loss", False)):
        return OrderType.SL
    if action == ActionType.CLOSE:
        return OrderType.MARKET
    return OrderType(default_order_type)


def _max_risk_mode(left: RiskMode, right: RiskMode) -> RiskMode:
    ordering = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return left if ordering[left] >= ordering[right] else right


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
