from __future__ import annotations

from datetime import UTC, datetime
from typing import Mapping
from uuid import uuid4

from src.agents.strategic.config import ExecutionConfig
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


class ExecutionPlanner:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

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
        )

        now = datetime.now(UTC)
        quantity = round_to_tradable_quantity(portfolio_check.adjusted_quantity, lot_size=lot_size)
        direction = _direction_from_decision(intent.decision.action)
        order_type = _choose_order_type(intent.decision.action, instrument_state, self.config.default_order_type)
        if not compliance.passed or quantity == 0:
            direction = "SELL" if intent.decision.action in {ActionType.CLOSE, ActionType.REDUCE, ActionType.SELL} else "BUY"
        limit_price = market_price if order_type == OrderType.LIMIT else None
        stop_price = market_price * 0.985 if order_type in {OrderType.SL, OrderType.SL_M} else None
        participation_limit = portfolio_check.metadata.get("participation_limit", 0.0)

        instruction = OrderInstruction(
            event_id=f"ord-{uuid4().hex[:12]}",
            timestamp=now,
            symbol=intent.symbol,
            direction=direction,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            participation_limit=float(participation_limit),
            risk_mode=compliance.risk_mode,
            rationale=intent.decision.rationale,
            metadata={
                "hedge_required": intent.hedge_required,
                "hedge_symbol": intent.hedge_symbol,
                "portfolio_rejections": portfolio_check.rejection_reasons,
            },
        )

        estimated_slippage_bps = self._estimate_slippage_bps(
            quantity=quantity,
            market_price=market_price,
            participation_limit=float(participation_limit),
            instrument_state=instrument_state,
        )
        partial_fill_plan = self._partial_fill_schedule(quantity)
        routing_status = "healthy" if routing_is_healthy else "degraded"
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
            for event_type in ("ORDER_INTENT", "ORDER_SUBMITTED")
        )
        return ExecutionPlan(
            instruction=instruction,
            compliance=compliance,
            audit_events=audit_events,
            estimated_slippage_bps=estimated_slippage_bps,
            routing_status=routing_status,
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
    ) -> ComplianceCheckResult:
        reasons: list[str] = list(portfolio_check.rejection_reasons)
        warnings: list[str] = list(portfolio_check.warnings)
        risk_mode = portfolio_check.risk_mode

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
            risk_mode = RiskMode.CLOSE_ONLY if len(reasons) >= self.config.routing_health_failure_limit else RiskMode.REDUCE_ONLY
            reasons.append("routing_health_degraded")
        return ComplianceCheckResult(
            passed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(dict.fromkeys(warnings)),
            risk_mode=risk_mode,
            metadata={
                "available_margin": available_margin,
                "required_margin": required_margin,
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


def _direction_from_decision(action) -> str:
    if str(action).endswith("SELL") or str(action).endswith("CLOSE") or str(action).endswith("REDUCE"):
        return "SELL"
    return "BUY"


def _choose_order_type(action, instrument_state: Mapping[str, object], default_order_type: str) -> OrderType:
    if bool(instrument_state.get("requires_stop_loss", False)):
        return OrderType.SL
    if str(action).endswith("CLOSE"):
        return OrderType.MARKET
    return OrderType(default_order_type)
