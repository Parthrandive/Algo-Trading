from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import math
from typing import Iterable
from uuid import uuid4

from src.agents.strategic.policy_manager import RiskMode
from src.agents.strategic.schemas import ActionType


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "sl"
    STOP_LOSS_MARKET = "sl-m"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class AuditEventType(str, Enum):
    ORDER_INTENT = "ORDER_INTENT"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCELLATION = "CANCELLATION"
    REJECTION = "REJECTION"


@dataclass(frozen=True)
class ExecutionContext:
    model_version: str
    signal_source: str
    universe_version: str
    plan_version: str
    data_source_tags: tuple[str, ...] = ("primary_api",)
    risk_mode: RiskMode = RiskMode.NORMAL


@dataclass(frozen=True)
class PreTradeLimits:
    max_position_qty: float
    max_sector_exposure_notional: float
    max_gross_exposure_notional: float
    capacity_ceiling_notional: float
    margin_limit_pct: float = 1.0
    max_participation_rate: float = 0.10


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: ActionType
    quantity: int
    order_type: OrderType
    price: float | None = None
    trigger_price: float | None = None
    segment: str = "nse_equity"
    current_position_qty: float = 0.0
    current_sector_exposure_notional: float = 0.0
    current_gross_exposure_notional: float = 0.0
    order_notional: float | None = None
    available_margin: float = 0.0
    required_margin: float = 0.0
    margin_utilization_post: float = 0.0
    lot_size: int = 1
    instrument_halted: bool = False
    instrument_circuit_hit: bool = False
    market_wide_halt: bool = False
    leverage_ok: bool = True
    shorting_allowed: bool = True
    adv_quantity: float = 0.0
    participation_limit_rate: float | None = None
    expected_slippage_bps: float = 0.0
    reference_price: float | None = None
    liquidity_fill_ratio: float = 1.0


@dataclass(frozen=True)
class ComplianceResult:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OrderFill:
    fill_timestamp: datetime
    fill_price: float
    fill_quantity: int
    fees: float = 0.0
    impact_cost_bps: float = 0.0


@dataclass
class OrderRecord:
    order_id: str
    request: OrderRequest
    status: OrderStatus
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    rejection_reason: str | None = None
    fills: list[OrderFill] = field(default_factory=list)

    @property
    def filled_quantity(self) -> int:
        return int(sum(fill.fill_quantity for fill in self.fills))

    @property
    def avg_fill_price(self) -> float | None:
        qty = self.filled_quantity
        if qty <= 0:
            return None
        total = sum(fill.fill_quantity * fill.fill_price for fill in self.fills)
        return float(total / qty)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    timestamp_utc: datetime
    instrument: str
    direction: str
    quantity: int
    price: float
    order_type: str
    model_version: str
    signal_source: str
    universe_version: str
    plan_version: str
    pre_trade_checks_passed: bool
    rejection_reason: str | None
    data_source_tags: tuple[str, ...]
    risk_mode: str


class RoutingHealthMonitor:
    def __init__(self, *, rejection_alert_threshold: float = 0.20):
        self.rejection_alert_threshold = float(rejection_alert_threshold)
        self.infrastructure_ok = True
        self.broker_ok = True
        self.rejection_rate = 0.0

    def update(
        self,
        *,
        infrastructure_ok: bool,
        broker_ok: bool,
        rejection_rate: float,
    ) -> RiskMode:
        self.infrastructure_ok = bool(infrastructure_ok)
        self.broker_ok = bool(broker_ok)
        self.rejection_rate = float(max(0.0, rejection_rate))
        return self.recommended_mode()

    def recommended_mode(self) -> RiskMode:
        if not self.broker_ok or self.rejection_rate > self.rejection_alert_threshold:
            return RiskMode.CLOSE_ONLY
        if not self.infrastructure_ok:
            return RiskMode.REDUCE_ONLY
        return RiskMode.NORMAL


class PreTradeComplianceChecker:
    def __init__(self, *, limits: PreTradeLimits):
        self.limits = limits
        self._allowed_order_types = {
            "nse_equity": {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET},
            "nse_futures": {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS},
            "nse_currency": {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS},
            "mcx_commodity": {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS},
        }

    def evaluate(self, request: OrderRequest, *, mode: RiskMode) -> ComplianceResult:
        reasons: list[str] = []

        if request.quantity <= 0:
            reasons.append("quantity_must_be_positive")

        signed_qty = _signed_quantity(
            request.direction,
            request.quantity,
            current_position_qty=request.current_position_qty,
        )
        projected_position = request.current_position_qty + signed_qty
        current_abs = abs(float(request.current_position_qty))
        projected_abs = abs(float(projected_position))

        if mode in {RiskMode.CLOSE_ONLY, RiskMode.KILL_SWITCH}:
            if projected_abs >= (current_abs - 1e-9):
                reasons.append(f"mode_{mode.value}_blocks_non_closing_orders")

        if mode == RiskMode.REDUCE_ONLY and projected_abs > (current_abs + 1e-9):
            reasons.append("reduce_only_blocks_risk_increase")

        if abs(projected_position) > self.limits.max_position_qty:
            reasons.append("position_limit_exceeded")

        order_notional = self._order_notional(request)
        if request.current_sector_exposure_notional + order_notional > self.limits.max_sector_exposure_notional:
            reasons.append("sector_concentration_exceeded")

        if request.current_gross_exposure_notional + order_notional > self.limits.max_gross_exposure_notional:
            reasons.append("gross_exposure_limit_exceeded")

        if order_notional > self.limits.capacity_ceiling_notional:
            reasons.append("capacity_ceiling_exceeded")

        if request.available_margin < request.required_margin:
            reasons.append("insufficient_margin")

        if request.margin_utilization_post > self.limits.margin_limit_pct:
            reasons.append("margin_utilization_exceeded")

        if request.lot_size > 1 and request.quantity % request.lot_size != 0:
            reasons.append("lot_size_violation")

        if request.market_wide_halt:
            reasons.append("market_wide_halt")
        if request.instrument_halted:
            reasons.append("instrument_halt")
        if request.instrument_circuit_hit:
            reasons.append("instrument_circuit_hit")

        allowed = self._allowed_order_types.get(request.segment.strip().lower(), set())
        if request.order_type not in allowed:
            reasons.append("order_type_not_allowed_for_segment")

        if request.direction == ActionType.SELL and not request.shorting_allowed and request.current_position_qty <= 0.0:
            reasons.append("shorting_not_allowed")

        if not request.leverage_ok:
            reasons.append("leverage_limit_exceeded")

        if request.adv_quantity > 0.0:
            participation_limit = request.participation_limit_rate
            if participation_limit is None:
                participation_limit = self.limits.max_participation_rate
            if request.quantity > request.adv_quantity * float(participation_limit):
                reasons.append("participation_limit_exceeded")

        return ComplianceResult(passed=len(reasons) == 0, reasons=tuple(reasons))

    @staticmethod
    def _order_notional(request: OrderRequest) -> float:
        if request.order_notional is not None:
            return float(max(0.0, request.order_notional))
        reference = request.price if request.price is not None else (request.reference_price or 0.0)
        return float(max(0.0, request.quantity * reference))


class SlippageModel:
    def estimate_bps(self, *, request: OrderRequest) -> float:
        participation = 0.0
        if request.adv_quantity > 0.0:
            participation = request.quantity / request.adv_quantity
        base = max(0.0, float(request.expected_slippage_bps))
        impact = 100.0 * participation
        return float(base + impact)

    def realized_bps(
        self,
        *,
        direction: ActionType,
        reference_price: float,
        fill_price: float,
    ) -> float:
        if reference_price <= 0.0:
            return 0.0
        if direction == ActionType.BUY:
            return float(((fill_price - reference_price) / reference_price) * 10_000.0)
        if direction == ActionType.SELL:
            return float(((reference_price - fill_price) / reference_price) * 10_000.0)
        return 0.0


class ExecutionEngine:
    """
    Week 2 Day 6 execution engine with compliance gate and audit trail.

    The engine is intentionally DB-agnostic and in-memory only.
    """

    def __init__(
        self,
        *,
        checker: PreTradeComplianceChecker,
        slippage_model: SlippageModel | None = None,
        health_monitor: RoutingHealthMonitor | None = None,
        mode: RiskMode = RiskMode.NORMAL,
        max_fill_slices: int = 4,
    ):
        self.checker = checker
        self.slippage_model = slippage_model or SlippageModel()
        self.health_monitor = health_monitor or RoutingHealthMonitor()
        self.mode = mode
        self.max_fill_slices = max(1, int(max_fill_slices))

    def update_health(
        self,
        *,
        infrastructure_ok: bool,
        broker_ok: bool,
        rejection_rate: float,
    ) -> RiskMode:
        suggested = self.health_monitor.update(
            infrastructure_ok=infrastructure_ok,
            broker_ok=broker_ok,
            rejection_rate=rejection_rate,
        )
        self.mode = _more_restrictive(self.mode, suggested)
        return self.mode

    def route(
        self,
        request: OrderRequest,
        *,
        context: ExecutionContext,
        now: datetime | None = None,
    ) -> tuple[OrderRecord, list[AuditEvent]]:
        ts = (now or datetime.now(UTC)).astimezone(UTC)
        active_mode = _more_restrictive(self.mode, context.risk_mode)
        compliance = self.checker.evaluate(request, mode=active_mode)

        order_id = f"ord_{uuid4().hex[:16]}"
        events: list[AuditEvent] = []

        events.append(
            self._audit_event(
                event_type=AuditEventType.ORDER_INTENT,
                timestamp=ts,
                request=request,
                context=context,
                mode=active_mode,
                pre_trade_checks_passed=compliance.passed,
                rejection_reason=";".join(compliance.reasons) if not compliance.passed else None,
                quantity=request.quantity,
                price=request.price or request.reference_price or 0.0,
            )
        )

        if not compliance.passed:
            rejected = OrderRecord(
                order_id=order_id,
                request=request,
                status=OrderStatus.REJECTED,
                submitted_at=ts,
                rejection_reason=";".join(compliance.reasons),
            )
            events.append(
                self._audit_event(
                    event_type=AuditEventType.REJECTION,
                    timestamp=ts,
                    request=request,
                    context=context,
                    mode=active_mode,
                    pre_trade_checks_passed=False,
                    rejection_reason=rejected.rejection_reason,
                    quantity=request.quantity,
                    price=request.price or request.reference_price or 0.0,
                )
            )
            return rejected, events

        order = OrderRecord(order_id=order_id, request=request, status=OrderStatus.SUBMITTED, submitted_at=ts)
        events.append(
            self._audit_event(
                event_type=AuditEventType.ORDER_SUBMITTED,
                timestamp=ts,
                request=request,
                context=context,
                mode=active_mode,
                pre_trade_checks_passed=True,
                rejection_reason=None,
                quantity=request.quantity,
                price=request.price or request.reference_price or 0.0,
            )
        )

        fills = self._simulate_fills(request=request, timestamp=ts)
        order.fills.extend(fills)

        if order.filled_quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "no_liquidity_fill"
            events.append(
                self._audit_event(
                    event_type=AuditEventType.REJECTION,
                    timestamp=ts,
                    request=request,
                    context=context,
                    mode=active_mode,
                    pre_trade_checks_passed=True,
                    rejection_reason=order.rejection_reason,
                    quantity=request.quantity,
                    price=request.price or request.reference_price or 0.0,
                )
            )
            return order, events

        order.filled_at = fills[-1].fill_timestamp
        order.status = OrderStatus.FILLED if order.filled_quantity >= request.quantity else OrderStatus.PARTIAL

        for idx, fill in enumerate(fills):
            event_type = AuditEventType.FILL if idx == len(fills) - 1 and order.status == OrderStatus.FILLED else AuditEventType.PARTIAL_FILL
            events.append(
                self._audit_event(
                    event_type=event_type,
                    timestamp=fill.fill_timestamp,
                    request=request,
                    context=context,
                    mode=active_mode,
                    pre_trade_checks_passed=True,
                    rejection_reason=None,
                    quantity=fill.fill_quantity,
                    price=fill.fill_price,
                )
            )

        return order, events

    def cancel(
        self,
        order: OrderRecord,
        *,
        context: ExecutionContext,
        reason: str,
        now: datetime | None = None,
    ) -> AuditEvent:
        ts = (now or datetime.now(UTC)).astimezone(UTC)
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = ts
        order.rejection_reason = reason
        return self._audit_event(
            event_type=AuditEventType.CANCELLATION,
            timestamp=ts,
            request=order.request,
            context=context,
            mode=_more_restrictive(self.mode, context.risk_mode),
            pre_trade_checks_passed=True,
            rejection_reason=reason,
            quantity=max(0, order.request.quantity - order.filled_quantity),
            price=order.request.price or order.request.reference_price or 0.0,
        )

    def _simulate_fills(self, *, request: OrderRequest, timestamp: datetime) -> list[OrderFill]:
        fill_ratio = min(max(float(request.liquidity_fill_ratio), 0.0), 1.0)
        target_fill_qty = int(math.floor(request.quantity * fill_ratio))
        if target_fill_qty <= 0:
            return []

        slices = min(self.max_fill_slices, max(1, target_fill_qty))
        base_slice = max(1, target_fill_qty // slices)
        remaining = target_fill_qty

        ref_price = request.reference_price or request.price or 0.0
        if ref_price <= 0.0:
            ref_price = request.price or 0.0

        fills: list[OrderFill] = []
        for idx in range(slices):
            if remaining <= 0:
                break
            qty = remaining if idx == slices - 1 else min(base_slice, remaining)
            est_slippage_bps = self.slippage_model.estimate_bps(request=request)
            direction_mult = 1.0 if request.direction == ActionType.BUY else -1.0
            fill_price = ref_price * (1.0 + direction_mult * (est_slippage_bps / 10_000.0))
            impact_bps = self.slippage_model.realized_bps(
                direction=request.direction,
                reference_price=ref_price,
                fill_price=fill_price,
            )
            fills.append(
                OrderFill(
                    fill_timestamp=timestamp,
                    fill_price=float(fill_price),
                    fill_quantity=int(qty),
                    impact_cost_bps=float(impact_bps),
                )
            )
            remaining -= qty

        return fills

    def _audit_event(
        self,
        *,
        event_type: AuditEventType,
        timestamp: datetime,
        request: OrderRequest,
        context: ExecutionContext,
        mode: RiskMode,
        pre_trade_checks_passed: bool,
        rejection_reason: str | None,
        quantity: int,
        price: float,
    ) -> AuditEvent:
        return AuditEvent(
            event_id=f"evt_{uuid4().hex}",
            event_type=event_type,
            timestamp_utc=timestamp.astimezone(UTC),
            instrument=request.symbol,
            direction=_direction_label(request.direction),
            quantity=int(quantity),
            price=float(price),
            order_type=request.order_type.value.upper(),
            model_version=context.model_version,
            signal_source=context.signal_source,
            universe_version=context.universe_version,
            plan_version=context.plan_version,
            pre_trade_checks_passed=bool(pre_trade_checks_passed),
            rejection_reason=rejection_reason,
            data_source_tags=tuple(context.data_source_tags),
            risk_mode=mode.value.upper(),
        )


def _signed_quantity(
    direction: ActionType,
    quantity: int,
    *,
    current_position_qty: float = 0.0,
) -> float:
    if direction == ActionType.BUY:
        return float(quantity)
    if direction == ActionType.SELL:
        return float(-quantity)
    if direction in {ActionType.CLOSE, ActionType.REDUCE}:
        if current_position_qty > 0.0:
            return float(-quantity)
        if current_position_qty < 0.0:
            return float(quantity)
        return 0.0
    return 0.0


def _more_restrictive(a: RiskMode, b: RiskMode) -> RiskMode:
    order = {
        RiskMode.NORMAL: 0,
        RiskMode.REDUCE_ONLY: 1,
        RiskMode.CLOSE_ONLY: 2,
        RiskMode.KILL_SWITCH: 3,
    }
    return a if order[a] >= order[b] else b


def _direction_label(action: ActionType) -> str:
    if action == ActionType.BUY:
        return "BUY"
    if action == ActionType.SELL:
        return "SELL"
    return action.value.upper()


def audit_events_to_dict(events: Iterable[AuditEvent]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        rows.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp_utc": event.timestamp_utc.isoformat(),
                "instrument": event.instrument,
                "direction": event.direction,
                "quantity": event.quantity,
                "price": event.price,
                "order_type": event.order_type,
                "model_version": event.model_version,
                "signal_source": event.signal_source,
                "universe_version": event.universe_version,
                "plan_version": event.plan_version,
                "pre_trade_checks_passed": event.pre_trade_checks_passed,
                "rejection_reason": event.rejection_reason,
                "data_source_tags": list(event.data_source_tags),
                "risk_mode": event.risk_mode,
            }
        )
    return rows
