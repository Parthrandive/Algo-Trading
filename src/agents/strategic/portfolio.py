from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.agents.strategic.config import PortfolioConfig
from src.agents.strategic.schemas import (
    ActionType,
    PortfolioCheckResult,
    PortfolioIntent,
    RiskMode,
)

logger = logging.getLogger(__name__)


def round_to_tradable_quantity(quantity: float, lot_size: int = 1) -> int:
    """Rounds a fractional quantity to the nearest lot size."""
    if lot_size <= 1:
        return int(round(quantity))
    sign = 1 if quantity >= 0 else -1
    absolute = abs(float(quantity))
    rounded = int(absolute // lot_size) * lot_size
    return sign * rounded


@dataclass(frozen=True)
class MarginRequirement:
    symbol: str
    required_margin: float
    available_margin: float
    maintenance_margin: float
    is_sufficient: bool


class MarginRequirementCalculator:
    def __init__(self, leverage: float = 1.0):
        if leverage <= 0:
            raise ValueError("leverage must be > 0")
        self.leverage = float(leverage)

    def calculate(self, price: float, quantity: int, equity: float) -> MarginRequirement:
        notional = abs(price * quantity)
        required = notional / self.leverage
        # Simplified: available margin is equity - some buffer
        available = max(0.0, equity * 0.95)
        return MarginRequirement(
            symbol="unknown",
            required_margin=required,
            available_margin=available,
            maintenance_margin=required * 0.5,
            is_sufficient=available >= required,
        )


class PortfolioOptimizer:
    """
    Phase 3 Portfolio Optimization and Risk Adherence.
    Ensures all intents pass through Risk Overseer level constraints.
    """

    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()
        self.margin_calc = MarginRequirementCalculator(leverage=self.config.max_leverage)

    def validate_intent(
        self,
        intent: PortfolioIntent,
        equity: float,
        current_price: float,
        *,
        risk_mode: RiskMode = RiskMode.NORMAL,
        exposure_cap_fraction: float = 1.0,
    ) -> PortfolioCheckResult:
        """Validates a proposed intent against portfolio risk constraints."""
        reasons: list[str] = []
        warnings: list[str] = []
        if current_price <= 0.0:
            return PortfolioCheckResult(
                approved=False,
                risk_mode=risk_mode,
                adjusted_quantity=0.0,
                adjusted_notional=0.0,
                rejection_reasons=("invalid_current_price",),
                metadata={"equity": equity, "current_price": current_price},
            )

        # 1. Lot Size Rounding and base target quantity resolution.
        lot_size = intent.metadata.get("lot_size", 1)
        requested_qty = intent.target_quantity
        if requested_qty == 0.0:
            requested_qty = intent.target_notional / current_price
        target_qty = round_to_tradable_quantity(requested_qty, lot_size)

        # 2. Risk mode constraints.
        if risk_mode == RiskMode.KILL_SWITCH:
            reasons.append("risk_mode_kill_switch_blocks_all_orders")
            target_qty = 0
        elif risk_mode == RiskMode.CLOSE_ONLY and intent.decision.action == ActionType.BUY:
            reasons.append("risk_mode_close_only_blocks_buy")
            target_qty = 0
        elif risk_mode == RiskMode.REDUCE_ONLY and intent.decision.action == ActionType.BUY:
            reasons.append("risk_mode_reduce_only_blocks_buy")
            target_qty = 0

        # 3. Risk Overseer exposure cap constraints.
        cap_fraction = max(0.0, min(1.0, float(exposure_cap_fraction)))
        notional = abs(target_qty * current_price)
        max_notional = equity * self.config.max_exposure_per_symbol * cap_fraction
        if notional > max_notional:
            warnings.append(f"exposure_capped:{notional:,.0f}>{max_notional:,.0f}")
            target_qty = round_to_tradable_quantity(max_notional / current_price, lot_size)
            notional = abs(target_qty * current_price)

        # 4. Margin Check.
        margin = self.margin_calc.calculate(current_price, target_qty, equity)
        if not margin.is_sufficient:
            reasons.append(f"insufficient_margin:required={margin.required_margin:,.0f},available={margin.available_margin:,.0f}")
            target_qty = 0  # Reject when margin is insufficient.
            notional = 0.0

        approved = len(reasons) == 0 and target_qty >= 0
        return PortfolioCheckResult(
            approved=approved,
            adjusted_quantity=float(target_qty),
            adjusted_notional=float(notional),
            rejection_reasons=tuple(reasons),
            warnings=tuple(warnings),
            risk_mode=risk_mode,
            metadata={
                "equity": equity,
                "current_price": current_price,
                "margin_required": margin.required_margin,
                "exposure_cap_fraction": cap_fraction,
                "max_notional": max_notional,
            }
        )


# Backward-compatible aliases
PortfolioManager = PortfolioOptimizer
PortfolioConstructor = PortfolioOptimizer
