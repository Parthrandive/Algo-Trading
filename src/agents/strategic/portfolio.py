from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

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
    if lot_size <= 0:
        return int(round(quantity))
    return int(int(quantity // lot_size) * lot_size)


@dataclass(frozen=True)
class MarginRequirement:
    symbol: str
    required_margin: float
    available_margin: float
    maintenance_margin: float
    is_sufficient: bool


class MarginRequirementCalculater:
    def __init__(self, leverage: float = 1.0):
        self.leverage = leverage

    def calculate(self, price: float, quantity: int, equity: float) -> MarginRequirement:
        notional = price * quantity
        required = notional / self.leverage
        # Simplified: available margin is equity - some buffer
        available = equity * 0.95 
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
        self.margin_calc = MarginRequirementCalculater(leverage=self.config.max_leverage)

    def validate_intent(
        self,
        intent: PortfolioIntent,
        equity: float,
        current_price: float,
        *,
        risk_mode: RiskMode = RiskMode.NORMAL,
    ) -> PortfolioCheckResult:
        """Validates a proposed intent against portfolio risk constraints."""
        reasons: list[str] = []
        
        # 1. Lot Size Rounding
        lot_size = intent.metadata.get("lot_size", 1)
        target_qty = round_to_tradable_quantity(intent.target_notional / current_price, lot_size)
        
        # 2. Risk Overseer Constraint: Max Exposure
        notional = target_qty * current_price
        max_notional = equity * self.config.max_exposure_per_symbol
        if notional > max_notional:
            reasons.append(f"exposure_limit_exceeded:{notional:,.0f}>{max_notional:,.0f}")
            target_qty = round_to_tradable_quantity(max_notional / current_price, lot_size)

        # 3. Margin Check
        margin = self.margin_calc.calculate(current_price, target_qty, equity)
        if not margin.is_sufficient:
            reasons.append(f"insufficient_margin:required={margin.required_margin:,.0f},available={margin.available_margin:,.0f}")
            target_qty = 0 # Reject if no margin

        # 4. Global Kill Switch check (indirectly through risk_mode)
        if risk_mode == RiskMode.CLOSE_ONLY and intent.action == ActionType.BUY:
            reasons.append("risk_mode_close_only_blocks_buy")
            target_qty = 0

        passed = len(reasons) == 0
        return PortfolioCheckResult(
            passed=passed,
            adjusted_target_quantity=target_qty,
            reasons=tuple(reasons),
            risk_mode=risk_mode,
            metadata={
                "equity": equity,
                "current_price": current_price,
                "margin_required": margin.required_margin,
            }
        )


# Alias for main branch compatibility
PortfolioManager = PortfolioOptimizer
