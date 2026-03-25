from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping

from src.agents.strategic.config import PortfolioConfig
from src.agents.strategic.schemas import (
    ActionType,
    EnsembleDecision,
    PortfolioCheckResult,
    PortfolioIntent,
    RiskMode,
)


@dataclass(frozen=True)
class PortfolioState:
    nav: float
    gross_exposure: float
    net_exposure: float
    current_positions: Mapping[str, float]
    sector_exposures: Mapping[str, float]
    correlations: Mapping[tuple[str, str], float]
    capacity_validated_for_scale: bool = False


class PortfolioConstructor:
    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def build_intent(
        self,
        *,
        symbol: str,
        decision: EnsembleDecision,
        mark_price: float,
        forecast_uncertainty: float,
    ) -> PortfolioIntent:
        if mark_price <= 0.0:
            raise ValueError("mark_price must be positive")
        sizing_scalar = self._uncertainty_scalar(forecast_uncertainty)
        target_notional = decision.action_size * sizing_scalar * mark_price * 100.0
        if decision.action in {ActionType.REDUCE, ActionType.CLOSE}:
            target_notional *= 0.5
        target_quantity = target_notional / mark_price
        hedge_required = decision.risk_mode in {RiskMode.REDUCE_ONLY, RiskMode.CLOSE_ONLY}
        return PortfolioIntent(
            symbol=symbol,
            decision=decision,
            target_notional=target_notional,
            target_quantity=target_quantity,
            hedge_required=hedge_required,
            hedge_symbol=self._default_hedge_symbol(symbol) if hedge_required else None,
            metadata={"forecast_uncertainty": forecast_uncertainty, "mark_price": mark_price},
        )

    def validate_intent(
        self,
        *,
        intent: PortfolioIntent,
        state: PortfolioState,
        sector: str,
        avg_daily_volume: float,
    ) -> PortfolioCheckResult:
        reasons: list[str] = []
        warnings: list[str] = []
        hedge_actions: list[str] = []
        risk_mode = intent.decision.risk_mode

        capacity_cap = (
            self.config.validated_notional_cap if state.capacity_validated_for_scale else self.config.normal_notional_cap
        )
        adjusted_notional = intent.target_notional
        adjusted_quantity = intent.target_quantity
        if state.gross_exposure + adjusted_notional > capacity_cap:
            reasons.append("capacity_ceiling_breach")
        if adjusted_notional > state.nav * self.config.max_single_position_fraction:
            reasons.append("single_position_limit_breach")

        sector_after = state.sector_exposures.get(sector, 0.0) + adjusted_notional
        if sector_after > state.nav * self.config.max_sector_fraction:
            reasons.append("sector_concentration_breach")

        for existing_symbol in state.current_positions:
            correlation = self._lookup_correlation(state, intent.symbol, existing_symbol)
            if correlation > self.config.max_correlation:
                reasons.append(f"correlation_limit_breach:{existing_symbol}")
                break

        participation_limit = self._participation_limit(intent.symbol)
        if avg_daily_volume > 0:
            max_qty = avg_daily_volume * participation_limit
            if adjusted_quantity > max_qty:
                warnings.append("participation_limit_clamped")
                adjusted_quantity = max_qty
                adjusted_notional = adjusted_quantity * float(intent.metadata["mark_price"])

        if intent.hedge_required and intent.hedge_symbol:
            hedge_actions.append(f"open_hedge:{intent.hedge_symbol}")
        if risk_mode == RiskMode.CLOSE_ONLY:
            adjusted_quantity = 0.0
            adjusted_notional = 0.0
            warnings.append("close_only_blocks_new_exposure")

        approved = not reasons
        return PortfolioCheckResult(
            approved=approved,
            risk_mode=risk_mode,
            adjusted_quantity=adjusted_quantity,
            adjusted_notional=adjusted_notional,
            rejection_reasons=tuple(reasons),
            warnings=tuple(warnings),
            hedge_actions=tuple(hedge_actions),
            metadata={"participation_limit": participation_limit, "sector": sector},
        )

    def build_hedge_recommendation(self, *, symbol: str, risk_mode: RiskMode) -> str | None:
        if risk_mode == RiskMode.NORMAL:
            return None
        hedge_symbol = self._default_hedge_symbol(symbol)
        if risk_mode == RiskMode.CLOSE_ONLY:
            return f"emergency_neutralize:{hedge_symbol}"
        return f"partial_unwind_with_hedge:{hedge_symbol}"

    def _uncertainty_scalar(self, forecast_uncertainty: float) -> float:
        clipped = min(max(forecast_uncertainty, 0.0), 1.0)
        scale = 1.0 - clipped
        return min(self.config.uncertainty_size_ceiling, max(self.config.uncertainty_size_floor, scale))

    def _lookup_correlation(self, state: PortfolioState, symbol_a: str, symbol_b: str) -> float:
        if symbol_a == symbol_b:
            return 1.0
        return abs(
            state.correlations.get((symbol_a, symbol_b), state.correlations.get((symbol_b, symbol_a), 0.0))
        )

    def _default_hedge_symbol(self, symbol: str) -> str:
        if "USDINR" in symbol.upper():
            return "USDINR_FUT"
        if "GOLD" in symbol.upper():
            return "MCX_GOLDM"
        return "NIFTY_FUT"

    def _participation_limit(self, symbol: str) -> float:
        upper = symbol.upper()
        if "USDINR" in upper:
            return self.config.participation_limit_fx
        if "GOLD" in upper:
            return self.config.participation_limit_gold
        if any(tag in upper for tag in ("BANK", "RELIANCE", "INFY", "TCS", "HDFCBANK")):
            return self.config.participation_limit_large_cap
        return self.config.participation_limit_mid_liquidity


def round_to_tradable_quantity(quantity: float, *, lot_size: int = 1) -> int:
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if quantity <= 0:
        return 0
    lots = ceil(quantity / lot_size)
    return int(lots * lot_size)
