from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.agents.strategic.policy_manager import RiskMode
from src.agents.strategic.schemas import ActionType


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    avg_entry_price: float
    market_price: float
    sector: str
    asset_class: str
    hedge_ratio: float = 0.0

    @property
    def net_notional(self) -> float:
        return float(self.quantity * self.market_price)

    @property
    def gross_notional(self) -> float:
        return float(abs(self.net_notional))

    @property
    def hedge_adjusted_notional(self) -> float:
        clipped = min(max(float(self.hedge_ratio), 0.0), 1.0)
        return float(self.net_notional * (1.0 - clipped))


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    mode: RiskMode = RiskMode.NORMAL

    def gross_exposure(self) -> float:
        return float(sum(pos.gross_notional for pos in self.positions.values()))

    def net_exposure(self) -> float:
        return float(sum(pos.net_notional for pos in self.positions.values()))

    def hedge_adjusted_exposure(self) -> float:
        return float(sum(pos.hedge_adjusted_notional for pos in self.positions.values()))

    def sector_exposure(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for pos in self.positions.values():
            totals[pos.sector] = totals.get(pos.sector, 0.0) + pos.gross_notional
        return totals


@dataclass(frozen=True)
class RiskBudgetConfig:
    initial_notional_cap_inr: float = 100_000_000.0  # ₹10 Cr
    max_notional_cap_inr: float = 250_000_000.0  # ₹25 Cr after validation window
    max_single_symbol_pct: float = 0.25
    max_sector_pct: float = 0.35
    max_correlation: float = 0.85
    hedge_trigger_pct: float = 0.75
    elevated_vol_cap_pct: float = 0.70
    protective_vol_cap_pct: float = 0.30
    reduce_only_cap_pct: float = 0.50
    close_only_cap_pct: float = 0.0
    stress_impact_cap_pct: float = 0.25
    participation_limit_by_bucket: Mapping[str, float] = field(
        default_factory=lambda: {
            "liquid": 0.10,
            "mid": 0.05,
            "fx": 0.08,
            "commodity": 0.06,
        }
    )

    def __post_init__(self) -> None:
        if self.initial_notional_cap_inr <= 0.0:
            raise ValueError("initial_notional_cap_inr must be > 0")
        if self.max_notional_cap_inr < self.initial_notional_cap_inr:
            raise ValueError("max_notional_cap_inr must be >= initial_notional_cap_inr")
        if not 0.0 < self.max_single_symbol_pct <= 1.0:
            raise ValueError("max_single_symbol_pct must be within (0, 1]")
        if not 0.0 < self.max_sector_pct <= 1.0:
            raise ValueError("max_sector_pct must be within (0, 1]")
        if not 0.0 < self.max_correlation <= 1.0:
            raise ValueError("max_correlation must be within (0, 1]")


@dataclass(frozen=True)
class SizingRequest:
    symbol: str
    sector: str
    asset_class: str
    direction: ActionType
    price: float
    forecast_confidence: float
    forecast_uncertainty: float
    liquidity_score: float
    adv_notional: float
    participation_bucket: str = "liquid"
    proposed_notional: float | None = None
    correlations: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HedgeInstruction:
    action: ActionType
    symbol: str
    quantity: float
    reason: str


@dataclass(frozen=True)
class SizingDecision:
    accepted: bool
    action: ActionType
    target_quantity: float
    target_notional: float
    max_participation_notional: float
    risk_budget_used_pct: float
    reject_reason: str | None = None
    hedge_instruction: HedgeInstruction | None = None


class PortfolioConstructor:
    """
    Week 2 Day 5 portfolio construction with risk-budget enforcement.

    Includes concentration/correlation gates, capacity ceilings, and
    exposure-aware hedging instructions.
    """

    def __init__(
        self,
        *,
        config: RiskBudgetConfig | None = None,
        state: PortfolioState | None = None,
    ):
        self.config = config or RiskBudgetConfig()
        self.state = state or PortfolioState(cash=0.0)

    def set_mode(self, mode: RiskMode) -> None:
        self.state.mode = mode

    def current_notional_cap(
        self,
        *,
        volatility_state: str = "normal",
        stress_impact_pct_at_3x: float | None = None,
    ) -> float:
        base = float(self.config.initial_notional_cap_inr)
        if volatility_state.strip().lower() == "elevated":
            base *= self.config.elevated_vol_cap_pct
        elif volatility_state.strip().lower() in {"high", "protective", "extreme"}:
            base *= self.config.protective_vol_cap_pct

        if self.state.mode == RiskMode.REDUCE_ONLY:
            base *= self.config.reduce_only_cap_pct
        elif self.state.mode in {RiskMode.CLOSE_ONLY, RiskMode.KILL_SWITCH}:
            base *= self.config.close_only_cap_pct

        if stress_impact_pct_at_3x is not None and float(stress_impact_pct_at_3x) > self.config.stress_impact_cap_pct:
            # Hard cap at 2x initial if 3x impact exceeds 0.25%.
            max_cap = min(self.config.max_notional_cap_inr, self.config.initial_notional_cap_inr * 2.0)
        else:
            max_cap = self.config.max_notional_cap_inr
        return float(min(base, max_cap))

    def propose(
        self,
        request: SizingRequest,
        *,
        volatility_state: str = "normal",
        stress_impact_pct_at_3x: float | None = None,
    ) -> SizingDecision:
        if request.price <= 0.0:
            return self._reject(request.direction, "invalid_price")
        if request.adv_notional <= 0.0:
            return self._reject(request.direction, "invalid_adv_notional")

        if self.state.mode in {RiskMode.CLOSE_ONLY, RiskMode.KILL_SWITCH} and request.direction in {
            ActionType.BUY,
            ActionType.SELL,
        }:
            return self._reject(request.direction, f"mode_{self.state.mode.value}_blocks_new_opens")

        if self.state.mode == RiskMode.REDUCE_ONLY and request.direction in {ActionType.BUY, ActionType.SELL}:
            return self._reject(request.direction, "reduce_only_blocks_new_opens")

        cap = self.current_notional_cap(
            volatility_state=volatility_state,
            stress_impact_pct_at_3x=stress_impact_pct_at_3x,
        )

        target_notional = self._target_notional(request, cap=cap)
        participation_cap = self._participation_cap(request)
        target_notional = min(target_notional, participation_cap)

        if target_notional <= 0.0:
            return self._reject(request.direction, "target_notional_zero_after_caps")

        corr_violation = self._find_correlation_violation(request.correlations)
        if corr_violation is not None:
            return self._reject(request.direction, f"correlation_limit_exceeded:{corr_violation}")

        projected_gross = self.state.gross_exposure() + target_notional
        if projected_gross > cap:
            return self._reject(request.direction, "portfolio_capacity_exceeded")

        symbol_notional = self.state.positions.get(request.symbol).gross_notional if request.symbol in self.state.positions else 0.0
        if symbol_notional + target_notional > cap * self.config.max_single_symbol_pct:
            return self._reject(request.direction, "single_symbol_concentration_exceeded")

        sector_exposure = self.state.sector_exposure().get(request.sector, 0.0)
        if sector_exposure + target_notional > cap * self.config.max_sector_pct:
            return self._reject(request.direction, "sector_concentration_exceeded")

        quantity = target_notional / request.price
        risk_budget_used = projected_gross / max(cap, 1e-9)
        hedge_instruction = self._hedge_if_needed(
            request=request,
            cap=cap,
            projected_net=self._projected_net_exposure(request, target_notional),
        )

        return SizingDecision(
            accepted=True,
            action=request.direction,
            target_quantity=float(quantity),
            target_notional=float(target_notional),
            max_participation_notional=float(participation_cap),
            risk_budget_used_pct=float(risk_budget_used),
            hedge_instruction=hedge_instruction,
        )

    def apply_fill(
        self,
        *,
        symbol: str,
        side: ActionType,
        quantity: float,
        fill_price: float,
        sector: str,
        asset_class: str,
        hedge_ratio: float = 0.0,
    ) -> None:
        signed_qty = float(quantity)
        if side == ActionType.SELL:
            signed_qty *= -1.0
        elif side in {ActionType.CLOSE, ActionType.REDUCE}:
            signed_qty *= -1.0

        existing = self.state.positions.get(symbol)
        if existing is None:
            new_qty = signed_qty
            avg_price = fill_price
        else:
            new_qty = existing.quantity + signed_qty
            if abs(new_qty) < 1e-9:
                self.state.positions.pop(symbol, None)
                return
            weighted_notional = (existing.quantity * existing.avg_entry_price) + (signed_qty * fill_price)
            avg_price = weighted_notional / new_qty

        self.state.positions[symbol] = PositionSnapshot(
            symbol=symbol,
            quantity=new_qty,
            avg_entry_price=float(avg_price),
            market_price=float(fill_price),
            sector=sector,
            asset_class=asset_class,
            hedge_ratio=hedge_ratio,
        )

    def snapshot(self) -> dict[str, float | str]:
        return {
            "mode": self.state.mode.value,
            "gross_exposure": self.state.gross_exposure(),
            "net_exposure": self.state.net_exposure(),
            "hedge_adjusted_exposure": self.state.hedge_adjusted_exposure(),
            "position_count": float(len(self.state.positions)),
        }

    def _target_notional(self, request: SizingRequest, *, cap: float) -> float:
        if request.proposed_notional is not None:
            return float(max(0.0, request.proposed_notional))

        confidence = min(max(float(request.forecast_confidence), 0.0), 1.0)
        uncertainty = min(max(float(request.forecast_uncertainty), 0.0), 1.0)
        liquidity = min(max(float(request.liquidity_score), 0.0), 1.0)

        signal_strength = confidence * (1.0 - uncertainty) * liquidity
        # Limit any single sizing proposal to 10% of current cap before further checks.
        return float(cap * min(signal_strength, 0.10))

    def _participation_cap(self, request: SizingRequest) -> float:
        bucket = request.participation_bucket.strip().lower()
        limit = self.config.participation_limit_by_bucket.get(bucket)
        if limit is None:
            limit = self.config.participation_limit_by_bucket.get("liquid", 0.05)
        return float(request.adv_notional * float(limit))

    def _find_correlation_violation(self, correlations: Mapping[str, float]) -> str | None:
        for symbol, corr in correlations.items():
            if symbol not in self.state.positions:
                continue
            if abs(float(corr)) > self.config.max_correlation:
                return symbol
        return None

    def _projected_net_exposure(self, request: SizingRequest, target_notional: float) -> float:
        projected = self.state.net_exposure()
        if request.direction == ActionType.BUY:
            projected += target_notional
        elif request.direction == ActionType.SELL:
            projected -= target_notional
        elif request.direction in {ActionType.CLOSE, ActionType.REDUCE}:
            projected *= 0.8
        return float(projected)

    def _hedge_if_needed(
        self,
        *,
        request: SizingRequest,
        cap: float,
        projected_net: float,
    ) -> HedgeInstruction | None:
        threshold = cap * self.config.hedge_trigger_pct
        if abs(projected_net) <= threshold:
            return None

        # Hedge by reducing the requested symbol first (explicit risk unwind).
        hedge_qty = max(0.0, abs(projected_net) - threshold) / max(request.price, 1e-9)
        return HedgeInstruction(
            action=ActionType.REDUCE,
            symbol=request.symbol,
            quantity=float(hedge_qty),
            reason="projected_net_exposure_above_hedge_trigger",
        )

    def _reject(self, action: ActionType, reason: str) -> SizingDecision:
        return SizingDecision(
            accepted=False,
            action=action,
            target_quantity=0.0,
            target_notional=0.0,
            max_participation_notional=0.0,
            risk_budget_used_pct=0.0,
            reject_reason=reason,
        )
