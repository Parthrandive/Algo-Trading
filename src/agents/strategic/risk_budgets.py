from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Mapping

from src.agents.strategic.config import RiskBudgetConfig


class VolatilityRegime(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True)
class VolatilityReading:
    symbol: str
    asset_cluster: str
    realized_vol: float
    timestamp: datetime


@dataclass(frozen=True)
class RiskBudgetDecision:
    symbol: str
    asset_cluster: str
    regime: VolatilityRegime
    cap_fraction: float
    changed: bool
    event_type: str
    false_trigger_rate: float
    auto_adjustment_paused: bool


@dataclass
class _RiskState:
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    cap_fraction: float = 1.0
    updated_at: datetime | None = None


class VolatilityScaledRiskBudgetEngine:
    """
    Tier 1-B volatility-scaled risk budget engine.
    Applies deterministic cap transitions and exposes false-trigger telemetry.
    """

    def __init__(
        self,
        sigma_baseline_by_cluster: Mapping[str, float],
        *,
        config: RiskBudgetConfig | None = None,
    ) -> None:
        if not sigma_baseline_by_cluster:
            raise ValueError("sigma_baseline_by_cluster must not be empty")
        self.sigma_baseline_by_cluster = {
            str(cluster): float(value) for cluster, value in sigma_baseline_by_cluster.items()
        }
        if any(value <= 0.0 for value in self.sigma_baseline_by_cluster.values()):
            raise ValueError("all sigma baselines must be > 0")

        self.config = config or RiskBudgetConfig()
        self._state: dict[str, _RiskState] = {}
        self._trigger_history: deque[tuple[datetime, bool]] = deque()
        self._events: list[dict[str, object]] = []

    def update(self, reading: VolatilityReading) -> RiskBudgetDecision:
        now = reading.timestamp.astimezone(UTC)
        baseline = self.sigma_baseline_by_cluster[reading.asset_cluster]
        sigma_ratio = float(reading.realized_vol) / baseline
        state = self._state.setdefault(reading.symbol, _RiskState())
        paused = self.auto_adjustment_paused(now=now)

        next_regime = self._classify_regime(sigma_ratio)
        next_cap = self._cap_for_regime(next_regime)
        changed = next_regime != state.regime or abs(next_cap - state.cap_fraction) > 1e-9

        if paused:
            changed = False
            event_type = "RISK_CAP_PAUSED"
            next_regime = state.regime
            next_cap = state.cap_fraction
        else:
            event_type = "RISK_CAP_CHANGE" if changed else "RISK_CAP_STEADY"
            state.regime = next_regime
            state.cap_fraction = next_cap
            state.updated_at = now

        self._events.append(
            {
                "timestamp": now.isoformat(),
                "symbol": reading.symbol,
                "asset_cluster": reading.asset_cluster,
                "sigma_ratio": sigma_ratio,
                "regime": next_regime.value,
                "cap_fraction": next_cap,
                "event_type": event_type,
            }
        )

        return RiskBudgetDecision(
            symbol=reading.symbol,
            asset_cluster=reading.asset_cluster,
            regime=next_regime,
            cap_fraction=next_cap,
            changed=changed,
            event_type=event_type,
            false_trigger_rate=self.false_trigger_rate(now=now),
            auto_adjustment_paused=paused,
        )

    def register_trigger_outcome(self, *, timestamp: datetime, was_false_trigger: bool) -> None:
        now = timestamp.astimezone(UTC)
        self._trigger_history.append((now, bool(was_false_trigger)))
        self._trim_trigger_history(now=now)

    def current_cap(self, symbol: str) -> float:
        return self._state.get(symbol, _RiskState()).cap_fraction

    def false_trigger_rate(self, *, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        self._trim_trigger_history(now=now)
        if not self._trigger_history:
            return 0.0
        false_count = sum(1 for _, is_false in self._trigger_history if is_false)
        return false_count / len(self._trigger_history)

    def auto_adjustment_paused(self, *, now: datetime | None = None) -> bool:
        rate = self.false_trigger_rate(now=now or datetime.now(UTC))
        return rate > float(self.config.false_trigger_acceptance_limit)

    def recent_events(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        return tuple(self._events[-max(limit, 0) :])

    def _trim_trigger_history(self, *, now: datetime) -> None:
        horizon = now - timedelta(days=int(self.config.false_trigger_rolling_days))
        while self._trigger_history and self._trigger_history[0][0] < horizon:
            self._trigger_history.popleft()

    @staticmethod
    def _classify_regime(sigma_ratio: float) -> VolatilityRegime:
        if sigma_ratio <= 1.0:
            return VolatilityRegime.NORMAL
        if sigma_ratio <= 1.5:
            return VolatilityRegime.ELEVATED
        if sigma_ratio <= 2.0:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME

    def _cap_for_regime(self, regime: VolatilityRegime) -> float:
        if regime == VolatilityRegime.NORMAL:
            return float(self.config.normal_cap)
        if regime == VolatilityRegime.ELEVATED:
            return float(self.config.elevated_cap)
        if regime == VolatilityRegime.HIGH:
            return float(self.config.high_cap)
        return float(self.config.extreme_cap)
